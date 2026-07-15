# VDA-HyperQuant — Optimization Ledger

Maintained by the Lead Architect (Fable). Each entry is an atomic task with strict
mathematical boundary conditions and assertion criteria. Implementation is delegated
to Sonnet per CLAUDE.md §3. Status: `OPEN` → `BUILT` → `VALIDATED` → `MERGED`.

Audit baseline: commit `68bb5a2`, 2026-07-15.

---

## Audit verdict (2026-07-15)

Core math (RHT forward/inverse, weight absorption, D4 decoder, rotated-attention
basis consistency) verified correct. Blocking findings, in priority order:

| ID | Severity | Finding |
|----|----------|---------|
| F1 | P0 | Bit accounting omits per-group fp16 scales (+4 b/scalar) ; QJL side-channel (272 b/vec at m=256) exceeds the 3-bit payload (192 b/vec). True effective rate at "3-bit": 7.0 b/scalar (no QJL), 11.25 b/scalar (QJL). Ratios quoted vs FP32; deployment baseline is fp16. |
| F2 | P0 | Benchmark "datasets" are proxy videos/synthetic pan-zoom frames; metrics are fidelity-vs-FP32, not ground truth; no scale-shift alignment; `PUBLISHED_BASELINES` numbers unverifiable. Not publishable as dataset results. |
| F3 | P0 | QJL estimator uses raw sign-correlation ρ̂ = (1/m)·sign(RQ)ᵀsign(Rε) as cos θ. E[ρ̂] = 1 − 2θ/π (SimHash), not cos θ. Requires ρ̂ → cos(π(1−ρ̂)/2). Max systematic error ≈ 0.21. |
| F4 | P0 | Temporal surgery disabled (`replace_temporal=False`); no TAE metric exists; temporal-consistency claims unfalsifiable. |
| F5 | P1 | D4 `effective_bits = bits − 0.25` unearned (no index coding); clamp after parity correction can break lattice membership; sign()==0 tie leaves odd parity. |
| F6 | P1 | `tests/test_core.py` has zero asserts — `pytest tests/` passes vacuously; the §3 merge gate is not enforced. |
| F7 | P1 | `fast_hadamard_transform` mutates caller's tensor in-place when d is a power of 2; two clones per butterfly level. |
| F8 | P1 | `--use_qjl` cannot be disabled (`store_true` + `default=True`); QJL ablation impossible. QJL correction costs 4× score-matmul FLOPs at m=4d. |
| F9 | P2 | RotatedSelfAttention silently ignores attn_bias/mask kwargs. |

Target after remediation: **≤ 4.0 effective bits/scalar all-inclusive**
(8× vs FP32, 4× vs fp16) with GT-verified δ1 drop ≤ 0.02 vs FP32 on real NYUv2,
plus a measured TAE result. That is the NeurIPS claim.

---

## T1 — Honest bit accounting (F1) — OPEN
File: `scripts/run_pareto_benchmark_suite.py` (`compute_real_bit_accounting`)
- Must count: payload `d·b`, group scales `(d/g)·s_bits`, QJL `(m + norm_bits)` when enabled.
- Parameterize baseline: `baseline_bits ∈ {16, 32}`; report both.
- Assertions: for d=64, g=4, b=3, s_bits=16, QJL(m=256, norm=16):
  total == 720 bits, eff == 11.25 b/scalar, ratio_fp32 == 2.8×, ratio_fp16 == 1.4×.
  Without QJL: total == 448, eff == 7.0.

## T2 — Real ground-truth evaluation protocol (F2) — OPEN
Files: new `scripts/datasets_gt.py`, edits in `scripts/run_pareto_benchmark_suite.py`
- NYUv2 official labeled test split (654 imgs); affine-invariant protocol:
  per-image least-squares (s, t) alignment of predicted inverse depth to GT
  before AbsRel/δk, GT valid mask 0.1–10 m.
- Quarantine `PUBLISHED_BASELINES` (move to `docs/`; never print beside measured rows).
- Keep fidelity-vs-FP32 as a separate explicitly-labeled mode.
- Assertions: identity prediction (pred == gt) → AbsRel < 1e-6, δ1 == 1.0;
  pred = 2·gt + 3 (inverse-depth affine) → post-alignment AbsRel < 1e-4.

## T3 — QJL estimator + overhead fix (F3, F8) — BUILT (see F3b below)
File: `research/quantizers/qjl_bias.py`, argparse in both scripts
- Apply ρ̂ → cos(π/2·(1−clamp(ρ̂,−1,1))) in `correct_scores`. DONE.
- Default m: `min(2·d, 128)` for d ≤ 128 (`default_qjl_projections`, shared with
  `compute_real_bit_accounting` so the two can never silently disagree); norm_bits
  8-bit option added (per-call max, uint8). DONE.
- Replaced `--use_qjl` with `--qjl/--no-qjl` (BooleanOptionalAction) in both scripts. DONE.
- Assertions: E[transform(ρ̂)] within ±0.05 of true cos θ at θ ∈ {30°, 45°, 60°, 90°}
  over 2000 trials, m=128, d=64 — PASSES. corrected-score MAE strictly < uncorrected
  at 3-bit — see F3b, only holds at m≳256, not at the T3 default (m=128).

### F3b (discovered while validating T3) — bias-variance tradeoff in the fix
Empirically, after RHT, Q and the per-vector quantization error ε are close to
ORTHOGONAL (measured θ ≈ 90.9° ± a few degrees on synthetic ViT-Small-shaped
d=64 data). `cos(π/2·(1−ρ̂))` has its steepest derivative (π/2) exactly at ρ̂=0
(θ=90°) — precisely the regime that dominates this data. Fixing the estimator's
systematic bias therefore AMPLIFIES its finite-sample variance exactly where it
matters most. Consequence, measured on 3-bit `ScalarRoundQuantizer` + RHT data
(see `tests/test_core.py::test_qjl_default_projections_insufficient_at_low_m`
and `test_qjl_bias_correction`):
  - At the T3-prescribed default (m=128 for d=64): corrected estimator is
    WORSE than applying no correction at all (~7–18% worse MAE across seeds).
  - Crossover to net-positive vs. no correction: roughly m∈[200, 256].
  - Crossover to beating the OLD (mathematically wrong, lower-variance)
    raw-ρ̂ estimator: roughly m∈[700, 1000].
  - At m=4096: corrected MAE ≈0.36 vs raw-ρ̂'s ≈0.70 — the fix is asymptotically
    correct and strictly better, it just needs far more projections than the
    "honest accounting" bit budget can afford.
Implication: QJL is NOT viable in the ≤4-bit-effective regime this project
targets — its own side-channel at m≥256 (272+ bits/vector) already dwarfs a
3-bit payload (192 bits/vector) per F1, and now F3b shows smaller m isn't even
net-positive once the estimator is made honest. This is exactly why T8's
headline low-bit configuration already specifies `use_qjl=False` — treat that
as confirmed, not just convenient. If QJL is used at all (e.g. an ablation
table entry at 8-bit), use m≥512 and report its true cost via T1's accounting.

## T4 — D4 honesty + boundary correctness (F5) — OPEN
File: `research/quantizers/lattice_vq.py`
- Remove `effective_bits = bits − 0.25` unless true index coding is implemented;
  report `bits` plus separately-computed scale overhead in info dict.
- Clamp BEFORE parity decode (shrink round target), so returned point is always in D4
  within range; deterministic tie-break for zero residual (flip → +1).
- Add `scale_bits` option (16 or 8); 8-bit scales quantized with per-tensor scale.
- Assertions: for 10^5 random 4-vectors incl. values at ±(half_levels·1.5):
  `int(z.sum()) % 2 == 0` for 100% of outputs AND all coords within
  [−half_levels, half_levels−1]; MSE(D4) < MSE(scalar RTN) at equal b on
  N(0,1) inputs (validates the coding-gain direction empirically).

## T5 — Real pytest gate (F6) — OPEN
File: `tests/test_core.py`
- Every printed ✓/✗ becomes `assert` with the same tolerance; keep prints.
- Tolerances: Hadamard orthogonality max-err < 1e-4 (d ≤ 256); FHT-vs-matrix < 1e-4;
  RHT round-trip < 1e-4 (fp32); outlier max/mean reduction > 3×; QJL improvement > 0.
- Must fail: `pytest tests/ -x` exits nonzero if any tolerance is violated
  (verify by temporarily corrupting a sign and confirming failure, then revert).

## T6 — FHT purity + performance (F7) — OPEN
File: `research/transforms/hadamard.py`
- `fast_hadamard_transform` must never mutate its input: clone-on-entry when no
  padding occurs, or rewrite butterfly out-of-place via `torch.stack([a+b, a−b], dim=-2)`
  reshape (preferred: also removes 2 clones/level).
- Compute butterfly in fp32 when input is fp16/bf16; cast back on return.
- Assertions: input tensor bit-identical before/after call (compare against
  pre-call clone) for d ∈ {64, 384}; output matches `x @ H.T/√d` < 1e-4 (fp32)
  and < 2e-2 (fp16 in, fp16 out); round-trip fp16 max-err < 1e-2.

## T7 — Temporal surgery + TAE metric (F4, F9) — BUILT
Files: `research/models/rotated_attention.py`, `research/quantizers/lattice_vq.py`,
`scripts/run_pareto_benchmark_suite.py`, `tests/test_temporal_equivalence.py`,
`tests/test_tae_metric.py`

### Root cause found (NOT what commit 4cc719f's message claimed)
Video-Depth-Anything got cloned locally as a side effect of this session's earlier
test runs (`run_pareto_benchmark_suite.py` auto-clones at MODULE IMPORT time if
missing — a design smell worth fixing separately; flagged to the user). This let
T7 be validated against the REAL `TemporalAttention` class instead of a mock.

Tracing `TemporalAttention.forward` (`Video-Depth-Anything/video_depth_anything/
motion_module/motion_module.py`) line-by-line against `RotatedTemporalAttention`
shows the outer video↔spatial rearrange AND the inner per-head attention math are
mathematically IDENTICAL — `(B,h,N,d)` batched matmul is provably equivalent to
AnimateDiff's fold-heads-into-batch `(B·h,N,d)` convention when the reshape/permute
order matches (which it already did). The "reshape incompatibility" blamed for the
garbled output was a red herring.

The REAL bug: `TemporalAttention` inherits `CrossAttention`'s `bias=False` default
for `to_q`/`to_k`/`to_v` (never overridden anywhere in VDA's source — grepped for
`bias` in `motion_module.py`, zero hits). The surgery in
`apply_rotated_quantization_to_vda` never checked this — `RotatedTemporalAttention`
was always constructed with `qkv_bias` defaulting to `True`, and only `.weight` was
ever copied, never `.bias`. The replacement's Q/K/V projections therefore carried a
random, untrained bias vector the original layer never had.

**Verified empirically against the real class** (`d=64, heads=8`, identity quantizer,
QJL off, random weights, same seed):
- Old behavior (`qkv_bias=True`, mismatched): max abs error vs real module = **0.1008**
  — easily enough to garble a depth map.
- Fixed (`qkv_bias` detected from source and matched, bias copied when present):
  max abs error = **0.000000** (exact float match).

### Fixes applied
- `apply_rotated_quantization_to_vda`: detects `qkv_has_bias = weight_attr.bias is
  not None` from the source module and passes `qkv_bias=qkv_has_bias`; copies
  `.bias.data` for q/k/v when present (previously only `.weight` was ever copied).
- `replace_temporal=True` re-enabled in both call sites in
  `scripts/run_pareto_benchmark_suite.py` (fidelity loop and groundtruth loop).
- Added `IdentityQuantizer` (`research/quantizers/lattice_vq.py`) and `'identity'`
  quantizer choice, enabling the quantization-bypassed equivalence gate T7 specified.
- `RotatedSelfAttention.forward` raises `NotImplementedError` for non-None
  `attn_bias`/`attention_mask`/`mask` kwargs (F9) — tested in `test_core.py`.
- `compute_tae` implemented in `scripts/run_pareto_benchmark_suite.py`: affine-aligned
  `mean |D_t − D_{t+1}|` static-camera fallback (default), with an optional
  `flow_fields` argument for true `warp(D_{t+1})` motion compensation when available
  (no optical-flow estimator is run by this repo, so the static fallback is what's
  actually reported). Wired into both FP32 baseline and each bit-width in the suite's
  fidelity loop; printed in the summary table.

### Assertions (all passing)
- `tests/test_temporal_equivalence.py` (skips gracefully if the VDA repo isn't
  cloned): non-cached and cached/streaming calls match the real `TemporalAttention`
  within 1e-3 (measured: exact 0.0) with the bias fix; a regression guard confirms
  the OLD `qkv_bias=True` mismatch reproduces a clear (>1e-3) divergence; a portable
  mock-based test (no VDA needed) confirms the surgery itself detects `bias=False`.
- `tests/test_tae_metric.py`: TAE=0 for a static sequence; matches manual frame-diff
  when unaligned; per-pair affine alignment removes benign global scale drift
  (near-0 TAE) while raw unaligned diff stays large; single-frame sequence → 0;
  batched input averages correctly; zero optical flow is a no-op warp; a matching
  flow field reduces TAE for a pure-translation synthetic sequence.

### Caveat
Equivalence was verified against `TemporalAttention` constructed with random weights
and small synthetic dims (CPU, no checkpoint, no xformers) — this validates the
RESHAPE/BIAS CONTRACT, not full numerical parity under a real checkpoint + GPU +
xformers memory-efficient-attention path (untested; that path uses
`reshape_heads_to_4d` + `_memory_efficient_attention_xformers`, algebraically
equivalent but not exercised here). Re-run the Colab suite's
`verify_quantization_surgery` sanity check after pulling to confirm on the real
checkpoint before trusting TAE/accuracy numbers from the temporal decoder.

## T8 — Path to a true ≤4.0 effective bits/scalar (paper claim) — BUILT
Files: `research/quantizers/lattice_vq.py` (`LatticeE8Quantizer`),
`research/models/rotated_attention.py` (`scale_bits` threaded through
`_get_quantizer`/`RotatedSelfAttention`/`RotatedTemporalAttention`/
`apply_rotated_quantization_to_vda`), `scripts/run_pareto_benchmark_suite.py`
(`--quantizer`, `--scale-bits` CLI flags, `QUANTIZER_GROUP_SIZE`),
`scripts/run_vda_quant_eval.py` (same flags), `tests/test_e8_quantizer.py`.

- E8 lattice (Λ8 = D8 ∪ D8+½) implemented via Conway & Sloane's best-of-two-cosets
  decoder: candidate A = nearest D8 point (reusing D4's parity-fix algorithm
  generalized to 8 dims); candidate B = nearest D8 point to (x−½) shifted back
  by +½; return whichever is closer. group_size=8 halves the per-group scale
  overhead vs D4's group_size=4 — the key lever for the ≤4-bit target.
- Headline config confirmed exactly via `compute_real_bit_accounting`
  (d=64, group=8, b=3, scale_bits=8, no QJL): 64·3 + (64/8)·8 = 192+64 =
  **256 bits/vector = exactly 4.0 effective bits/scalar** (test:
  `test_bit_accounting_e8_target`). Reachable end-to-end via
  `--quantizer lattice_e8 --scale-bits 8 --no-qjl --bits 3`.
- Same pre-decode clamping discipline as D4 (T4): `x_scaled` clamped to
  `[-half_levels+1.0, half_levels-1.5]` before either coset decode, guaranteeing
  both the integer coset AND the half-integer-coset decode input stay within
  the boundary-safe range — no post-hoc clamp needed (verified: 100% valid
  lattice membership + even parity over 10,000 random 8-vectors, including an
  extreme-outlier case, plus a targeted boundary-tie regression test).

### Measured results (`tests/test_e8_quantizer.py`, N(0,1) input, 4-bit, seed 0)
| Method | MSE | 
|---|---|
| Scalar (RTN) | 0.036848 |
| D4 | 0.007258 |
| E8 | 0.006223 |

**MSE(E8) ≤ MSE(D4) ≤ MSE(scalar)** confirmed. Measured coding gain **E8 vs D4 =
0.67 dB** — matches Conway & Sloane's theoretical 0.65 dB almost exactly. (E8 vs
scalar = 7.72 dB here, well above the oft-quoted 1.5 dB "theory" figure — that
figure is an asymptotic high-rate constant relative to an idealized scalar
quantizer, not this specific 4-bit RTN baseline on this distribution; the
E8-vs-D4 differential, which cancels most of that modeling mismatch, is the
number that validates the implementation.)

### Not done in this session (explicitly deferred, per instruction not to
### download datasets or run GPU work locally)
The T8 "suite gate" (full pipeline at b=3/E8/8-bit-scales/no-QJL reporting
`eff == 4.0` AND GT δ1 drop ≤ 0.02 on real NYUv2 via T2's protocol) requires a
real VDA checkpoint + the NYUv2 labeled dataset + GPU inference — none of which
were run locally. All the code paths are wired and unit-tested; running
`python scripts/run_pareto_benchmark_suite.py --dataset nyuv2 --eval-mode
groundtruth --bits 3 --quantizer lattice_e8 --scale-bits 8 --no-qjl` on Colab
is the remaining step to produce the actual headline-table numbers.

---

Execution order: T5 → T6 → T1 → T4 → T3 → T2 → T7 → T8.
(T5/T6 first: nothing else is trustworthy until the test gate is real and the
transform is pure.)
