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

### [SUPERSEDED by F10 — used broken metric-space alignment; see VALIDATED NYU below] GATE PASSED — real Colab run, N=200 NYUv2 ground-truth images, T8 headline result
`python scripts/run_pareto_benchmark_suite.py --dataset nyuv2 --eval-mode groundtruth
--bits 3 --quantizer lattice_e8 --scale-bits 8 --no-qjl --max-samples 200`

Surgery verified active before trusting these numbers (not a no-op): 12 backbone
+ 8 temporal attention layers replaced; FP32-vs-quantized activation diff
MSE=0.2854, max abs diff=4.95 (`verify_quantization_surgery`, added this session
specifically to rule out a silent-no-op explanation for suspiciously-small
accuracy deltas — see the N=20 smoke-test run that preceded this one).

| | δ1 ↑ | δ2 | δ3 | AbsRel ↓ | RMSE ↓ | eff bits/scalar | vs FP32 | vs FP16 |
|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 0.8105 | 0.9482 | 0.9741 | 0.1513 | 0.5477 | 32.0 | 1.0x | 0.5x |
| 3-bit E8 (T8 config) | 0.8044 | 0.9444 | 0.9741 | 0.1523 | 0.5417 | **4.0** | **8.0x** | **4.0x** |

**δ1 drop = 0.0061** (0.8105 → 0.8044) — more than 3x under the ≤0.02 gate.
AbsRel moved +0.0010 (negligible); RMSE and δ3 essentially unchanged/improved.
**Gate: PASSED.** eff == 4.0 exactly; GT δ1 drop ≤ 0.02.

This is the paper's headline result: **8x real compression vs FP32 (4x vs the
honest FP16 deployment baseline), on real NYUv2 ground truth (not the
fidelity-vs-FP32 proxy the original code reported), with δ1 degrading by only
0.61 points.**

Remaining before treating this as final for publication:
- N=200 of 654 in the official test split — re-run with `--max-samples 654`
  (full split) for the paper's real number; N=200 is a strong signal but not
  the complete evaluation.
- This run used QJL disabled per the T8 config; an ablation row with QJL on
  (at whatever m makes F3b's bias-variance tradeoff net-positive, m≳512) would
  quantify what QJL costs/buys at this operating point, if wanted for the paper.
- Only bits=3 was swept here; run `--bits 8 4 3 2` for the full Pareto curve
  the suite already produces (`generate_pareto_charts`).

### [SUPERSEDED by F10 — broken metric-space alignment; see VALIDATED NYU below] Full Pareto sweep — N=500, real NYUv2 GT, E8/scale_bits=8/no-QJL
`--bits 8 4 3 2 --quantizer lattice_e8 --scale-bits 8 --no-qjl --max-samples 500`

| Config | eff b/scalar | δ1 ↑ | δ2 | δ3 | AbsRel ↓ | RMSE ↓ | vs FP32 | vs FP16 |
|---|---|---|---|---|---|---|---|---|
| FP32 | 32.0 | 0.8143 | 0.9476 | 0.9750 | 0.1496 | 0.4999 | 1.0x | 0.5x |
| 8-bit | 9.0 | 0.8143 | 0.9477 | 0.9750 | 0.1497 | 0.5002 | 3.6x | 1.8x |
| 4-bit | 5.0 | 0.8097 | 0.9467 | 0.9745 | 0.1521 | 0.5095 | 6.4x | 3.2x |
| 3-bit | **4.0** | 0.8142 | 0.9456 | 0.9748 | 0.1502 | 0.5032 | **8.0x** | **4.0x** |
| 2-bit | 3.0 | 0.6632 | 0.8890 | 0.9602 | 0.2316 | 0.6678 | 10.7x | 5.3x |

Reading of the curve:
- **The knee is at 3-bit/4.0-eff**: 8/4/3-bit are all within δ1 noise of FP32
  (max spread 0.0046), then 2-bit falls off a cliff (δ1 −0.151, AbsRel +55%).
  The claim "essentially lossless at 4.0 all-inclusive effective bits/scalar,
  catastrophic one notch below" is the paper's rate-distortion story.
- **Non-monotonicity caveat (do not hide this in the paper)**: 3-bit δ1
  (0.8142) nominally beats 4-bit (0.8097). A drop of 0.0046 followed by a
  recovery of 0.0045 means these three configs are statistically
  indistinguishable at N=500 under this protocol — the honest claim is
  "≥8-bit through 3-bit are within measurement noise of FP32", NOT "3-bit is
  better than 4-bit". Verify the ordering persists (or average out) at the
  full N=654 before writing any per-bit ranking into the paper.
- **FP32 baseline is below VDA's published NYU numbers** (δ1 0.814 here vs
  ~0.94 reported for VDA-S with its official protocol). Known causes: 266×266
  input (official uses 518), single image repeated as a 3-frame static clip,
  and no official eval crop. This does not invalidate the *relative*
  quantization deltas, but reviewers WILL line our FP32 row up against the
  VDA paper — before submission, re-run at 518 input with the official
  protocol so the baseline is defensible on its own.

---

## F9 — GT inference harness fixes (Colab validation, post-T8)

Discovered while validating the headline result on KITTI. Two harness bugs,
neither in the quantization math:

### F9a — inference resolution (FIXED, but NOT the KITTI cause)
`predict_depth` hard-resized every image to 266×266 SQUARE. Rewrote to mirror
VDA's own `infer_video_depth`: aspect-preserving resize to 518 shorter-side
(multiple of 14) via VDA's `Resize` transform, incl. the ratio>1.78 shrink
branch KITTI triggers; prediction upsampled back to native res, GT no longer
downsampled. Verified this changed the predictions — but KITTI FP32 δ1 stayed
~0.43, so resolution was NOT the dominant problem. Ruling it out led to F10.

## F10 — alignment space mismatch (THE KITTI cause) — FIXED

VDA's `vits` checkpoint (the one the suite auto-downloads) is the RELATIVE
model, which outputs **disparity (inverse depth)**, not metric depth (metric
is a separate checkpoint needing --metric; confirmed in VDA README). The GT
protocol was fitting `s·pred + t ≈ gt_metric` **linearly** — a scale+shift map
from disparity to metric depth, which cannot represent the reciprocal
relationship. Being range-dependent, it half-worked on NYU's narrow 0.1–10 m
range (δ1 0.81) and collapsed on KITTI's wide 1–80 m range (δ1 0.43).

**Fix** (`compute_gt_depth_metrics`, `pred_is_disparity=True` default): align in
disparity space, then invert to metric for the metrics —
`gt_disp = 1/gt; (s,t) = lstsq(pred → gt_disp); pred_depth = 1/clamp(s·pred+t)`.
This is the standard MiDaS/DepthAnything protocol. `--pred-space {disparity,
metric}` exposes it (metric for a metric checkpoint). Locked in by two new
tests: perfect disparity → δ1 1.0; disparity-alignment beats metric-alignment
by >0.20 δ1 on a wide range (reproduces the NYU/KITTI differential).

**IMPORTANT — invalidates earlier absolute numbers.** Every GT δ1/AbsRel/RMSE
recorded above (NYU N=200/N=500, KITTI) used the WRONG (metric-space) alignment
and must be RE-RUN before any use. In particular the NYU "δ1 drop 0.0061 / gate
PASSED" headline needs regenerating under disparity alignment — the *relative*
quantization deltas are likely still small (quantization noise ≪ the alignment
error), but the absolute baselines will move and must not be cited until re-run.
The compression/bit-accounting results (T1/T8, eff=4.0) are unaffected — those
are exact and independent of the eval protocol.

Pending re-runs (disparity alignment, 518px): NYU + KITTI full sweeps, then
Sintel. Compare FP32 baselines against published (NYU ~0.94, KITTI Eigen ~0.96)
to confirm the harness is finally sound.

### VALIDATED — KITTI, disparity alignment + 518px + depth cap, N=200
`--dataset kitti --bits 8 4 3 2 --quantizer lattice_e8 --scale-bits 8 --no-qjl --max-samples 200`

FP32 baseline δ1 **0.9275** / AbsRel **0.0922** — matches published VDA-S KITTI
(~0.96 / ~0.07) closely at N=200 with this protocol. **The harness is sound.**
This SUPERSEDES all earlier KITTI numbers (which used the broken metric-space
alignment). All four metrics now clean and mutually consistent:

| Config | eff b/scalar | δ1 ↑ | AbsRel ↓ | RMSE ↓ | vs FP16 |
|---|---|---|---|---|---|
| FP32 | 32.0 | 0.9275 | 0.0922 | 3.795 | — |
| 8-bit | 9.0 | 0.9276 | 0.0921 | 3.792 | 1.8x |
| 4-bit | 5.0 | 0.9277 | 0.0925 | 3.875 | 3.2x |
| **3-bit** | **4.0** | **0.9194** | **0.0962** | **4.127** | **4.0x** |
| 2-bit | 3.0 | 0.7104 | 0.1919 | 7.281 | 5.3x |

**3-bit @ 4.0 eff bits: δ1 drop 0.0081 (< 0.02 gate), AbsRel +4.3% rel, then a
sharp 2-bit cliff (δ1 -0.22, AbsRel 2x).** Same rate-distortion shape as NYU,
now on a SECOND domain with a defensible baseline — the core cross-domain
result for the paper. 8/4/3-bit statistically indistinguishable from FP32; do
not rank within them (N=200).

### VALIDATED — NYUv2, disparity alignment + 518px + depth cap, N=200 (supersedes the two SUPERSEDED blocks above)
`--dataset nyuv2 --bits 8 4 3 2 --quantizer lattice_e8 --scale-bits 8 --no-qjl --max-samples 200`

FP32 baseline δ1 **0.9036** / AbsRel **0.098** — up from 0.81 under the broken
metric-space alignment, now near published VDA-S NYU (~0.94). Confirms F10 was
suppressing NYU too, just less than KITTI (narrower range). Both datasets now
have defensible baselines.

| Config | eff b/scalar | δ1 ↑ | AbsRel ↓ | RMSE ↓ | vs FP16 |
|---|---|---|---|---|---|
| FP32 | 32.0 | 0.9036 | 0.0980 | 0.456 | — |
| 8-bit | 9.0 | 0.9034 | 0.0981 | 0.456 | 1.8x |
| 4-bit | 5.0 | 0.9030 | 0.0997 | 0.457 | 3.2x |
| **3-bit** | **4.0** | **0.9128** | **0.0925** | **0.428** | **4.0x** |
| 2-bit | 3.0 | 0.6430 | 0.2052 | 0.810 | 5.3x |

3-bit δ1 (0.9128) nominally ABOVE FP32 (0.9036) — noise, same non-monotonicity
caveat: 8/4/3-bit are statistically indistinguishable from FP32; do not rank
within them. 2-bit cliff (δ1 -0.26). Rate-distortion shape matches KITTI exactly.

**Cross-domain result now solid: at 4.0 all-inclusive effective bits/scalar (4x
vs FP16), 3-bit is within δ1 noise of FP32 on BOTH NYUv2 (indoor) and KITTI
(outdoor), with a sharp collapse at 2-bit — both against baselines matching
published VDA numbers.**

Still pending: full N=654 both datasets, Sintel (temporal/TAE), and matched-bit
baseline rows (scalar/D4) for a comparison column.

### Sintel — N=200, corrected protocol (quant replicates; weak ACCURACY baseline)
`--dataset sintel --bits 8 4 3 2 --quantizer lattice_e8 --scale-bits 8 --no-qjl --max-samples 200`

| Config | eff b/scalar | δ1 ↑ | AbsRel ↓ | RMSE ↓ |
|---|---|---|---|---|
| FP32 | 32.0 | 0.5739 | 2.228 | 115.8 |
| 8-bit | 9.0 | 0.5741 | 2.232 | 115.9 |
| 4-bit | 5.0 | 0.5692 | 2.131 | 114.3 |
| 3-bit | 4.0 | 0.5723 | 2.193 | 120.1 |
| 2-bit | 3.0 | 0.3717 | 3.029 | 103.0 |

Quantization pattern REPLICATES a third time (8/4/3-bit within δ1 noise of FP32,
2-bit cliff) — the compression claim now holds on indoor (NYU) + outdoor (KITTI)
+ synthetic (Sintel). BUT the FP32 *accuracy* baseline is weak (δ1 0.57, AbsRel
2.23) and this is consistent across ALL bit-widths, so it's a model+protocol
property, NOT a quantization failure and NOT the KITTI-style alignment bug.

Cause: Sintel's depth range spans ~4 orders of magnitude (foreground ~1 m to
near-sky), which a single global scale+shift affine (even in disparity space)
fits poorly; AbsRel is dominated by far pixels where small disparity errors
become huge relative depth errors. Sintel is a known-hard zero-shot benchmark;
δ1 ~0.57 with global-affine relative depth is roughly expected. Deliberately NOT
tightening gt_range to prettify the baseline (that would be fishing).

Implication for the paper: do NOT lead with Sintel accuracy. Sintel's real value
is TEMPORAL (it's the only GT dataset that is also real video) — but TAE is still
`n/a` because run_groundtruth_eval evaluates each frame as an independent static
3-frame clip. A real Sintel TAE number needs a temporal path feeding consecutive
frames (PENDING — this is what earns Sintel its place in the paper).

---

# Phase 2 — Paper-grade evaluation (specs for Sonnet)

KEY DISCOVERY enabling all three tasks: the cloned Video-Depth-Anything repo
ships its own benchmark protocol in `Video-Depth-Anything/benchmark/`:
- `eval/eval.py` — disparity-space lstsq alignment, clip aligned disparity to
  ≥1e-3, invert, clip depth to [1e-3, max_depth_eval]. **Identical to our F10
  fix** (independent confirmation), with per-dataset `max_depth_eval`:
  Sintel = 80.0 / min 0.1.
- `eval/eval_tae.py` — their TAE (`tae_torch(depth1, depth2, R_2_1, T_2_1, K,
  mask)`) is GEOMETRIC REPROJECTION between consecutive frames using camera
  intrinsics K and relative pose R,T — NOT optical flow. Sintel's
  `camdata_left/` (already in our depth zip) provides exactly K + extrinsics.
Mirroring these makes every number directly comparable to the VDA paper's own
tables — the strongest possible protocol citation.

## T11 — Protocol alignment with VDA + full splits — BUILT (code); full-split runs PENDING (GPU)
Files: `scripts/datasets_gt.py`
- **Correction while implementing**: the spec above guessed Sintel
  `max_depth_eval=80.0` from a grep that turned out to have false-matched
  KITTI's constant. Reading `Video-Depth-Anything/benchmark/eval/eval.py`
  directly gives the REAL per-dataset constants: nyuv2 (0.1, 10.0) — already
  matched; kitti (0.1, 80.0) — ours had min=1e-3, fixed to 0.1; **sintel
  (0.1, 70.0), NOT 80.0**. All three now pinned exactly, with a test
  (`test_dataset_gt_config_matches_vda_eval_protocol`) citing the source
  lines so this can't silently drift again. This is exactly the kind of
  "spec said X, source says Y" check worth doing before hardcoding — flagging
  it here so future ledger specs get re-verified against source, not trusted
  at face value.
- Pending: full-split re-runs (NYU 654, KITTI 1000, Sintel all frames) —
  GPU/Colab work, not done locally per standing instruction.

## T9 — VALIDATED on real data (Sintel, 20 scenes, 894 pairs, seeded)

`--dataset sintel --eval-mode temporal --bits 8 4 3 2 --quantizer lattice_e8
--scale-bits 8 --no-qjl --max-samples 2000 --max-scenes 20 --rht-seed 0`

| Config | eff b/scalar | δ1 ↑ | AbsRel ↓ | RMSE ↓ | **TAE median ↓** | TAE mean |
|---|---|---|---|---|---|---|
| FP32 | 32.0 | 0.7074 | 0.2330 | 5.077 | **6.892** | 57.89 |
| 8-bit | 9.0 | 0.7073 | 0.2334 | 5.088 | **6.886** | 57.92 |
| 4-bit | 5.0 | 0.7044 | 0.2384 | 5.190 | **6.817** | 57.90 |
| **3-bit** | **4.0** | **0.6852** | **0.2555** | **5.551** | **6.870** | 58.09 |
| 2-bit† | 3.0 | 0.4990 | 0.3754 | 5.528 | 4.700† | 7.27† |

**HEADLINE: at 4.0 all-inclusive effective bits/scalar (8x vs FP32, 4x vs FP16),
3-bit costs δ1 −0.022 / AbsRel +0.023 and leaves temporal consistency
UNCHANGED (TAE median 6.892 → 6.870).** Across FP32/8/4/3-bit — the configs
with comparable accuracy — TAE median is flat within ±0.08 while accuracy
degrades monotonically. The claim is robust to the summary statistic: the
mean is equally flat (57.89 → 58.09), so it does not depend on choosing the
median. FP32 baseline δ1 0.7074 also beats DepthCrafter's published Sintel
0.697, so the baseline is defensible.

### † CRITICAL CAVEAT — TAE is gameable by degenerate predictions
2-bit has the WORST accuracy by far (δ1 0.499, a −0.21 collapse) yet posts the
BEST TAE (median 4.70 vs FP32's 6.89; mean 7.27 vs 57.89). Its `ambush_2` TAE
is **13.7 vs 906.4** for every accurate config — a 66x "improvement" from the
model that is comprehensively worse. This is not a result; it is the metric
being gamed. A prediction that has collapsed toward smooth/near-constant
reprojects onto itself almost perfectly, so reprojection-based TAE rewards it.

**Therefore TAE must NEVER be reported alone.** It is only meaningful between
configs of comparable accuracy. Any TAE comparison in the paper must exclude
configs whose accuracy has collapsed, and must state why. This is worth
reporting explicitly as a methodological finding — it is a real, demonstrable
weakness of the reprojection-TAE protocol that the field uses.

### Per-scene distribution is heavy-tailed (mean is the wrong default)
FP32 per-scene TAE ranges shaman_2 1.0% … ambush_2 906.4%; median 6.89 vs mean
57.89, with ambush_2 alone contributing ~45 of the 57.9 mean. Ordering tracks
camera motion exactly (static scenes ~1%, violent-motion scenes 20–900%).

### Z-buffer: real fix, but NOT the cause here (hypothesis was wrong)
Upstream `eval_tae.py` resolves reprojection collisions by arbitrary
last-write-wins (`depth_proj[Y,X] = Z`), letting a FAR pixel overwrite a NEAR
one. Replaced with nearest-wins (`scatter_reduce` amin) — genuinely more
correct, verified on a constructed isolated collision (z-buffer 0.0 exact vs
last-write 0.629). **But it did NOT explain the inflation**: ambush_2 moved
906.9 → 906.4, median 7.01 → 6.89. Recording this because the earlier
hypothesis ("disocclusion via collisions is the cause") was falsified by the
data. The residual cause is true DISOCCLUSION, which a z-buffer structurally
cannot fix — frame t simply has no information about content newly visible in
t+1. Proper fix (PENDING, refinement not blocker): a GT-derived co-visibility
mask — use GT depth + GT poses to determine which pixels are genuinely visible
in both frames and evaluate only those. Non-circular (mask from ground truth,
metric on predictions). The median already sidesteps this, which is why the
headline above stands without it.

## T9 — implementation notes (code BUILT, now validated above)
Files: `scripts/run_pareto_benchmark_suite.py`, `scripts/datasets_gt.py`,
`tests/test_temporal_tae_path.py`
- `scene`/`frame_idx` added to every loader (None for NYU/KITTI — confirmed
  non-groupable; real values for Sintel). `group_samples_by_scene` raises on
  scene=None rather than silently fabricating a fake sequence out of
  unrelated frames.
- `--eval-mode temporal` (Sintel-only, raises clearly for other datasets):
  `chunk_scene_into_windows` splits each scene into real consecutive
  `--temporal-window` (default 16) frame windows, static-padding a short
  trailing window by repeating its last real frame and tracking `n_real` so
  padding is provably excluded from every downstream temporal comparison.
  `_predict_window` feeds each window through the model in ONE forward call
  (`video_length=window`) — this is the actual fix that makes TAE
  measurable at all: the pre-existing static-3-frame-clip hack shares no
  state across calls, so it could never have shown temporal (in)consistency
  no matter what was measured.
- **Finding while porting**: VDA's own repo does NOT actually wire up TAE
  for Sintel — `eval_tae.py`'s CLI only has a ScanNet branch; Sintel's own
  extraction script (`dataset_extract_sintel.py`) builds a JSON manifest via
  the plain `gen_json` (no K/pose fields at all), unlike ScanNet's
  `gen_json_scannet_tae`. The `tae_torch` math is real and dataset-agnostic,
  but the Sintel camera-pose plumbing to feed it doesn't exist upstream —
  built it here from the raw `.cam` files (verified against two independent
  public copies of the official Sintel SDK's `cam_read`): magic 202021.25,
  3x3 intrinsic (9×float64), 3x4 extrinsic `[R|t]` (12×float64),
  `x = M·N·X` (world-to-camera). `_cam_to_world_pose` inverts this into the
  camera-to-world 4x4 VDA's own pose-composition convention expects
  (`T_2_1 = inv(pose_2) @ pose_1`) — proven algebraically exact (not just
  numerically close) in `test_cam_to_world_pose_is_true_inverse`.
- `_tae_geometric_single` + `_pool_align_scene_disparity` +
  `compute_tae_geometric_for_scene` port `tae_torch`/`eval_TAE` faithfully
  (pooled scene-wide disparity alignment, bidirectional per-pair AbsRel via
  camera reprojection, ×100 → percent). A caught-before-running bug: the
  summary table checked `'tae' in m` but temporal-mode metrics use
  `'tae_percent'` — would have silently printed "n/a" for real TAE numbers.
- **Documented limitation** (not hidden): windows run independently with no
  shared cache across boundaries, so frame pairs straddling a window edge
  don't get the same cross-frame context within-window pairs do. Fine for a
  first correct measurement; a future pass could use `cached_hidden_states`
  (already plumbed since T7) to close this gap.
- Tests (20, all synthetic/no-GPU): `.cam` round-trip exact; pose-inverse
  proven algebraically; window-chunking edge cases (exact multiple,
  remainder, shorter-than-window, empty, invalid); geometric TAE math
  (identity-motion → exactly 0; a NON-trivial lateral-translation case
  verified analytically, not just "doesn't crash"; deliberate inconsistency
  detected at the exact expected value; pooled alignment recovers a known
  affine exactly); and a full-pipeline oracle test computing the expected
  multi-frame TAE from first principles (matches to 6 decimal places) rather
  than a guessed bound — the first version of that test asserted a wrong
  guessed bound and failed; the fix was deriving the correct expected value
  by hand, not loosening the assertion.
- **Pending — the actual headline number**: this is all validated as
  CORRECT MATH on synthetic fixtures. It has never run against a real
  checkpoint or real Sintel frames (no GPU here). First Colab run should
  watch for: does `m(clip)` actually accept `video_length=16` cleanly
  end-to-end through the DPT temporal decoder (only ever exercised at T=3
  before this); do the reported TAE percentages fall in a sane range
  relative to VDA's own published Sintel TAE table.

## T10 — Matched-rate baselines + rotation ablation — BUILT (code); baseline sweep runs PENDING (GPU)
Files: `research/transforms/hadamard.py`, `research/models/rotated_attention.py`,
`scripts/run_pareto_benchmark_suite.py`, `tests/test_rotation_ablation.py`
- `HadamardRotation` gains `identity=True` (true no-op: no signs buffer, no
  padding, forward/inverse both return input unchanged) and `seed=N` (local
  `torch.Generator`, does NOT touch the global RNG — proven, not assumed, via
  `test_hadamard_seed_does_not_disturb_global_rng`). Threaded as
  `--rotation/--no-rotation` and `--rht-seed` through both attention classes
  and `apply_rotated_quantization_to_vda`. Each replaced layer derives its
  OWN seed (`rht_seed + counter`) rather than sharing one literal seed across
  every layer — reproducible across full runs, still independent per layer
  like the original unseeded behavior.
- **Empirical validation the ablation infrastructure actually shows the
  expected effect** (not just "the flag doesn't crash"):
  on synthetic activations with injected per-row outliers (the realistic ViT
  pattern this whole method targets), RHT-then-D4-quantize gives **2.85x
  lower MSE** than quantizing raw (0.131 vs 0.374, 3-bit). This is the
  empirical core of the paper's justification for RHT, now a locked-in
  regression test — if a future change silently broke the rotation's benefit,
  this test would catch it.
- **Real accounting bug found and fixed**: `QUANTIZER_GROUP_SIZE` charged
  the scalar baseline as if it shared D4's per-4-element-group scale
  (`group_size=4`), but `ScalarRoundQuantizer` is always called with
  `per_channel=False` in this codebase (verified by reading every call
  site) — meaning ONE scale for the entire tensor, not one per 4 elements.
  This overcharged scalar's scale overhead by **16x** (256 vs true ~16
  bits/vector at head_dim=64) — an error that always flattered our
  quantizers relative to the baseline, exactly backwards from the honesty
  bar the rest of this ledger holds itself to (F1/T1). Fixed via
  `resolve_group_size()`, mapping `'scalar'` → `head_dim` (one scale per
  whole vector). Effective bits/scalar for scalar @ 3-bit, head_dim=64,
  scale_bits=16, no QJL: **3.25** (was wrongly reporting **7.0**).
- Pending: the actual baseline sweep runs (`--quantizer scalar` /
  `--quantizer lattice_d4` on NYU+KITTI at bits 4/3/2, and a
  `--no-rotation` vs `--rotation` comparison at matched bits on real data)
  — GPU/Colab work, not done locally.

Execution order: T5 → T6 → T1 → T4 → T3 → T2 → T7 → T8 → F9/F10 → T11 → T9 → T10.
All Phase 2 CODE is now built and unit-tested (94/94 total tests green,
0 network/GPU/dataset access). Nothing has been pushed — committed locally,
one commit per task, awaiting review. Every remaining item across T9/T10/T11
is now purely "run it on Colab and report the numbers," not further
implementation.

---

# Phase 3 — GPU run-sheet: the complete list of experiments left for the paper

Status of validated results going in: NYU N=200 GT sweep ✓ (δ1 0.904 FP32),
KITTI N=200 GT sweep ✓ (δ1 0.928), Sintel 20-scene temporal sweep ✓ (TAE
median flat 6.89→6.87 at 3-bit). Everything below is CLI-only; no code
changes required except E7 (optional). All runs: `--rht-seed 0` unless the
experiment is the seed study itself. Time estimates assume the same Colab GPU
class used so far (N=200 5-config sweep ≈ 15–25 min).

## E1 — Baseline comparison rows (the "vs what?" table) — HIGHEST PRIORITY
Reviewers' first question. Same protocol as the E8 rows so the table is
apples-to-apples. NYU + KITTI, N=200 (upgrade to full split only for finals).
  E1a scalar RTN:   --quantizer scalar     --bits 4 3 2   (eff 4.25/3.25/2.25)
  E1b D4 lattice:   --quantizer lattice_d4 --scale-bits 8 --bits 4 3 2 (eff 6/5/4)
Note the matched-rate comparison for the paper: E8 3-bit (4.0 eff) vs D4 2-bit
(4.0 eff) vs scalar ~4-bit (4.25 eff) — same budget, three schemes; the
prediction from the coding-gain theory is E8 > D4 > scalar at equal rate.
2 datasets × 2 quantizers ≈ 4 runs ≈ 1.5 h.

## E2 — Rotation ablation on real data (justifies RHT's existence)
  --quantizer lattice_e8 --scale-bits 8 --no-qjl --bits 4 3 --no-rotation
KITTI N=200 (worst case for outliers) + NYU N=200. Compare to existing
rotation-on rows. Expected: large δ1 drop without rotation at 3-bit (synthetic
test showed 2.85x MSE gap). 2 runs ≈ 40 min.

## E3 — RHT seed robustness (kills the "lucky draw" objection)
3-bit E8 only, NYU N=200, --rht-seed 1 and 2 (seed 0 already exists).
Report δ1/AbsRel mean±std over 3 seeds. 2 runs ≈ 20 min.

## E4 — Ablation rows: metadata & QJL cost at the operating point
  E4a scale_bits: 3-bit E8 --scale-bits 16 (eff 5.0) vs existing 8 (eff 4.0), NYU N=200.
  E4b QJL on:     3-bit E8 --scale-bits 8 --qjl (default m=128 → eff 6.25), NYU N=200.
      Expected ≈ no accuracy gain for +2.25 bits — this is the empirical row
      that justifies use_qjl=False in the headline config and cites F3b.
2 runs ≈ 30 min.

## E5 — Full-split finals (the paper's actual tables; run LAST, after
## E1–E4 lock the story at N=200)
  E5a NYU full test split:   --dataset nyuv2 --max-samples 654
  E5b KITTI full val split:  --dataset kitti --max-samples 1000
  E5c Sintel temporal, all scenes/frames: --max-samples 2000 --max-scenes 23
Headline sweep (E8 8/4/3/2) on each; plus ONE full-split baseline run each of
E1a/E1b at the matched-rate bits if time allows. 3 runs ≈ 2–3 h.
Paper convention note: Sintel accuracy should be quoted from the TEMPORAL
path (video-native, δ1 0.707 ≫ static-clip 0.574); the static-clip Sintel
GT numbers are superseded and must not be mixed into tables.

## E6 — Qualitative figures for the paper
  scripts/dump_depth_samples.py on nyuv2 / kitti / sintel, bits 8 4 3 2,
  --make-video for Sintel. Produces the strips + MP4. ≈ 20 min total.

## E7 — (Optional, small code task) GT co-visibility mask for TAE
Use GT depth + GT poses to mask pixels not visible in both frames; makes the
TAE *mean* trustworthy (median already is) and gives sane per-scene numbers
for ambush_2/market_5. One code task + one 20-scene rerun. Refinement, not a
blocker — headline stands on the median, corroborated by the flat mean.

## Explicitly OUT of scope for this paper (write as future work, do not run)
- Real INT/lattice CUDA kernels & measured latency (memory compression is
  analytic and exact; throughput is simulated — say so plainly).
- DAVIS (no depth GT; temporal-only story already carried by Sintel).
- ScanNet (ToU-gated; start the email if wanted for camera-ready).
- Metric-depth checkpoint / on-device scale recovery (deployment paper).

Suggested order: E1 → E2 → E3 → E4 (locks the full story at N=200, ~3 h)
→ E5 finals (~3 h) → E6 figures. E7 if a spare hour remains.

---

## F11 — Scalar baseline is UNFAIRLY WEAK (scale granularity confound) — OPEN

Found mid-E1 while the Phase 3 sweep was running. Verified by reading
`research/quantizers/lattice_vq.py`: `ScalarRoundQuantizer.forward` is always
invoked with `per_channel=False` (both call sites in
`research/models/rotated_attention.py`, lines ~200/383, pass no flag), so it
takes `x_max = x.abs().amax()` — **ONE global scale for the ENTIRE tensor**.
Meanwhile `lattice_e8` uses a scale per 8-element group and `lattice_d4` one
per 4-element group.

So the headline "E8 beats scalar" comparison currently conflates TWO effects:
  1. lattice / vector-quantization coding gain (what we claim), and
  2. scale granularity — per-8-group vs per-WHOLE-TENSOR (a much coarser
     baseline, where one outlier anywhere inflates the scale for everything).

Measured E1 numbers (N=200, RHT on for both, GT protocol):
| Scheme | eff bits | NYU δ1 | KITTI δ1 |
|---|---|---|---|
| FP32 | 32 | 0.9036 | 0.9275 |
| E8 3-bit | 4.00 | 0.9128 | 0.9194 |
| scalar 4-bit | 4.25 | 0.8908 | 0.9002 |
| scalar 3-bit | 3.25 | 0.6455 | 0.6316 |
| E8 2-bit | 3.00 | 0.6430 | 0.7104 |

E8 wins at matched rate (4.00 vs 4.25 eff bits: +0.022 NYU / +0.019 KITTI,
using FEWER bits) and wins clearly at ~3 eff bits on KITTI (0.7104 vs 0.6316).
The result direction is right — but a reviewer WILL ask "is that the lattice,
or just finer scales?" and right now we cannot answer.

REQUIRED before this table goes in the paper (pick at least one):
  (a) Add a fair scalar baseline with per-group scales (expose `per_channel`
      or a group_size on ScalarRoundQuantizer) and charge its scale overhead
      honestly in `compute_real_bit_accounting` — the T10 `resolve_group_size`
      already handles the accounting side.
  (b) Report `lattice_d4` (group 4) vs `lattice_e8` (group 8) as the
      lattice-dimension comparison at MATCHED scale granularity — E1b is
      already producing this, and it isolates lattice gain far better than
      the scalar row does.
  (c) At absolute minimum, disclose the granularity difference explicitly in
      the table caption. Do not let "E8 > scalar" stand unqualified.

NOTE: this does NOT invalidate the E8 vs FP32 headline (compression at
matched accuracy) — only the E8-vs-scalar-baseline comparison.

---

# PHASE 3 RESULTS â€” COMPLETE (13/13 experiments, collaborator GPU box, 2026-07-21)

All Phase 3 experiments finished, including E4b (which OOM-recovered on rerun)
and the full-split finals E5a/b/c. Raw results: `phase3_results.zip`.
Findings F12â€“F17 below are derived from those JSONs and CLOSE decision gates
DG-1 and DG-3 (see `docs/phase4_a_star_plan.md`).

## F12 â€” DG-1 RESOLVED: E8@3-bit is LOSSLESS on full splits â€” the +0.01 was small-sample noise

The N=200 result "quantization beats FP32 by +0.01 delta1", which was the single
biggest reviewer-credibility risk (attack A2), **does not survive the full split.**

| Split | Config | eff bits | delta1 | AbsRel | Delta-delta1 vs FP32 |
|---|---|---|---|---|---|
| NYU-654 | FP32 | 32 | 0.9099 | 0.0963 | â€” |
| NYU-654 | E8 8-bit | 9.0 | 0.9095 | 0.0965 | âˆ’0.0004 |
| NYU-654 | E8 4-bit | 5.0 | 0.9067 | 0.0991 | âˆ’0.0032 |
| NYU-654 | **E8 3-bit** | **4.0** | **0.9085** | **0.0961** | **âˆ’0.0014** |
| NYU-654 | E8 2-bit | 3.0 | 0.6392 | 0.2096 | âˆ’0.2707 |
| KITTI-1000 | FP32 | 32 | 0.9280 | 0.0909 | â€” |
| KITTI-1000 | E8 8-bit | 9.0 | 0.9282 | 0.0908 | +0.0002 |
| KITTI-1000 | E8 4-bit | 5.0 | 0.9319 | 0.0883 | +0.0039 |
| KITTI-1000 | **E8 3-bit** | **4.0** | **0.9271** | **0.0912** | **âˆ’0.0009** |
| KITTI-1000 | E8 2-bit | 3.0 | 0.6179 | 0.2241 | âˆ’0.3101 |

At N=200 the gap was +0.0092 (0.9128 vs 0.9036); at N=654 it is âˆ’0.0014.
BOTH endpoints moved (FP32 0.9036 -> 0.9099, E8@3b 0.9128 -> 0.9085), which is the
signature of subset sampling noise, not a systematic effect.

**HEADLINE CLAIM (locked):** E8 lattice at 3-bit payload / **4.0 all-inclusive
effective bits per scalar** is statistically indistinguishable from FP32 on
NYU-654 and KITTI-1000 â€” 8.0x compression vs FP32, 4.0x vs FP16.
**NEVER claim quantization improves accuracy.** Report full-split numbers only;
N=200 deltas of +/-0.01 are inside subset noise and must not be reported as findings.

Caveat still open: a real scale-bits effect may exist at fixed N (E4a sb=16 gave
0.9029 while E3 seeds 1/2 at sb=8 gave 0.9134/0.9148 on the SAME 200 images).
Different seeds, so confounded. G2b (full split x {8,16} x 3 seeds) settles it.
Note the zip contains NO seed-0/sb8/QJL-off/rotation-on run at N=200 â€” the
0.9128 figure came from an older run, so G3a control is still required.

## F13 â€” E8@2-bit COLLAPSES: 4.0 eff bits is the frontier, not 3.0

NYU 0.6392, KITTI 0.6179, Sintel 0.5017 â€” a cliff, not a slope, on all three
datasets. The hoped-for "3.0 effective bits" headline is **dead**.

Consequence for framing: the claim is **"3-bit payload, 4.0 all-inclusive bits"**,
and the metadata bit must be stated every single time. Writing "3-bit KV cache"
without the qualifier is exactly the undercounting this project exists to
criticise (see F1/T1). The Pareto figure story is the CLIFF between 4.0 and 3.0
eff bits, and that every baseline family falls off it earlier than E8 does.

## F14 â€” Rotation (RHT) is INERT at 4.0â€“5.0 eff bits (DG-2 evidence, not yet closed)

E2 no-rotation, N=200: NYU 3-bit (4.0 eff) delta1 **0.9138**, KITTI **0.9244**.
With-rotation at the same N and rate (E3 seeds 1,2): **0.9134 / 0.9148**.
No-rotation lands INSIDE the with-rotation seed spread. Rotation buys nothing
measurable at this rate.

This is now well-enough supported that DG-2 branch (c) is the leading hypothesis:
**contra LLM practice (QuaRot/SpinQuant), video-ViT KV activations do not need
rotation at moderate rates.** Still required before locking:
  - G3a: seed-0 with-rotation control at N=200, same code version (missing above).
  - G3b: rotation on/off at 2-bit â€” the ONE regime where RHT could still earn its
    place. E2 only ran bits 4 and 3, so this is untested. Note E8@2b collapses
    anyway (F13), so the realistic outcomes are "rotation does not save 2-bit
    either" (-> ablate RHT out of the contributions) or "rotation shifts the cliff"
    (-> RHT is the extreme-rate enabler, a strong story).
  - G3d: activation kurtosis evidence for WHY.

If branch (c) holds, RHT moves from "contribution" to "ablated component", the
method simplifies to E8-lattice-on-KV-cache, and the overlap with QuaRot/SpinQuant
shrinks â€” which is good for novelty, not bad.

## F15 â€” QJL is strictly dominated: costs 2.25 bits AND loses accuracy

E4b (QJL on, sb=8, 3-bit, N=200): eff **6.25** bits, delta1 **0.9057**.
Comparable QJL-off runs at eff **4.0**: delta1 0.9134 / 0.9148 (E3 seeds 1,2).
QJL costs +2.25 effective bits and is no better â€” worse, on these samples.
Combined with F3b (the m=512-works / m=128-hurts bias-variance result), the
headline config being QJL-off is now empirically justified, not just a convention.
Report this as a clean negative result in the ablation table.

## F16 â€” TAE IS GAMEABLE BY DEGENERATE SMOOTHING (major; reshapes contribution 3)

The most important finding in Phase 3. E5c Sintel, N=1045 frames / 23 scenes:

| Config | eff bits | delta1 | AbsRel | TAE mean % | TAE median % |
|---|---|---|---|---|---|
| FP32 | 32 | 0.7122 | 0.2335 | 55.41 | 7.83 |
| E8 8-bit | 9.0 | 0.7122 | 0.2339 | 55.44 | 7.83 |
| E8 4-bit | 5.0 | 0.7082 | 0.2419 | 55.49 | 7.54 |
| E8 3-bit | 4.0 | 0.6882 | 0.2601 | 54.90 | 6.89 |
| E8 2-bit | 3.0 | **0.5017** | **0.3767** | **7.38** | **4.50** |

The 2-bit config â€” which has *collapsed* on accuracy â€” posts a **7.5x better TAE
mean** and the best median of any configuration, FP32 included. Read naively,
TAE says 2-bit is the most temporally consistent model we have.

It is not. Per-scene decomposition proves degeneracy rather than genuine improvement:

| Scene | FP32 TAE | 3-bit | 2-bit | 2b/FP32 |
|---|---|---|---|---|
| ambush_2 | 906.30 | 907.26 | 12.67 | **0.01x** |
| temple_2 | 54.36 | 42.59 | 2.83 | 0.05x |
| market_5 | 92.74 | 82.93 | 17.02 | 0.18x |
| temple_3 | 61.67 | 62.87 | 18.60 | 0.30x |
| ... | | | | |
| bandage_2 | 2.11 | 2.36 | 4.50 | **2.14x** |
| alley_2 | 1.87 | 2.57 | 4.33 | **2.31x** |
| shaman_2 | 1.00 | 1.43 | 3.90 | **3.92x** |

The direction of change **flips with scene difficulty**: high-motion scenes fall
toward a ~1â€“18% floor while easy, near-static scenes get 2â€“4x WORSE. A genuine
temporal-consistency improvement would improve all scenes. What actually happens
is the 2-bit output loses spatial structure â€” with little structure left, there is
little to misalign under reprojection, so hard scenes hit a floor; on easy scenes
the added quantization noise dominates and TAE rises.

**Consequences (act on all three):**
1. **TAE must never be reported alone.** Every TAE number in the paper appears
   jointly with delta1/AbsRel, and the paper states explicitly that TAE is only
   meaningful at matched accuracy.
2. **This strengthens contribution 3 rather than weakening it.** A metrics
   contribution that documents its own failure mode is far more credible than one
   that does not. Elevate to a named subsection: *"Temporal consistency metrics are
   gameable: a degenerate low-rate model wins on TAE while failing on accuracy."*
   The 2-bit row becomes a deliberate, valuable exhibit â€” not an embarrassment.
3. Add a **structure/degeneracy diagnostic** (new task S10) so this is quantified,
   not merely argued: spatial-gradient energy and per-frame prediction variance of
   the predicted disparity, per config. Expect a sharp drop at 2-bit. Pair it with
   the E6 2-bit qualitative frames, which are already generated.

Also note: the FP32 TAE mean of 55.41% is dominated by ambush_2 at 906% â€”
disocclusion inflation, exactly as suspected. The median (7.83%) is the honest
aggregate today, and **S6 GT co-visibility masking is now mandatory, not optional**,
because a 906% "error" in a scene with correct geometry is indefensible in review.

## F17 â€” Temporal/Sintel degrades at 4.0 eff bits where static datasets do not

At 4.0 eff bits: NYU âˆ’0.0014 delta1, KITTI âˆ’0.0009 delta1, but **Sintel âˆ’0.0240**
(0.7122 -> 0.6882), with AbsRel 0.2335 -> 0.2601. The temporal path is measurably
harder than the static path at the same rate.

Plausible mechanism (UNTESTED â€” do not assert in the paper without evidence):
quantization error accumulating across the 32-frame temporal window, since each
frame attends to a cache that is itself quantized. Cheap test: sweep
`--temporal-window` in {8, 16, 32} at 4.0 eff bits and check whether the delta1 gap
vs FP32 grows with window length. If it does, that is a genuine and interesting
finding about temporal error accumulation and belongs in the paper; if flat, the
gap is just Sintel being harder and should be reported plainly as such.
Added as G10 in the Phase 4 plan.


---

# PHASE 4 GATES RESULTS â€” DECISION GATES CLOSED (collaborator GPU box, 2026-07-25)

Phase 4 STAGE=gates complete (G1, G2b, G3a/b/d, G10, G11). Raw: `phase4_results.zip`.
Findings F18â€“F22 close DG-1, DG-2, DG-3 and resolve F16/F17. Bootstrap CIs via
scripts/compute_stats.py (S2). NET: the headline survives; the "lattice is the key
ingredient" pillar is weakened; the rotation story got much stronger.

## F18 â€” DG-1 CLOSED WITH CIs: E8@3b (4.0 eff bits) is lossless; scale-bits 8 â‰¡ 16

Full-654 NYU, E8@3b, paired BCa bootstrap of (E8 âˆ’ FP32) delta1, per RHT seed:

| scale_bits | seed | E8@3b Î´1 | Î” vs FP32 | 95% BCa CI | excludes 0 |
|---|---|---|---|---|---|
| 8 | 0 | 0.9085 | âˆ’0.0014 | [âˆ’0.0047, +0.0019] | no |
| 8 | 1 | 0.9169 | +0.0071 | [+0.0032, +0.0109] | yes (E8 higher) |
| 8 | 2 | 0.9193 | +0.0094 | [+0.0059, +0.0131] | yes (E8 higher) |
| 16 | 0 | 0.9093 | âˆ’0.0006 | â€” | â€” |
| 16 | 1 | 0.9160 | +0.0061 | â€” | â€” |
| 16 | 2 | 0.9195 | +0.0096 | â€” | â€” |

Two locked conclusions:
1. **scale_bits 8 â‰¡ scale_bits 16** (per-seed within ~0.001: 0.9085/0.9093,
   0.9169/0.9160, 0.9193/0.9195). The F12 scale-bits caveat is CLOSED: 8-bit
   scales cost nothing in accuracy, so **4.0 eff bits is the honest headline rate**
   (5.0 with 16-bit scales buys nothing). Use scale_bits=8.
2. The within-seed image bootstrap is tight, but the **RHT seed is the dominant
   source of variance** (âˆ’0.0014 to +0.0094 across three draws). E8@3b is never
   significantly WORSE than FP32 (worst seed âˆ’0.0014, CI spans 0) and is sometimes
   significantly better. **Report as one-sided "lossless / no degradation," mean
   Î”Î´1 = +0.005 across 3 seeds.** NEVER headline "improves accuracy" â€” a lattice
   projection nudging a regression metric up by 0.005 is mild denoising, not a
   capability, and claiming otherwise reads as a bug to reviewers. The credible
   sentence: "matches FP32 within RHT-seed variance; never significantly worse."

## F19 â€” DG-3 PARTIALLY ADVERSE: the fair scalar baseline is STRONG; lattice gain is modest and dataset-dependent

The reason F11 mattered. E8@3b vs the fair scalar_g8@3b (BOTH 4.0 eff bits, BOTH
per-8-group scales, BOTH RHT seed 0, N=200), paired BCa bootstrap of (E8 âˆ’ scalar_g8):

| dataset | scalar_g8 Î´1 | E8 Î´1 | diff | 95% BCa CI | excludes 0 |
|---|---|---|---|---|---|
| NYU | 0.9090 | 0.9057 | âˆ’0.0033 | [âˆ’0.0070, +0.0012] | no (TIED) |
| KITTI | 0.9123 | 0.9286 | +0.0163 | [+0.0121, +0.0209] | yes (E8 higher) |

**The F11 "E8 beats scalar by 0.38" gap was ENTIRELY an artifact of the unfair
per-tensor scalar baseline.** Against a properly-grouped scalar baseline:
  - KITTI: E8 wins by a real, significant +0.016.
  - NYU: statistically TIED (E8 nominally 0.003 BEHIND).
And scalar_g8 itself is near-lossless vs FP32 at 4.0 eff bits (NYU 0.909 vs 0.904,
KITTI 0.912 vs 0.928) â€” i.e. **the video-depth KV cache is just highly compressible;
grouped scalar already gets most of the way, and lattice VQ adds a modest,
dataset-dependent margin.**

Caveat weakening this further: seed 0 is E8's WORST NYU draw (F18: seed0 0.9085 vs
seed2 0.9193 at N=654), so the NYU "tie" is under E8's least favorable rotation. But
we have no scalar_g8 at other seeds to keep it fair, so **the honest, defensible
statement is a tie on NYU**. Do not cherry-pick a better E8 seed against a fixed
scalar_g8 seed.

CONSEQUENCE â€” contribution 2 MUST be reframed. "Lattice dimension is THE active
ingredient" is no longer supportable as a headline. Honest replacement:
"At matched rate and granularity, lattice VQ gives a small, dataset-dependent gain
over grouped scalar (significant on KITTI, neutral on NYU); the dominant lesson is
that a *fair* grouped-scalar baseline is far stronger than the per-tensor baseline
usually shown, and both reach ~FP32 parity at 4.0 eff bits." This is less flashy but
true, and pre-empts the reviewer who reruns our own scalar_g8.

## F20 â€” DG-2 CLOSED: rotation is inert at 4 bits, ESSENTIAL at 2 bits, with a mechanism

Rotation on vs off at 2-bit (E8, N=200):

| dataset | rot ON Î´1 | rot OFF Î´1 | Î” (rotation benefit) |
|---|---|---|---|
| NYU | 0.6631 | 0.6194 | **+0.0437** |
| KITTI | 0.6795 | 0.5826 | **+0.0969** |

Contrast F14 (4.0 eff bits): rotation benefit was ~0 (no-rot 0.9138 inside the
with-rot seed spread). So rotation's value GROWS as the bit-rate drops â€” exactly
what RHT theory predicts (its tail-suppression only becomes load-bearing when the
quantization grid is too coarse to absorb outliers).

MECHANISM (G3d activation stats, real NYU, 8 images, 40 K/V tensors):
  - **DinoV2 backbone spatial attention has heavy tails**: pretrained.blocks.0.attn
    V kurtosis = 20.1, blocks.1 V = 7.06, blocks.0 K = 4.13. Rotation ~halves them
    (20.1â†’10.1, 7.06â†’2.53, 4.13â†’2.11).
  - **The temporal KV cache (motion_modules) is ALREADY near-Gaussian**: kurtosis
    âˆ’0.4 to 0.66 raw, essentially unchanged by rotation (nothing to fix).
  - Median kurtosis 0.64 raw â†’ 0.24 rotated; max 20.1 â†’ 10.1.

Clean, publishable DG-2 = branch (b)+(c) hybrid: "The temporal KV cache we target is
already near-Gaussian, so rotation is unnecessary at the 4-bit operating point;
rotation's benefit is concentrated in the heavy-tailed backbone activations and only
becomes load-bearing below 3 bits. This both explains our null result at 4 bits and
predicts (confirmed) the 2-bit ablation." Rotation STAYS in the method (free at
inference, essential if anyone pushes below 4 bits, and it's what makes seeds
comparable) â€” reported as a rate-dependent ablation, not a headline contribution.

## F21 â€” F17 FALSIFIED: no temporal error accumulation across the window

G10, Sintel temporal, E8@3b vs FP32 at 4.0 eff bits, sweeping video_length:

| window | FP32 Î´1 | E8@3b Î´1 | Î” vs FP32 |
|---|---|---|---|
| 8  | 0.6232 | 0.6154 | âˆ’0.0078 |
| 16 | 0.6286 | 0.6281 | âˆ’0.0005 |
| 32 | 0.6309 | 0.6313 | +0.0004 |

The quantization gap SHRINKS with window length (âˆ’0.0078 â†’ +0.0004), the OPPOSITE
of the error-accumulation hypothesis. Longer temporal context makes the quantized
model MORE robust, not less. F17's mechanism guess is dead; report as a clean
negative result: "no evidence of cross-window quantization-error accumulation; the
gap is largest at the shortest window and vanishes by 32 frames." (The earlier
F17 Sintel gap was a full-split-vs-subset artifact; at matched subset it's ~0.)

## F22 â€” F16 structure diagnostic: PARTIAL corroboration (report honestly)

G11, per-frame min-max-normalized predicted disparity:

| Config | eff | grad_energy (NYU) | laplacian_var (NYU) | grad_energy (Sintel) |
|---|---|---|---|---|
| FP32 | 32 | 0.0478 | 0.00344 | 0.0350 |
| 3bit | 4.0 | 0.0465 | 0.00323 | 0.0354 |
| 2bit | 3.0 | **0.0346** | **0.00264** | **0.0282** |

At 2-bit, edge/detail energy drops sharply (NYU grad_energy âˆ’28%, laplacian_var
âˆ’23%; Sintel grad_energy âˆ’20%), while 8/4/3-bit track FP32 closely. This supports
the degeneracy direction (loss of fine spatial structure). BUT it is NOT a total
collapse: entropy_8bit and pred_std actually RISE at 2-bit (the output isn't a flat
blob; it loses coherent detail while gaining scattered value spread).

So F16's headline evidence stays the **per-scene TAE sign-flip** (hard scenes â†’
floor, easy scenes â†’ worse â€” airtight proof of gameability). The structure
diagnostic is SUPPORTING evidence for the "detail loss" reading, not a standalone
proof of collapse. Present both honestly; do not oversell grad_energy as "the output
became mush" â€” a reviewer with the numbers will see entropy went up.

## Net effect on the paper (for the record)

- Headline (DG-1): SOLID. 4.0 eff bits, 8Ã— vs FP32, lossless. Unchanged.
- Contribution 2 (lattice is key): WEAKENED to "modest, dataset-dependent" (F19).
- Rotation story (DG-2): STRENGTHENED â€” now mechanism-backed and rate-dependent (F20).
- TAE gameability (F16/F22): still the most novel piece; primary evidence intact.
- F17: dead (clean negative).
Contribution ranking for the writeup should now lead with TAE-gameability and the
"honest all-inclusive 4-bit lossless + fair-baseline" story, with lattice-vs-scalar
demoted to an honest matched-rate ablation, and rotation as a rate-dependent ablation.

