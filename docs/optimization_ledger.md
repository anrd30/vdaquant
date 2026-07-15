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

## T3 — QJL estimator + overhead fix (F3, F8) — OPEN
File: `research/quantizers/qjl_bias.py`, argparse in both scripts
- Apply ρ̂ → cos(π/2·(1−clamp(ρ̂,−1,1))) in `correct_scores`.
- Default m: `min(2·d, 128)` for d ≤ 128; norms storable at 8-bit (per-tensor scale).
- Replace `--use_qjl` with `--qjl {on,off}` (or `--no-qjl`); must be ablatable.
- Assertions: E[transform(ρ̂)] within ±0.05 of true cos θ at θ ∈ {30°, 45°, 60°, 90°}
  over 2000 trials, m=128, d=64; corrected-score MAE strictly < uncorrected at 3-bit;
  monotone improvement vs raw-ρ̂ variant at θ=45°.

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

## T7 — Temporal surgery + TAE metric (F4, F9) — OPEN
Files: `research/models/rotated_attention.py`, `scripts/run_pareto_benchmark_suite.py`
- Fix AnimateDiff head-reshape incompatibility (match `reshape_heads_to_batch_dim`
  convention exactly: (b, n, h·d) ↔ (b·h, n, d) ordering) and re-enable
  `replace_temporal=True` in the suite.
- Implement TAE: mean |D_t − warp(D_{t+1})| on aligned inverse depth; static-camera
  fallback = mean |D_t − D_{t+1}|. Report FP32 vs quantized TAE per bit-width.
- `RotatedSelfAttention.forward` must `raise NotImplementedError` if an
  attn_bias/mask kwarg is passed non-None (no silent drop).
- Assertions: temporal parity test — surgery output shape/tuple contract identical
  to original `TemporalAttention` on (b·f, d_tokens, c) input with video_length=f;
  fp32-surgery (bits=16, no-op quantizer) output within 1e-3 of original layer.

## T8 — Path to a true ≤4.0 effective bits/scalar (paper claim) — OPEN
Files: `research/quantizers/lattice_vq.py` (new `LatticeE8Quantizer`), suite
- E8 lattice (Λ8 = D8 ∪ D8+½): decode via best-of-two D8 cosets; group_size=8
  halves scale overhead; with 8-bit scales + QJL-lite(m=128, 8-bit norms):
  3-bit payload → 3 + 1 + 2.25 ≈ 6.25 b/scalar; without QJL 4.0 b/scalar.
- Boundary conditions: E8 decode must return exact lattice points (integer or
  half-integer coords, coordinate-sum even in the integer coset); MSE(E8) ≤ MSE(D4)
  at equal rate on N(0,1); document measured coding gain vs scalar (theory: 0.65 dB
  over D4).
- Gate: full suite at b=3/E8/8-bit scales/no-QJL reports eff ≤ 4.0 b/scalar and
  GT δ1 drop ≤ 0.02 on NYUv2 protocol from T2. This is the headline table.

---

Execution order: T5 → T6 → T1 → T4 → T3 → T2 → T7 → T8.
(T5/T6 first: nothing else is trustworthy until the test gate is real and the
transform is pure.)
