# Phase 4 — Path to an A* Paper (Master Plan)

**Author:** Fable (Lead Architect) · **Date:** 2026-07-20 · **Revised:** 2026-07-21 (Phase 3 complete)
**Extends:** `docs/optimization_ledger.md` Phase 3 (E1–E7). Read that first for findings F1–F17.

> ## ⚠ REVISION 2026-07-21 — Phase 3 landed; read this before anything below
>
> All 13 Phase 3 experiments completed. Findings **F12–F17** in the ledger supersede
> parts of this plan. Summary of what changed:
>
> - **DG-1 is CLOSED (F12).** The "+0.01 over FP32" was small-sample noise; it vanishes
>   on the full splits. **Headline locked: E8 @ 3-bit payload / 4.0 all-inclusive eff
>   bits is statistically indistinguishable from FP32** (NYU-654 −0.0014 δ1,
>   KITTI-1000 −0.0009 δ1). G2b now *confirms with CIs*; it no longer decides the story.
> - **E8@2-bit is DEAD (F13).** Collapses on all three datasets. The 3.0-eff-bit
>   headline dream is over; 4.0 is the frontier. **G9 is cancelled.**
> - **DG-2 leans hard to branch (c) (F14):** no-rotation sits *inside* the with-rotation
>   seed spread. G3a/G3b/G3d still required to close it, and G3b (2-bit) is now the
>   only place RHT could still earn its keep.
> - **QJL is strictly dominated (F15):** +2.25 eff bits for *worse* accuracy. Settled;
>   report as a negative result. No further QJL runs needed.
> - **F16 is the paper's best finding and it was unplanned:** TAE is *gameable* — the
>   collapsed 2-bit model posts a 7.5× better TAE than FP32. Contribution 3 is upgraded
>   accordingly, and **S6 (co-visibility mask) is now mandatory, not P1**.
> - **New work added:** **S10** (degeneracy diagnostic) and **G10** (temporal-window
>   sweep for F17). Both are cheap and both are load-bearing.
>
> Net effect on priorities: DG-3 (fair scalar baseline, S1→G1) is now the **single
> most important open item**, because it is the last thing standing between us and a
> defensible main claim.
**Execution model:** Sonnet implements each S-task below (prompts are verbatim-ready);
the boss reviews and pushes; GPU runs happen on the collaborator's local box via
`scripts/run_phase4_experiments.sh` (S9). All CPU tests must stay green at every commit.

---

## 0. Venue and clock

NeurIPS 2026 is gone (deadline was May 2026). Realistic A* targets, in order:

| Venue | Deadline (verify on venue site — do not trust these dates blindly) | Fit |
|---|---|---|
| **ICLR 2027** | abstract ~mid-Sep 2026, paper ~late Sep 2026 | Best fit: quantization/representation crowd, ablation-heavy papers do well |
| **CVPR 2027** | ~mid-Nov 2026 | Backup: vision crowd loves depth + qualitative figures; quantization less central |
| NeurIPS 2027 | May 2027 | Fallback with maximal polish time |

**Primary target: ICLR 2027 → ~9 weeks from today.** The plan below is paced to that.
First action item for the boss: verify the exact ICLR 2027 dates at iclr.cc.

---

## 1. Where we stand (honest gap analysis)

### Assets already banked
- Working RHT + D4/E8 lattice KV-cache quantization of VDA temporal attention, 96 CPU tests green.
- Real-GT protocol matching VDA's own eval constants; disparity-space alignment bug (F10) found and fixed.
- Honest all-inclusive bit accounting (payload + scale metadata + QJL side channel) — most competing papers don't do this.
- E1–E4 core results (N=200): matched-rate E8@3b (4.0 eff) = **0.9128 NYU / 0.9194 KITTI δ1** vs FP32 0.9036 / 0.9275.
- **F11 resolved by E1b**: at matched 4.0 eff bits and matched scale machinery, E8@3b 0.9128 vs D4@2b 0.5316 (NYU) — lattice dimension isolated.
- Seed robustness: 0.9128 / 0.9125 / 0.9143 across RHT seeds 0/1/2. Tight.
- Geometric-reprojection TAE with z-buffer on Sintel; median as honest headline.
- E5 full-split finals + E6 qualitative figures running now.

### What an A* reviewer will attack, today
| # | Attack | Current state | Fix |
|---|---|---|---|
| A1 | "Your scalar baseline is a strawman — nobody ships per-tensor scales; KIVI/KVQuant use group scales." | True. E1a scalar has ONE scale per tensor. | **G1** (scalar_g8 baseline) — non-negotiable |
| A2 | "Quantization beats FP32 (+0.01 δ1)? You have a bug." | E4a shows 16-bit scales ≈ FP32, 8-bit scales +0.01 → scale-noise regularization artifact | **G2** (scale-bits sweep + paired bootstrap, full split, 3 seeds) → lock headline claim |
| A3 | "Your rotation ablation shows rotation does nothing (E2 no-rot NYU = 0.9128 = with-rot). Why is RHT in the title?" | Unexplained. Either dead flag or true null. | **G3** (control run, 2-bit extension, activation-stats evidence) → lock narrative |
| A4 | "One model (ViT-S). Does this generalize?" | vits only | **G4** (vitl) |
| A5 | "No comparison to published KV-cache methods." | Only internal baselines | **G1** covers the KIVI-style scalar-group family; lit review positions the rest honestly (we compare *families* under identical protocol, not reimplementations under unknown protocols) |
| A6 | "Where are the error bars?" | Point estimates only | **S2** (per-image dumps + paired BCa bootstrap) |
| A7 | "TAE mean is inflated by disocclusion." | Known (median used) | **E7** (GT co-visibility mask) |
| A8 | "What does this buy me in memory?" | Nothing reported | **G8/S7** (analytic + measured KV memory table) |
| A9 | "Is this actually novel? QuIP# has E8, QuaRot rotates KV caches, CQ does VQ on KV." | Unchecked against 2025–26 lit | **Section 5** lit review with kill criteria — run BEFORE writing anything |
| A10 | "D4 non-monotonic on NYU (3b 0.9078 > 4b 0.8961)?" | Unexplained; likely same scale-noise as A2 | Covered by G2's mechanism; verify with G2 data, disclose in appendix |

### The one potential wow-result nobody has noticed yet
E8@2-bit = **3.0 effective bits/scalar** (2 payload + 8/8 scale). D4@2b collapses (0.53/0.38)
and scalar@2b collapses (0.49/0.37) — but E8@2b is only being measured now, in E5.
If E8@2b holds ≥ ~0.85 δ1 where every baseline at 3.0-ish eff bits collapses, the paper's
title claim moves from "4 bits" to "3 bits" and the Pareto figure becomes dramatic.
**Watch the E5 output for the 2-bit E8 rows before locking the story.**

---

## 2. Decision gates (the story is decided by data, in this order)

**DG-1 — Headline claim (after G2).**
- If full-split, 3-seed, 16-bit-scale E8@3b is within CI of FP32 → headline = **"lossless at ≤5.0 eff bits, near-lossless at 4.0"**. The +0.01 at 8-bit scales is reported as a regularization curiosity in the appendix, never headlined.
- If the +0.01 survives 16-bit scales on the full split (unlikely) → we have something strange; investigate before any claim.
- Never, under any circumstance, headline "quantization improves accuracy."

**DG-2 — Rotation narrative (after G3).**
- Branch a: `--no-rotation` flag is dead (control run G3a prints identical numbers) → fix bug, rerun E2, re-enter this gate.
- Branch b: rotation genuinely null at 3–4 bits but matters at 2 bits → story: "RHT is the enabler of the extreme-rate regime" (clean, strong).
- Branch c: rotation null even at 2 bits, AND activation stats (G3d) show ViT KV activations are already near-Gaussian/low-kurtosis → story: **"contra LLM practice (QuaRot/SpinQuant), video ViT KV caches do not need rotation; lattice coding gain is the active ingredient."** This is a publishable, contrarian, evidence-backed finding — do not fear it. RHT moves from "contribution" to "we ablate it away."

**DG-3 — Lattice margin (after G1).**
- If E8@3b (4.0 eff) beats scalar_g8@3b (4.0 eff) by ≥ ~0.01 δ1 with non-overlapping CIs on both datasets → main claim intact.
- If the gap is inside noise at 4.0 eff → the story moves DOWN the rate axis: compare at 3.0 eff (E8@2b vs scalar_g8@2b) where lattice structure should dominate. If E8 wins there, the paper becomes "the 3-bit frontier"; if nothing separates anywhere, the honest paper is protocol + TAE + negative result → retarget venue (workshop). State this now so nobody is surprised later.

**DG-4 — Novelty (after Section 5 lit review).**
- Known adjacent results to check for and position against (do not assume these are the worst case — search for worse):
  - QuIP# (E8 lattice, weights), QTIP (trellis, weights), AQLM/GPTVQ (learned VQ, weights)
  - QuaRot / SpinQuant (rotation incl. KV cache, scalar quant)
  - CQ "Coupled Quantization" (learned/coupled VQ **on KV cache** — closest known prior; our delta: data-oblivious lattice, no calibration, video domain, temporal metric)
  - KIVI / KVQuant / QJL / ZipCache / GEAR (scalar KV cache)
- Kill condition: a 2024–26 paper doing **lattice (data-oblivious) VQ on a KV cache**. If found → reposition on video-geometry + TAE + extreme-rate angle (still viable). If also found on a *video* model → escalate to boss immediately; the framing meeting happens before any more GPU time is spent.

---

## 3. Phase 4 experiment matrix (G-series)

All runs N=200 unless stated. Same conventions as Phase 3: `--rht-seed 0`, `--no-qjl`
on every comparison row. Runtime estimates from observed Phase 3 timings (2–5 min/experiment
on the collaborator's GPU; vitl ~3× that).

| ID | What | Command sketch | Cost | Blocks | Priority |
|---|---|---|---|---|---|
| G1 | Fair scalar-group baseline (KIVI-style), NYU+KITTI, bits 4/3/2 | `--quantizer scalar_g8 --scale-bits 8` | ~10 min | S1 | **P0** |
| G2a | Scale-bits sweep at E8@3b NYU: add {6, 12} to existing {8, 16} | `--scale-bits 6` / `12` | ~5 min | — | P0 |
| G2b | **Decisive**: full-654 NYU, E8@3b, scale-bits {8,16} × seeds {0,1,2} | 6 runs, `--max-samples 654` | ~1 h | S2 (per-image dump) | **P0** |
| G3a | Rotation control: with-rotation, same code version/N as E2 | headline config, fresh run | ~3 min | — | **P0** |
| G3b | Rotation ablation at 2-bit, NYU+KITTI | `--no-rotation --bits 2` + with-rotation `--bits 2` | ~10 min | — | P0 |
| G3c | (only if G3a says flag dead) fix + rerun E2 all | — | ~15 min | bugfix | conditional |
| G3d | Activation stats: kurtosis/outliers pre/post RHT | `scripts/dump_activation_stats.py` | ~5 min | S3 | P0 |
| G4 | VDA-Large: FP32 + E8@{4,3} + scalar_g8@3, NYU+KITTI | `--encoder vitl` | ~45 min | S4 | P1 |
| G5 | K-only / V-only ablation, NYU E8@3b | `--quantize-target k` / `v` | ~6 min | S5 | P1 |
| E7 | Sintel temporal rerun with GT co-visibility mask | e5c config + `--tae-covis` | ~30 min | S6 | P1 |
| G8 | KV memory table (analytic + one measured run) | `scripts/report_kv_memory.py --measure` | ~5 min | S7 | P1 |
| ~~G9~~ | ~~E8@2b vs scalar_g8@2b at 3.0 eff~~ | **CANCELLED** — F13: E8@2b collapses everywhere; there is no 3.0-eff story to chase | — | — | — |
| G10 | **NEW (F17)**: temporal-window sweep {8,16,32} at 4.0 eff bits, Sintel, to test error accumulation | `--temporal-window 8` / `16` / `32` | ~25 min | — | **P0** |
| G11 | **NEW (F16)**: degeneracy diagnostic across all bit-widths, Sintel + NYU | `scripts/dump_structure_stats.py` | ~10 min | S10 | **P0** |
| F-v2 | Finals v2: whatever configs the gates lock, full splits (654/1000/all-Sintel), 3 seeds for headline | — | ~3 h | all gates | P0 (last) |

**Note on G2b after F12:** its job changed. It no longer decides whether we have a
bug — F12 already showed the +0.01 was subset noise. It now supplies the *confidence
intervals* that let the paper say "indistinguishable from FP32" rigorously, and
settles the residual scale-bits question (E4a sb=16 gave 0.9029 vs E3 sb=8 seeds
giving 0.9134/0.9148 on identical images — different seeds, so confounded).

**GPU total: roughly 6–7 hours across the phase, comfortably one weekend on the local box.**

Run order: G1, G2, G3 first (they decide the story). G4–G8 after. F-v2 dead last —
never burn full-split time on a config that a gate might kill.

---

## 4. Sonnet task specs (S1–S9) — verbatim prompts

Rules that apply to EVERY task (paste along with each prompt):

> You are Sonnet, the implementation agent on vdaquant. Constraints: CPU-only local
> testing, no dataset downloads, no GPU runs. Run `pytest tests/ -q` before you start
> (expect 96+ passing — the count grows as tasks land) and after every task; all
> existing tests must stay green. Match surrounding code style and comment density.
> Commit per task with the given message; DO NOT push — the boss reviews first.
> If you find yourself wanting to change a file not listed under "Files to touch",
> stop and report instead. After any function edit, re-run a static undefined-name
> check (AST walk) on that function — this codebase has been bitten by stale-variable
> bugs before (see ledger F-series).

---

### S1 — `ScalarGroupQuantizer` (fair KIVI-style baseline) — P0, do first

```
TASK S1: Add a group-wise scalar quantizer named 'scalar_g8' so the paper has a
fair scalar baseline at matched scale granularity (resolves ledger F11, option (a)).

CONTEXT: research/quantizers/lattice_vq.py has ScalarRoundQuantizer which, as called
from research/models/rotated_attention.py (two call sites, ~lines 200 and 383, both
passing no per_channel flag), uses ONE global scale for the entire tensor
(x_max = x.abs().amax()). E8 uses one 8-bit scale per 8 elements. Comparing them
conflates lattice coding gain with scale granularity. scalar_g8 fixes that: scalar
rounding, but one scale per contiguous group of 8 along the last dim — exactly the
scale machinery E8 pays for.

FILES TO TOUCH:
  research/quantizers/lattice_vq.py        (new class)
  scripts/run_pareto_benchmark_suite.py    (registration + accounting)
  research/models/rotated_attention.py     (factory/name dispatch only)
  tests/test_scalar_group.py               (new)
  tests/test_bit_accounting.py             (extend)

SPEC:
1. New class ScalarGroupQuantizer(bits, group_size=8) in lattice_vq.py:
   - forward(x): view last dim as (..., n_groups, group_size); assert
     x.shape[-1] % group_size == 0 with a clear error message.
   - Per group: scale = amax(|x_g|) / (2**(bits-1) - 1), guarded against zero
     (scale=1.0 where amax==0, mirroring ScalarRoundQuantizer's guard — read it first).
   - Symmetric round-to-nearest: q = clamp(round(x_g/scale), -(2**(bits-1)-1), +(2**(bits-1)-1));
     dequant = q * scale. Reshape back. No learned anything, no calibration.
   - Follow the same simulated-quantization interface the other quantizers expose
     (read LatticeE8Quantizer for the exact contract: same method names, same
     dtype/device handling, same fp16-safety pattern from T6).
2. Registration: QUANTIZER_GROUP_SIZE['scalar_g8'] = 8 in run_pareto_benchmark_suite.py
   so resolve_group_size charges scale_bits/8 per scalar — identical overhead to E8.
   Add 'scalar_g8' to the --quantizer choices and to whatever name->class factory
   rotated_attention.py uses (grep for how 'lattice_e8' is dispatched; mirror it exactly).
3. TESTS (tests/test_scalar_group.py):
   a. Exactness: at bits=8, a tensor with values already on the quantization grid
      reconstructs bit-exactly.
   b. Group independence (THE point of this class): build x of shape (1, 16) where
      group 0 contains an outlier (e.g. 100.0) and group 1 contains small values
      (~0.01). Assert group 1's reconstruction error is unaffected by group 0's
      outlier (compare against quantizing group 1 alone). Then show
      ScalarRoundQuantizer on the same tensor DOES destroy group 1 (error orders
      of magnitude larger) — this asserts the pathology we claim exists, exists.
   c. Zero-guard: all-zeros tensor -> all-zeros out, no NaN/inf.
   d. Shape/dtype: (2,3,7,64) fp32 and fp16 pass through with same shape/dtype;
      head_dim=64 divides by 8 cleanly.
   e. Accounting (in test_bit_accounting.py): scalar_g8 at bits=3, scale_bits=8
      -> exactly 4.0 effective bits/scalar; bits=4 -> 5.0.
4. ACCEPTANCE: pytest green; then a CPU smoke run must complete:
   python scripts/run_pareto_benchmark_suite.py --dataset nyuv2 --eval-mode fidelity \
     --quantizer scalar_g8 --bits 3 --scale-bits 8 --no-qjl --max-samples 1
   (fidelity mode needs no dataset download — confirm before running; if it does
   need one, stop and use whatever offline fixture path the existing tests use.)
COMMIT: "feat(S1): scalar_g8 group-wise baseline quantizer with matched scale accounting"
```

---

### S2 — Per-image metric dumps + paired bootstrap stats — P0

```
TASK S2: Give every ground-truth eval per-image metric dumps, and add a stats
script that computes paired bootstrap confidence intervals between any two configs.
This is what turns "0.9128 vs 0.9036" into a defensible claim (reviewer attack A6,
and it decides ledger decision gate DG-1).

FILES TO TOUCH:
  scripts/run_pareto_benchmark_suite.py   (per-image dump)
  scripts/compute_stats.py                (new)
  tests/test_stats.py                     (new)

SPEC:
1. In run_groundtruth_eval, alongside the existing aggregates, record per-sample
   metrics: for each evaluated image i, store {"idx": i, "delta1": ..., "absrel": ...,
   "rmse": ...} into the results JSON under each config as key "per_image".
   Also store it for the FP32 baseline pass. Images skipped for zero valid pixels
   must be skipped identically in all configs (they already are — skip logic depends
   only on GT; do not change it, just preserve index alignment in the dump).
   Guard: this grows the JSON by ~200-1000 small dicts per config — fine; do NOT
   dump per-pixel anything.
2. scripts/compute_stats.py CLI:
     python scripts/compute_stats.py results.json --config-a FP32 --config-b <name> \
       --metric delta1 --n-boot 10000 --seed 0
   - Loads per_image arrays for both configs, asserts identical idx lists (hard fail
     if not — misalignment silently invalidates pairing).
   - Computes paired differences d_i = metric_b_i - metric_a_i, reports mean diff,
     and a 95% BCa bootstrap CI of the mean diff (implement BCa properly: bias
     correction z0 from the proportion of bootstrap means below the observed mean,
     acceleration a from jackknife). numpy only, seeded Generator, deterministic.
   - Also print per-config mean ± bootstrap std, and whether the CI excludes 0.
   - --all flag: compare every config against FP32 and emit a markdown table +
     stats.json next to the input file.
3. TESTS:
   a. Synthetic: two arrays with known constant shift 0.01 and noise sd 0.001,
      n=200 -> CI must contain 0.01 and exclude 0 (seeded, deterministic).
   b. Null case: identical arrays -> CI contains 0, mean diff == 0 exactly.
   c. Misaligned idx lists -> hard error, not a silent result.
   d. Determinism: same seed twice -> byte-identical stats.json.
4. ACCEPTANCE: pytest green. compute_stats.py runs against a small fixture JSON
   you create under tests/fixtures/ (hand-written, 5 images, 2 configs).
COMMIT: "feat(S2): per-image metric dumps + paired BCa bootstrap stats script"
```

---

### S3 — Activation statistics dump (rotation evidence) — P0

```
TASK S3: A diagnostic script that measures WHY rotation does or doesn't matter:
per-layer outlier statistics of K and V activations, before and after RHT.
This is the evidence for decision gate DG-2 branch (c) and paper figure F2.

CONTEXT: E2 shows --no-rotation losing nothing at 3-4 bits on this model. The LLM
literature (QuaRot, SpinQuant) says rotation kills activation outliers that wreck
scalar quantization. If ViT KV activations are already low-kurtosis, rotation
genuinely has nothing to fix at moderate bits — we need the numbers either way.

FILES TO TOUCH:
  scripts/dump_activation_stats.py   (new)
  tests/test_activation_stats.py     (new)

SPEC:
1. CLI: python scripts/dump_activation_stats.py --dataset nyuv2 --num-images 8 \
          --output-dir outputs/activation_stats
   Runs the model twice via the existing surgery machinery:
   pass 1 with rotation ON + IdentityQuantizer (captures POST-RHT K,V),
   pass 2 with rotation OFF (identity) + IdentityQuantizer (captures RAW K,V).
   Capture via forward hooks on the temporal-attention K/V points — find the exact
   tensors the quantizer sees by reading rotated_attention.py; the statistic must be
   computed on EXACTLY what gets quantized (post-reshape, per-head, head_dim=64),
   not on some upstream tensor.
2. Per layer, per tensor (K and V separately), computed over all captured tokens:
   - excess kurtosis of the flattened values
   - max|x| / RMS (outlier ratio)
   - per-channel amax spread: max_c(amax_c) / median_c(amax_c) over the 64 channels
   Save one JSON: {layer, tensor, rotated: bool, kurtosis, outlier_ratio, chan_spread,
   n_tokens}. Also save matplotlib histograms (log-y) for 2 representative layers
   (first temporal layer, last temporal layer) x {raw, rotated} x {K, V} as PNGs.
3. Keep it CPU-runnable at tiny scale for the test: --num-images 1 --max-tokens 512
   must work without GPU or dataset (accept a --synthetic flag that feeds
   torch.randn frames through the model instead of dataset frames; the hook
   machinery is what's under test, not the data).
4. TESTS: run the script with --synthetic --num-images 1; assert the JSON exists,
   has entries for both rotated and raw, both K and V, kurtosis is finite, and
   (sanity) rotated kurtosis of a deliberately heavy-tailed synthetic input fed
   through JUST the HadamardRotation module (unit-level check, not full model) is
   substantially lower than raw kurtosis — RHT gaussianizes, that's its one job
   (Ailon-Chazelle). Use a fixed seed.
5. INTERPRETATION KEY (write into the script's docstring so the boss can read
   results without you): raw kurtosis ~0 (Gaussian-ish) across layers => rotation
   has nothing to fix at 3-4 bits => DG-2 branch (c). Raw kurtosis high (>5) but
   rotation still doesn't help accuracy => suspicious, escalate. Raw kurtosis high
   and rotated ~0 => rotation should matter at low bits => expect G3b to show it.
COMMIT: "feat(S3): pre/post-RHT activation outlier statistics dump"
```

---

### S4 — VDA-Large (vitl) support — P1

```
TASK S4: Add --encoder {vits,vitl} to the benchmark suite so G4 can test whether
results transfer to VDA-Large (reviewer attack A4).

FILES TO TOUCH:
  scripts/run_pareto_benchmark_suite.py
  tests/test_encoder_configs.py   (new)

SPEC:
1. run_pareto_benchmark_suite.py:1322 hardcodes
   model_configs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}.
   Replace with a dict keyed by encoder:
     'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]}
     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
   VERIFY the vitl values against the VDA repo in this codebase
   (Video-Depth-Anything/ — grep for vitl in their model builder / README) rather
   than trusting this prompt. If they differ, the repo wins; note the difference
   in your report.
2. Checkpoint handling (~lines 1325-1371): parametrize filename and URL by encoder:
     vits: video_depth_anything_vits.pth (existing URL)
     vitl: video_depth_anything_vitl.pth from
           https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
   Keep the existing corrupt-checkpoint removal + auto-download pattern. Do NOT
   download anything yourself; the URL is exercised on the GPU box.
3. Surgery: rotated_attention surgery must be dimension-agnostic. ViT-L is
   dim 1024 / 16 heads / head_dim 64 — head_dim stays 64, so HadamardRotation(64)
   and all group sizes are unchanged. VERIFY by reading the surgery code that
   nothing assumes num_heads=6 or embed_dim=384; if something does, fix it
   generically (derive from the module's own shapes), and add a unit test with a
   mock attention at dim 1024/16 heads proving surgery output matches source
   module in identity mode (reuse the pattern in the existing T7 tests).
4. Output labeling: results JSON and printed table must record the encoder so
   vits/vitl results can never be silently mixed.
5. TESTS: config selection returns right dict per flag; unknown encoder -> clear
   error; mock-surgery-at-1024 test from (3).
COMMIT: "feat(S4): --encoder vitl support with dimension-agnostic surgery"
```

---

### S5 — K-only / V-only quantization target — P1

```
TASK S5: Add --quantize-target {kv,k,v} (default kv) so G5 can measure K/V
asymmetry — a standard ablation in every KV-cache paper (KIVI found K and V need
different treatment; we should know if the same holds here).

FILES TO TOUCH:
  scripts/run_pareto_benchmark_suite.py   (flag + threading)
  research/models/rotated_attention.py    (per-path quantizer selection)
  tests/test_quantize_target.py           (new)

SPEC:
1. Thread the flag suite -> surgery -> attention constructor. When target='k',
   the V path gets IdentityQuantizer (and vice versa). Rotation stays ON for both
   paths regardless (rotation is free at eval; we're ablating quantization only —
   document this choice in the flag's help text).
2. Bit accounting: when target != 'kv', the results JSON must record
   {"k_bits": ..., "v_bits": "fp16"} style fields and the printed row must say so
   explicitly (e.g. "K@3b, V@fp16"). Do NOT average the two into a single
   misleading eff-bits number; report eff bits for the quantized tensor and label
   the other as fp16. Grep how compute_real_bit_accounting flows into the table
   and extend it honestly.
3. TESTS: with a mock attention (reuse T7 mock pattern), target='k' => V output
   bit-exact vs unquantized reference while K output differs; target='v' mirror;
   target='kv' matches current behavior exactly (regression guard).
COMMIT: "feat(S5): --quantize-target k/v/kv ablation flag with honest labeling"
```

---

### S6 — GT co-visibility mask for TAE (E7) — P1

```
TASK S6: Implement the GT-derived co-visibility mask for the geometric TAE so the
masked mean is honest (ledger E7; reviewer attack A7). Currently disocclusion in
violent-motion Sintel scenes inflates the mean and we fall back to the median.

CONTEXT: read _tae_geometric_single and compute_tae_geometric_for_scene in
scripts/run_pareto_benchmark_suite.py, and the z-buffer test in
tests/test_temporal_tae_path.py (which encodes the collision semantics — build on
that test's construction pattern; note the file previously had a docstring
syntax-error incident at line ~317, keep docstrings as actual strings).

SPEC:
1. Co-visibility: a pixel in frame t is co-visible in t+1 iff reprojecting the GT
   depth of frame t into t+1 (same K, relative pose machinery already in
   _tae_geometric_single) lands within the image AND the reprojected GT depth
   agrees with frame t+1's own GT depth at the landing pixel:
     |z_reproj - z_gt_{t+1}| / z_gt_{t+1} < tau   (default tau=0.05)
   Pixels failing any condition (out of frame, z-buffer occluded, depth mismatch)
   are excluded from the MASKED TAE. Compute the mask from GT ONLY — never from
   predictions, so FP32 and quantized configs get the IDENTICAL mask (same
   principle as the skip logic: read ledger F-series on why).
2. Flag: --tae-covis-tau (default 0.05); report three numbers per scene and
   aggregate: tae_mean (unmasked, existing), tae_median (existing), tae_covis_mean
   (new), plus covis_fraction (fraction of otherwise-valid pixels kept). All four
   go into the results JSON and the printed table gains a TAE-covis column.
3. TESTS (extend tests/test_temporal_tae_path.py):
   a. Synthetic scene with a translating foreground square over a far background
      (adapt the existing z-buffer collision construction): the disoccluded strip
      behind the square must be EXCLUDED by the mask — assert covis_fraction < 1
      and that masked TAE over the remaining pixels equals the hand-computed value.
      Verify numerically BEFORE writing the assert (this file has history: a
      previous version asserted a vacuous 0-translation case; do not repeat that).
   b. Static scene, zero motion: mask keeps everything (covis_fraction == 1),
      masked mean == unmasked mean.
   c. tau monotonicity: smaller tau -> covis_fraction non-increasing.
COMMIT: "feat(S6): GT-derived co-visibility mask for geometric TAE (E7)"
```

---

### S7 — KV-cache memory table — P1

```
TASK S7: scripts/report_kv_memory.py — the "what does this buy me" table
(reviewer attack A8). Analytic bytes, exact, with one optional measured run.

SPEC:
1. Read the VDA temporal attention implementation in this repo
   (Video-Depth-Anything/) and determine, from code, for each encoder:
   - number of temporal attention layers that cache K,V
   - tokens per frame at 518-px inference (patch 14 => spatial tokens; count any
     extra tokens the code actually keeps — cls/register — from the code, not
     from memory)
   - head_dim x num_heads = channel dim of cached K and V per layer
   Document every one of these with a file:line citation in the script docstring.
   If VDA's inference actually processes windows of W frames (grep the windowing
   in their inference wrapper; the suite uses 32-frame windows for temporal mode),
   parametrize by W with default matching our eval.
2. Analytic table: bytes = layers x 2(K,V) x W x tokens_per_frame x dim x
   (eff_bits/8), for eff_bits in {16 (fp16), 9.0, 5.0, 4.0, 3.0} and W in {8, 32}
   at both encoders. Emit a markdown table (rows: encoder x W; cols: precisions)
   with GB values and the compression ratio vs fp16. Print + write
   outputs/kv_memory_table.md.
3. --measure flag (GPU box only): wraps one real 32-frame temporal window forward
   in torch.cuda.reset_peak_memory_stats / max_memory_allocated for fp16 baseline
   vs quantized-sim config and prints both. Note in output that our quantizers are
   SIMULATED (dequantized on the fly), so measured peak does NOT show the savings —
   the analytic table is the deployable number; the measured number is only there
   for transparency about the sim. Make the script say this explicitly so nobody
   quotes the wrong number in the paper.
4. TESTS: analytic function is pure arithmetic — unit test it against a hand
   computed case (e.g. 1 layer, W=2, 4 tokens, dim 8, 4.0 eff bits => exact bytes).
   CLI runs CPU-only without --measure.
COMMIT: "feat(S7): analytic KV-cache memory table + optional measured peak"
```

---

### S8 — Figure factory — P1 (after S2 lands; extend as results arrive)

```
TASK S8: scripts/make_paper_figures.py — every paper figure from results JSONs,
one command, deterministic, colorblind-safe. No screenshots, no hand-editing.

SPEC:
1. Input: globs outputs/phase3/*/pareto_benchmark_results.json and
   outputs/phase4/*/pareto_benchmark_results.json (tolerate missing dirs), plus
   optional stats.json (S2) and activation_stats JSON (S3).
2. Style module at top: Okabe-Ito palette, single font family, shared figsize
   constants, PDF output (vector) into outputs/figures/. Every figure function
   is independently callable; --only <name> flag.
3. Figures:
   F1 pareto: delta1 (y) vs effective bits/scalar (x), one panel per dataset
      (NYU, KITTI), one colored line+markers per quantizer family (scalar,
      scalar_g8, D4, E8), FP32 as dashed horizontal asymptote, every point
      labeled with payload bits. THE money figure — make it clean.
   F2 activations: per-layer kurtosis raw vs rotated (two lines per tensor K/V),
      plus the 4 histograms from S3 as an inset grid.
   F3 tae: per-scene Sintel TAE distributions (violin or box) FP32 vs bit-widths;
      annotate mean vs median vs covis-mean so the disocclusion story is visible.
   F4 scalebits: delta1 vs scale-bits {6,8,12,16} at E8@3b with bootstrap CI band
      (from stats.json), FP32 band overlaid — this figure resolves DG-1 visually.
   F5 qual: grid montage from the E6 PNG dirs (rgb | gt | fp32 | e8-4b | e8-3b |
      e8-2b), 3 rows per dataset. Load PNGs; do not re-run models.
   F6 seeds: tiny strip plot of delta1 across rht seeds {0,1,2} (reads E3 + G2b).
4. Missing-input behavior: each figure that lacks its inputs prints a SKIPPED
   line naming the missing file — never a traceback, never an empty PDF.
5. TESTS: check in tiny hand-written fixture JSONs under tests/fixtures/figures/;
   run the factory against them; assert PDFs exist and are >1KB; assert a run with
   an empty results dir yields all-SKIPPED and exit code 0.
COMMIT: "feat(S8): deterministic paper figure factory"
```

---

### S9 — Phase 4 runner — P0 (small; after S1–S3 land)

```
TASK S9: scripts/run_phase4_experiments.sh mirroring run_phase3_experiments.sh
exactly (same run() helper, same skip-if-json-exists resumability, same
conventions: --rht-seed 0 default, --no-qjl on all comparison rows). Read that
script first and copy its structure.

STAGES:
  STAGE=gates  -> G1, G2a, G2b, G3a, G3b, G3d   (the decision-gate runs)
  STAGE=extend -> G4, G5, E7-rerun (e5c config + --tae-covis), G8 (with --measure)
  STAGE=all    -> both (default)
Exact configs are in docs/phase4_a_star_plan.md section 3 — transcribe them
faithfully; G2b is 6 runs (scale-bits {8,16} x seeds {0,1,2}, --max-samples 654,
NYU, E8@3b) each in its own output dir outputs/phase4/g2b_sb<S>_seed<N>/.
G3a is the current headline config rerun verbatim into outputs/phase4/g3a_control/.
Finals-v2 is NOT in this script — it gets written only after the gates close.
COMMIT: "feat(S9): resumable Phase 4 experiment runner (gates + extend stages)"
```

---

### S10 — Degeneracy / structure diagnostic (NEW, from F16) — P0

```
TASK S10: Quantify the degeneracy that makes the 2-bit model "win" on TAE, so the
paper's central methodological claim rests on numbers rather than on an argument.

CONTEXT — read ledger finding F16 first. On Sintel, E8@2-bit collapses on accuracy
(delta1 0.7122 -> 0.5017, AbsRel 0.2335 -> 0.3767) yet posts a 7.5x BETTER TAE mean
than FP32 (7.38% vs 55.41%) and the best median of any config. Per-scene analysis
shows the change flips sign with scene difficulty: hard high-motion scenes fall to a
floor (ambush_2: 906 -> 12.7) while easy near-static scenes get WORSE (shaman_2:
1.00 -> 3.90). Hypothesis: the 2-bit prediction has lost spatial structure, so
there is little left to misalign under reprojection. We need to measure that
structure loss directly.

FILES TO TOUCH:
  scripts/dump_structure_stats.py   (new)
  tests/test_structure_stats.py     (new)

SPEC:
1. CLI mirroring the benchmark suite's dataset/quantizer flags:
     python scripts/dump_structure_stats.py --dataset sintel --quantizer lattice_e8 \
       --scale-bits 8 --bits 8 4 3 2 --no-qjl --max-samples 200 \
       --output-dir outputs/phase4/g11_structure
   For FP32 and each bit-width, compute over the predicted disparity maps
   (the model's native output space — do NOT invert to depth; inversion is
   nonlinear and would confound the structure measurement):
     - grad_energy: mean of sqrt(dx^2 + dy^2) via Sobel, per frame, then averaged
     - pred_std: per-frame spatial standard deviation, then averaged
     - laplacian_var: variance of the Laplacian (standard blur/structure measure)
     - entropy_8bit: Shannon entropy of the 256-bin histogram of the min-max
       normalised map, per frame, averaged
   All four are normalised per frame (min-max) BEFORE computing, so an affine
   collapse of the prediction range does not masquerade as structure loss —
   we want to detect loss of STRUCTURE, not loss of scale. Document this choice
   in the docstring; it is the subtle part and a reviewer will ask.
2. Output: JSON {config -> {metric -> value}} plus a markdown table, and a
   matplotlib line plot of each metric vs effective bits with FP32 as a dashed
   reference line. Save to the output dir.
3. INTERPRETATION KEY (write into the docstring): if grad_energy and
   laplacian_var drop sharply at 2-bit while 3/4/8-bit track FP32 closely, the
   degeneracy hypothesis is CONFIRMED and F16 can be stated as fact in the paper.
   If they do NOT drop, the hypothesis is WRONG — the TAE improvement would then
   need another explanation, and the boss must be told before any claim is written.
   Do not tune the metrics to produce the expected answer.
4. TESTS: pure-function unit tests on synthetic inputs — a sharp random-noise
   image vs a heavily Gaussian-blurred copy: assert blurred has strictly lower
   grad_energy, lower laplacian_var, and lower entropy. A constant image gives
   grad_energy == 0 and does not produce NaN in any metric. Seeded, deterministic.
   Test the metric functions directly; do not require a model or dataset.
COMMIT: "feat(S10): spatial-structure diagnostic for the TAE degeneracy finding (F16)"
```

---

## 5. Literature review plan (run BEFORE writing; ~1 day; Fable executes via web search)

Not written now — but the searches happen **this week**, because DG-4 can re-aim the
whole paper. Buckets, queries, and what each can kill:

| Bucket | Queries | What we need | Kill risk |
|---|---|---|---|
| Lattice/VQ weight quant | "QuIP# E8 lattice", "QTIP trellis quantization", "AQLM", "GPTVQ" | position: they do weights, we do KV activations, no calibration | low |
| **VQ on KV cache** | "coupled quantization KV cache", "vector quantization KV cache 2025", "codebook KV cache compression", "lattice KV cache" | **the kill-shot search.** CQ (2024) is known-closest: learned coupled centroids. If anyone did *data-oblivious lattice* KV → DG-4 | **high — do first** |
| Scalar KV cache quant | KIVI, KVQuant, QJL, ZipCache, GEAR, Atom, "KV cache quantization survey 2026" | baseline family citations; confirm group-scale is their standard (justifies scalar_g8 as THE fair baseline) | low |
| Rotation | QuaRot, SpinQuant, DuQuant, "Hadamard rotation KV cache", "rotation vision transformer quantization" | position DG-2 branch (c) against their claims; check if anyone already reported "ViTs don't need rotation" | medium |
| Video depth | Video-Depth-Anything, DepthCrafter, ChronoDepth, "streaming video depth 2025 2026" | confirm nobody quantized these; get their TAE definitions (VDA's is flow-based — ours is geometric: differentiate explicitly) | medium |
| ViT PTQ | PTQ4ViT, RepQ-ViT, "vision transformer post-training quantization survey" | related-work paragraph; check for ViT KV-cache-specific work | low |
| Temporal metrics | "temporal alignment error depth", "temporal consistency metric video depth", OPW | cite lineage of TAE; establish what's new in geometric+z-buffer+covis version | low |

Output artifact: `docs/related_work_notes.md` — one entry per paper: venue/year,
one-line method, their rate (bits), their protocol, exact delta vs us, threat level.
Any HIGH threat found → stop, boss meeting, before more GPU is spent.

---

## 6. Paper blueprint

### Story (pending gates; current best guess)
Streaming video-depth models must cache temporal-attention K/V; at fp16 this cache
is the memory bottleneck on edge devices. We show a training-free, calibration-free
pipeline (RHT + E8 lattice + honest metadata accounting) holds FP32-level accuracy
at **4.0 effective bits/scalar** (and if E5's E8@2b holds up: usable accuracy at
**3.0**), while matched-granularity scalar and lower-dimensional lattices collapse.
We introduce a geometric-reprojection temporal-consistency metric (TAE with z-buffer
and GT co-visibility masking) and show per-frame metrics alone miss temporal
degradation. Everything is measured under all-inclusive bit accounting.

### Contributions (3, crisp) — REVISED 2026-07-21 after F12/F13/F16

1. **First systematic study of extreme KV-cache quantization for video depth
   transformers.** Data-oblivious E8 lattice VQ, no calibration data, under
   all-inclusive bit accounting: **FP32-parity at 4.0 effective bits/scalar**
   (8.0× vs FP32, 4.0× vs FP16) on NYU-654 (−0.0014 δ1) and KITTI-1000
   (−0.0009 δ1), with a sharp cliff at 3.0 eff bits that we characterise.
   *(Locked by F12/F13. Always state "3-bit payload / 4.0 all-inclusive bits" —
   never "3-bit" alone.)*

2. **Matched-rate, matched-granularity dissection of what actually buys the
   compression.** Lattice dimension is the active ingredient (E8 ≫ D4 at
   identical 4.0 eff bits and identical scale machinery: 0.9128 vs 0.5338 NYU);
   rotation is shown to be *inert* at these rates on video ViTs — a negative
   result that runs against LLM practice (QuaRot/SpinQuant) and that we explain
   with activation statistics; QJL is shown to be strictly dominated
   (+2.25 eff bits for no gain). *(F14, F15; final framing per DG-2/DG-3.)*

3. **Temporal-consistency metrics for video geometry — and their failure mode.**
   We define a geometric-reprojection TAE (z-buffer + GT co-visibility masking)
   and then show it is **gameable**: our own collapsed 2-bit model beats FP32 by
   7.5× on TAE while failing catastrophically on accuracy, because degenerate,
   structureless predictions have little left to misalign. We quantify the
   degeneracy directly (S10) and argue temporal metrics are only interpretable
   at matched accuracy. *(F16 — this is the paper's most novel and most
   defensible piece; a metric contribution that documents its own failure mode
   is far stronger than one that does not. Give it a named subsection.)*

### Tables
| # | Content | Source |
|---|---|---|
| T1 | Main: NYU-654 + KITTI-1000; FP32/fp16/E8@{4,3,2}/scalar_g8@best/D4@3 — δ1↑ AbsRel↓ RMSE↓ + eff bits + CI | F-v2, S2 |
| T2 | Matched-rate 3-way at 4.0 and 5.0 eff bits (the F11-resolution table) | G1 + E1b + E2-era E8 |
| T3 | Sintel temporal: δ1/AbsRel + TAE median / covis-mean per bit-width | E5c rerun + E7 |
| T4 | Ablations at NYU: rotation×{3,2}b, QJL on/off, scale-bits sweep, seeds | E2-4, G2, G3 |
| T5 | K/V target ablation | G5 |
| T6 | vits vs vitl transfer | G4 |
| T7 | KV memory: GB @ W=32, 518px, both encoders, fp16 vs 4.0/3.0 eff | S7 |

### Figures — F1–F6 as specified in S8.

### Appendix
A. Eval protocol in full: disparity-space alignment (the F10 story — write it as a
   warning to the community; it silently deflates results and VDA's own constants
   only work in disparity space), gt_ranges, aspect-ratio handling, skip census.
B. Bit-accounting derivation per quantizer with a worked example each.
C. E8 decode: D8 ∪ (D8+½) best-of-two-cosets, tie handling, D4 boundary fixes (T4),
   correctness tests summary.
D. QJL estimator correction (cos(π/2(1−ρ))) and the measured bias-variance tradeoff
   (F3b): why m=512 works and m=128 hurts — and why headline runs QJL-off.
E. Per-scene Sintel tables; disocclusion analysis; covis_fraction per scene.
F. Seeds, variance, hardware, runtimes, all hyperparameters.
G. Limitations: simulated quantization (no INT kernels — analytic memory only),
   relative depth (metric deployment needs a metric head), single model family.
H. Practitioner pitfalls distilled from the F-series ledger (this appendix is
   cheap to write and reviewers love it).

### Reproducibility package
Anonymized repo at submission: code + configs + results JSONs + figure factory +
`run_phase3/4_experiments.sh` + fixed seeds. The 96+-test suite ships with it.

---

## 7. Statistical rigor bar (enforced, not aspirational)
- Every headline number: full split, 3 RHT seeds, mean ± std.
- Every "A beats B" claim: paired BCa bootstrap 95% CI on per-image diffs (S2),
  CI excluding 0, stated in the caption.
- Sintel aggregation: scene is the independence unit — report per-scene then
  aggregate across scenes, never pool frames as if independent.
- No number in the paper that isn't in a results JSON in the repo. The
  quarantine rule for docs/unverified_baselines.md stays absolute: those numbers
  never appear in any table or plot unless a row gains a real citation + protocol match.

---

## 8. Timeline (target: ICLR 2027, ~9 weeks)

| Week | Dates | Work |
|---|---|---|
| 1 | Jul 20–26 | Phase 3 finishes (watch E5 E8@2b!). Lit review (Sec 5) — DG-4. Sonnet: S1, S2, S3, S9. GPU: STAGE=gates. |
| 2 | Jul 27–Aug 2 | **Close DG-1/2/3.** Sonnet: S4–S7. GPU: STAGE=extend. Story + contribution list frozen in ledger. |
| 3–4 | Aug 3–16 | Finals-v2 full splits (locked configs, 3 seeds). S8 figure factory; all figures/tables generated from JSONs. |
| 5–6 | Aug 17–30 | Writing: Fable drafts (intro/method/related from lit notes), boss edits. Appendix A–H. Repro package. |
| 7 | Aug 31–Sep 6 | Red-team pass: Fable attacks the draft as Reviewer 2 (checklist = Section 1 attack table). Fix holes; rerun anything a hole demands (buffer exists for this). |
| 8–9 | Sep 7–deadline | Polish, abstract, submit. Buffer for disasters. |

Slip rule: if DG-1..3 are not all closed by end of week 2, cut G9 and vitl-KITTI
(keep vitl-NYU), never cut S2 stats or the lit review. CVPR 2027 (~Nov) absorbs a
full slip without cutting anything.

---

## 9. Standing risk register

| Risk | Trigger | Response |
|---|---|---|
| Lattice-KV prior art found | Sec 5 search | Reposition: video-geometry domain + TAE metric + extreme-rate frontier; framing meeting first |
| Rotation flag dead | G3a identical numbers | Fix, rerun E2 + G3b, re-enter DG-2 |
| Lattice margin vanishes vs scalar_g8 | G1 CIs overlap at 4.0 eff | Move comparison to 3.0 eff (G9); if still nothing → honest pivot per DG-3 |
| E8@2b collapses in E5 | E5 output | Fine — 4.0-eff story stands; drop the 3-bit frontier framing |
| +0.01-over-FP32 survives full split at 16-bit scales | G2b | Stop. Investigate as a bug with the same rigor as F10. No claims until explained |
| vitl breaks surgery assumptions | S4 mock test or G4 crash | S4's dimension-agnostic fix; worst case ship vits-only + limitation note |
| Colab/GPU box loss | any | Everything resumable by design; results JSONs are the only state that matters — zip and copy them after every session |
