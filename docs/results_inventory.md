# Results inventory (Paper 2, Sep 2026 runs)

Built 2026-10-31 from (a) saved zips in the repo root (dated 2026-09-10 to 09-20)
and (b) run logs pasted into Claude chat sessions. ViT-S, seed 0, 8-bit scales,
QJL off, backbone + temporal quantized, unless stated. "eff" = b + 8/g.

FP32 reference (ViT-S): NYU 0.9099, KITTI 0.928 (val_selection_cropped, N=1000), Sintel 0.7106.

## In the paper already
Λ16 tables (rate-distortion, rotation, group size, window, seeds, Bonn), per-frame trace,
kernel microbenchmark, end-to-end FPS.

## Run, but NOT in the paper

| Experiment | Result | Source |
|---|---|---|
| Surgery identity control (F27), N=200 | ViT-S 0.9036 = FP32 0.9036; ViT-L 0.8869 = FP32 0.8869 (exact) | chat paste 2026-09-09 |
| E8 eff 4.0, seeds 0-4, NYU | 0.9069, 0.9221, 0.9189, 0.9184, 0.9227 -> 0.918 +- 0.006 | zip (s0,s3,s4) + paste (s1,s2) |
| E8 eff 4.0, seeds 0-4, KITTI | 0.9253, 0.9211, 0.9209, 0.9069, 0.9141 -> 0.918 +- 0.007 | zip + paste |
| E8 eff 3.0 (b=2) | NYU 0.6946, KITTI 0.7559, Sintel 0.5158 (source of intro 0.711->0.516) | paste 2026-09-09 |
| E8 eff 6/7/8 | NYU 0.9085/0.9096/0.9096, KITTI 0.9282/0.9281/0.9282 | zip paper2_verify |
| D4 eff 6.0/5.0/4.0 | NYU 0.9121/0.9093/0.7161, KITTI 0.9261/0.9172/0.7777 | paste 2026-09-09 |
| E8 ViT-B, N=200 | NYU FP32 0.8653, eff5 0.8711, eff4 0.885; KITTI 0.9298, 0.9308, 0.9199 | paste |
| E8 ViT-L, N=200 | NYU FP32 0.8869*, eff5 0.8906, eff4 0.9086; KITTI 0.9426, 0.9419, 0.9384 | paste |
| E8 ViT-L Sintel, 10 scenes | FP32 0.7434; eff5 0.741; eff4 0.7284; eff3 0.6671 (raw TAE mean drops 101.8 -> 95.2 at eff3) | paste |
| Asymmetric K/V, E8, eff 4.0 | K4/V2: NYU 0.9016, KITTI 0.9274. K2/V4: NYU 0.7715, KITTI 0.8163 -> keys need the bits | zip asym_kv (flags `--bits K --v-bits V`) |
| QJL on, E8 eff 4.0 | NYU 0.9063, KITTI 0.9258 (no gain) | zip paper2_verify |

*ViT-L NYU FP32 on N=200 from the identity run.

Note: seed-0 E8 eff 4.0 was run twice (NYU 0.9063 / 0.9069, KITTI 0.9258 / 0.9253): run-to-run jitter ~0.0006.

## Attempted, FAILED (still missing)

| Experiment | Failure | Fix |
|---|---|---|
| E8 at g=16/32/64 (`p1_e8_*`) | AssertionError: feature dim 8 (and 32) not divisible by group size | give E8/scalar the same token-absorb rule Λ16 uses |
| Scalar + Hadamard at g=16/32/64 (`p1_sg_*`) | same assertion | same fix |
| Golay24 lattice (`g24_*`) | CUDA OOM (22 GiB alloc in nearest-point search) | chunk the search |
| Sintel K/V asymmetry, Sintel Golay, bw16_sintel_b3 rerun | Sintel depth tree missing in that run dir | point at the correct data dir |

---

# Streaming temporal-cache results (2026-11-02, laptop RTX 4050)

Raw files: `results/2026-11-02_streaming/` (tracked in git). Accuracy is
GPU-independent; timings from this laptop are NOT paper-grade (re-measure on a
rented GPU). Setting: VDA streaming mode (`infer_video_depth_one`), only the
temporal-attention cache is compressed, backbone untouched, no rotation unless
`-rht`. Bits = measured packed size per cached scalar incl. scales (and the
delta anchor chain). Primary consistency metric: cov-TAE (tau = 0.05); δ1 and
raw TAE are nearly blind to the cache (see control).

## Bonn RGB-D, ViT-S, 3 scenes, 2456 frames (`streaming_vda/bonn_vits_*.json`)
FP32: δ1 0.9878, AbsRel 0.0442, TAE 2.831, cov-TAE 1.711, cache 355.9 MB.

| Config | bits | cov-TAE vs FP32 | AbsRel |
|---|---|---|---|
| fp16 / int8-g64 | 16 / 8.25 | 0.0% | 0.0442 |
| int4 (g16, g64-asym, g16-rht) | 4.5 | -0.1% | 0.0442 |
| int3-g16 / int3-g16-rht | 3.5 | -0.5% | 0.0442-0.0443 |
| int3-gT (one scale per tensor) | 3.0 | -0.2% | 0.0444 |
| int2-g16 (no rotation) | 2.5 | +2.9% | 0.0449 |
| int2-g16-rht | 2.5 | +0.8% | 0.0445 |
| bin-g16 (1-bit, no delta) | 1.5 | +35.9% | 0.0447 |
| **delta: int3 key + bin residual, K=8** | **2.05** | **+0.6%** | **0.0444** |
| delta: int3 key + bin, K=16 | 2.20 | +0.8% | 0.0444 |
| delta: int4 key + bin, K=8 | 2.21 | +0.6% | 0.0444 |
| Λ16 b3 / b2 | 3.0 / 2.0 | +0.2% / +13.0% | 0.0443 / 0.0448 |
| zero (cache wiped, control) | 0 | +35.5% | 0.0460 |

## Sintel, ViT-S, 23 scenes, 1045 frames (`streaming_vda/sintel_vits*.json`)
FP32: δ1 0.7194, AbsRel 0.2268, TAE 53.03, cov-TAE 6.848, cache 360.0 MB.

| Config | bits | cov-TAE vs FP32 | raw TAE vs FP32 | AbsRel | δ1 |
|---|---|---|---|---|---|
| int3-gT | 3.0 | -0.5% | -0.3% | 0.2277 | 0.7184 |
| int3-g16 | 3.5 | -0.4% | -0.3% | 0.2267 | 0.7196 |
| int2-g16 | 2.5 | -0.3% | +0.3% | 0.2314 | 0.7124 |
| int2-g16-rht | 2.5 | +4.6% | +0.7% | 0.2224 | 0.7244 |
| bin-g16 | 1.5 | +35.6% | - | 0.2223 | 0.7197 |
| **delta: int3 key + bin, K=8** | **1.87** | **+0.1%** | - | **0.2267** | **0.7188** |
| delta: int3 + int2, K=8 | 2.73 | 0.0% | - | 0.2271 | 0.7187 |
| Λ16 b3 / b2 | 3.0 / 2.0 | +0.4% / +14.5% | 0.0% / 0.0% | 0.2272 / 0.2242 | 0.7189 / 0.7157 |
| zero (control) | 0 | **+60.1%** | **+2.9%** | 0.2200 | 0.7163 |

Delta config chosen on Bonn, applied unchanged to Sintel.

## Memory, synthetic 50-frame video, 518x518 (`streaming_vda/mem_*.json`, `vits_518*.json`)
Cache MB is the packed size; peak MB in the VDA path still stores codes as
1 byte each (bit-packing not yet ported there), so peak savings are understated.

| Encoder | FP32 cache | delta cache | FP32 peak | delta peak | depth relL1 int2 / delta |
|---|---|---|---|---|---|
| ViT-S | 269.0 MB | (bw16:3 25.2 MB) | 770 MB | (bw16:3 337 MB) | - |
| ViT-B | 538.0 MB | 31.4 MB (17.1x) | 1786 MB | 1115 MB | 0.208 / 0.0048 |
| ViT-L | 1129.2 MB | 66.0 MB (17.1x) | 4150 MB | 2857 MB | 0.033 / 0.0071 |

## Other measurements
- Frame-to-frame residual of the cache (Bonn, 150 frames/scene,
  `scripts/stream_delta_probe.py`): residual / per-channel-centred variance
  0.013-0.255 per layer; weighted 0.111-0.133 -> ~1-2.4 bits/scalar saving.
- Projected-KV caching (W_k h + W_k pe): exact (relL1 1e-4) but no speedup and
  2x cache -> dropped (`vits_518_kv.json`).

## Hard setting: backbone + temporal quantized, offline T=32, Sintel first 8 scenes, RHT on
(`hard_setting_lattice_vs_int/*.json`). FP32 δ1 0.6876, cov-TAE 8.56.
CAUTION (found by the bootstrap/facts audit): the JSON `effective_bits_per_scalar`
is the table rate b + 8/g for every row. The "packed bits" below are ANALYTIC for
the lattice rows: E8 codeword = 4 + 8(b-1) bits per 8 scalars (Construction-A
index), Λ16 = 5 + 16(b-1) bits per 16 (implemented for Λ16 in packed_bw16.py, NOT
implemented for E8). INT rows are b + 8/g (achievable). So "E8 needs ~0.5 fewer
bits" assumes an E8 index packing that does not exist in the code yet.

| Config | packed bits | δ1 | cov-TAE |
|---|---|---|---|
| int3 g16 | 3.5 | 0.6869 | 11.08 |
| int3 g64 | 3.125 | 0.6938 | 14.03 |
| E8 g16 b3 | 3.0 | 0.6836 | 12.06 |
| E8 g32 b3 | 2.75 | 0.6775 | 12.44 |
| E8 g64 b3 | 2.625 | 0.6836 | 14.97 |
| Λ16 g64 b3 | 2.625 | 0.6225 | 52.69 |
| int2 g16 | 2.5 | 0.4187 (collapse) | 6.38* |
| int2 g64 | 2.125 | 0.4069 (collapse) | 5.96* |
| E8 g64 b2 | 1.625 | 0.4252 (collapse) | 7.61* |
| int3/int4 one scale per tensor | 3.125 / 4.125 | 0.4509 / 0.6314 | 63.88 / 14.49 |

*cov-TAE drops below FP32 when the output collapses to a flat map: report with δ1.
E8 matches fair INT with ~0.5 fewer (analytic) bits; Λ16 is worse than E8. Explains the old
"INT collapses" result: one-scale-per-tensor INT + backbone outliers.

## Statistics (scripts/bootstrap_ci.py -> results/2026-11-02_streaming/analysis/)
Paired bootstrap over scenes (10k resamples, seed 0), scene-mean weighting.
- Sintel ViT-S: delta int3+bin K8 cov-TAE +0.1% [-3.7, +5.2] (not distinguishable
  from FP32); int2-g16 -0.3% [-4.2, +5.2] (also not distinguishable on cov-TAE, but
  δ1 -0.006 [-0.015, +0.000], AbsRel +0.0046).
- Bonn ViT-S (3 scenes only -> coarse CIs): delta +0.6% [+0.4, +0.8] vs int2 +2.9% [+1.2, +4.5].
- Sintel ViT-B: delta +3.3% [+2.0, +5.4] vs int2 +7.2% [+4.6, +12.6]; zero +43.9%.
- Weighting matters (frame-weighted Sintel zero +73.0% vs scene-mean +60.1%): state it.
- Sintel: 1064 frames for TAE, 1045 for δ1/AbsRel (19 frames without in-range GT).


## Overnight 2026-11-03 (laptop; `streaming_vda/*_fidelity|*_fading|*_ablation|age_attention_*|bonn_vitb_key.json`)
Fidelity-to-FP32 (GT-free): fid = mean |d - d_fp32| / mean |d_fp32| (%); flk = same on frame-to-frame changes.
Bits = allocated (INT3/INT4 keyframes stored as 4-bit nibbles since the bit-packing port).

| Setting | bits | cov-TAE vs FP32 | fid | flk | δ1 chg | AbsRel chg |
|---|---|---|---|---|---|---|
| Sintel ViT-S zero | 0 | +60.1% | 31.4% | 8.68% | -0.0031 | -0.0068 |
| Sintel ViT-S int2-g16 | 2.50 | -0.3% | 2.75% | 1.24% | -0.0070 | +0.0046 |
| Sintel ViT-S delta int3 key + bin, K8 | 2.04 | +0.1% | 1.46% | 1.24% | -0.0006 | -0.0001 |
| Sintel ViT-S delta int4 key + bin, K8 | 2.04 | +1.0% | 1.40% | 1.21% | -0.0008 | -0.0001 |
| Bonn ViT-S zero | 0 | +35.5% | 24.1% | 2.43% | -0.0008 | +0.0019 |
| Bonn ViT-S int2-g16 | 2.50 | +2.9% | 2.18% | 0.56% | -0.0002 | +0.0007 |
| Bonn ViT-S delta int3/int4 key + bin, K8 | 2.21 | +0.6% | 0.46-0.48% | 0.38% | -0.0001 | +0.0002 |
| Bonn ViT-B zero | 0 | +13.6% | 38.0% | 1.74% | +0.0003 | -0.0003 |
| Bonn ViT-B int2-g16 | 2.50 | +8.7% | 6.38% | 0.62% | -0.0001 | +0.0007 |
| Bonn ViT-B delta int4 key + bin, K8 | 2.21 | +0.6% | 0.39% | 0.30% | -0.0001 | +0.0001 |
| Bonn ViT-L zero | 0 | +17.7% | 29.6% | 1.92% | +0.0006 | -0.0006 |
| Bonn ViT-L int2-g16 | 2.50 | +4.9% | 2.98% | 0.55% | -0.0001 | +0.0005 |
| Bonn ViT-L delta int4 key + bin, K8 | 2.21 | +0.2% | 0.50% | 0.31% | +0.0000 | +0.0000 |

Bonn ViT-L memory (real, bit-packed): FP32 cache 1494 MB, peak 4986 MB; delta cache 103.4 MB (14.5x), peak 2843 MB (-43%).

Sintel ViT-S delta ablations (int4 key + bin residual unless stated):
K=4 2.39 b fid 1.14%; K=8 2.04 b 1.40%; K=16 2.07 b 1.54% (anchor overhead, no saving);
K=32 1.93 b 1.58%; int8 key 2.70 b 1.39%; int2 residual 2.89 b 0.61%;
predictor none: cov-TAE +19.4%, fid 6.97% (prediction is the source of the gain);
predictor extrap2: cov -2.8%, fid 2.83% (worse fidelity than 'prev' 1.40%).
Bonn ablation failed with a transient CUDA "unknown error" (rerun pending).

Age analysis (VDA window, 31 cached slots; ages: recent 1-9, mid 10-19, old 20-29):
- Attention mass per slot vs uniform: recent 1.35-1.99x, old 0.24-0.52x, all 8 layers, both datasets.
- Causal: zeroing only recent frames cov-TAE +28.5% (Sintel) / +25.1% (Bonn); only old +3.7% / +0.7%.
  1-bit only recent +20.6% / +22.7%; only old +1.3% / +0.5%.
- Equal memory 2.79 b: fading (recent 4b, mid 2b, old 1b) cov +1.0% / +0.8%, fid 3.2% / 3.4%;
  reversed +23.1% / +24.0%; shuffled +3.1-8.7% / +2.6-9.1%. Uniform int2 (2.5 b) fid 2.7% / 2.2%.
  -> age matters strongly, but fading alone is dominated by delta coding (~2.0-2.2 b, fid 0.5-1.5%).

Codec probes (skip blocks, motion compensation): skippable groups at tau=0.1 0-41% per layer
(0% in the 192-ch layers); best-shift residual 73-100% of zero-shift -> small gains (~0.2-0.3 b).
Regression: refactored delta code reproduces Bonn balloon exactly (diff 0).

## Pending
- Sintel ViT-L streaming accuracy (running).
- StreamVGGT official video-depth eval + frames-until-OOM (rented GPU).
- Paper-grade timing and VDA peak memory with bit-packing.
