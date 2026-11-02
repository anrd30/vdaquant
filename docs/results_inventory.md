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
(`hard_setting_lattice_vs_int/*.json`). FP32 δ1 0.6876, cov-TAE 8.56. Real packed bits.

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
E8 matches fair INT with ~0.5 fewer bits; Λ16 is worse than E8. Explains the old
"INT collapses" result: one-scale-per-tensor INT + backbone outliers.

## Pending
- Sintel ViT-B / ViT-L streaming accuracy (running).
- StreamVGGT official video-depth eval + frames-until-OOM (rented GPU).
- Paper-grade timing and VDA peak memory with bit-packing.
