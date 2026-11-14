# cov-TAE v2 results (2026-11-14)

Source: `results/2026-11-14_covtae/analysis_{sintel,bonn}_{vits,vitl}.json` (scripts/covtae_analysis.py on
predictions saved by scripts/covtae_queue.sh; server run 10:16-11:49). Offline TAE reproduces the live
run per scene within 0.054% (relative). Scene value = mean over pairs and both directions (eval_tae.py);
tables below = mean over scenes unless "median". Sintel: 23 scenes. Bonn: balloon, balloon2, crowd, crowd2
(first 2000 frames of the sorted sequences; the other 4 sequences not yet run).

## Ground truth as the prediction (should be ~0)
| | TAE | TAE (no clipped px) | cov-TAE (tau .05) |
|---|---|---|---|
| Sintel GT | 51.81 (median 5.13) | 33.42 | 0.62 |
| Bonn GT | 16.92 | 5.33 | 0.90 |
| Sintel VDA-S FP32 | 54.30 | 34.73 | 5.86 |
| Bonn VDA-S FP32 | 3.98 | 3.98 | 2.00 |
Bonn: standard TAE ranks the model above perfect GT, also without hole pixels; cov-TAE ranks GT first.

## Per-frame model (window 1) vs video model (window 32): per-frame must be less consistent
Standard TAE ranks it WRONG on Sintel for ViT-S (53.82 vs 54.30) and ViT-L (51.66 vs 54.58), also with the
GT-valid mask; cov-TAE ranks it right in all 4 settings at every tau (0.01-0.2) and with GT denominator
(Sintel S 9.69 vs 5.86, L 7.98 vs 6.51; Bonn S 3.16 vs 2.00, L 2.79 vs 2.00).

## Positive control: per-frame scale jitter (sigma 1/2/5/10%), per-frame delta1 stays 1.000
cov-TAE GT Sintel 0.62 -> 1.55 -> 2.51 -> 5.39 -> 9.67; Bonn 0.90 -> 1.47 -> 2.41 -> 5.48 -> 10.24.
Standard TAE (mean) Sintel GT 51.81 -> 52.48 -> 53.00 -> 53.14 -> 32.95 (not monotone).

## Degradations
Blur on FP32 (sigma 0->8): TAE Sintel-S 54.30 -> 41.59, Bonn-S 3.98 -> 3.39 (lower = "better");
cov-TAE Sintel-S 5.86 -> 8.87, Bonn-S 2.00 -> 2.04 (flat).
E8 2-bit collapse: Sintel-S TAE 54.30 -> 10.53, cov 5.86 -> 8.70 (delta1 frame-aligned 0.716 -> 0.513);
Bonn-S TAE 3.98 -> 3.35, cov 2.00 -> 2.96. Bonn-L 2-bit: cov 2.00 -> 1.99 (mild, delta1 0.980 -> 0.977).

## Mechanism
Error on excluded (non-co-visible) pixels: Sintel ~190% (model), 229% (GT); Bonn 9.5% (model), 69% (GT).
Share of TAE error in the top 1% pixels: Sintel GT 0.50, model 0.30. Co-visible fraction: Sintel 0.86, Bonn 0.77.

## Must be stated
Constant depth per frame fools BOTH metrics (Sintel TAE=cov 3.5-4.3, Bonn 0.6-0.7, delta1 ~0.35-0.47):
temporal scores must be reported with accuracy. The old claim "TAE is not fooled by constant depth" is false.
VDA reports TAE only on ScanNet (eval_tae.sh: 20 scenes, frames 10-180, no mask).

## Flow-warping error (Lai et al. 2018 formula, Sintel GT flow + occlusions), 2026-11-14 evening
Source: results/2026-11-14_covtae_flow/analysis_sintel_{vits,vitl}.json (re-analysis of the same saved predictions;
TAE identical to the morning run on every scene and variant). F = aligned disparity / scene max GT disparity.
Exact VDA eval_tae.py TAE (tae_upstream): FP32 ViT-S 55.36 (old paper's upstream number), GT 53.69.
| Sintel ViT-S (mean over scenes) | TAE | warp no mask | warp official occl. mask | cov-TAE |
|---|---|---|---|---|
| GT | 51.81 | 0.00885 | 0.00187 | 0.62 |
| FP32 | 54.30 | 0.00738 | 0.00198 | 5.86 |
| per-frame (w1) | 53.82 | 0.00704 | 0.00219 | 9.69 |
| blur 8 | 41.59 | 0.00684 | 0.00202 | 8.87 |
| 2-bit | 10.53 | 0.00495 | 0.00213 | 8.70 |
| const | 4.31 | 0.00395 | 0.00215 | 4.31 |
Unmasked warping: GT worse than model; rewards blur/collapse/constant; per-frame ranked better (scenes: S 11/23, L 7/23 correct).
Official mask: per-frame worse in S 20/23, L 16/23 scenes; but GT ~ model (0.00187 vs 0.00198), weak flicker response
(GT jitter 1/5/10%: 0.00190/0.00263/0.00439 vs cov-TAE 1.55/5.39/9.67). Likely cause (unverified): depth values change
with camera motion, so flow-warped depth differs even for perfect GT; 3D reprojection accounts for this.
