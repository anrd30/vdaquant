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

## Model comparison: StreamVGGT vs VDA (2026-11-14 night reruns, all OK)
Source: results/2026-11-14_covtae_bonnL/analysis_models_{sintel,bonn}_{vits,vitl}.json (server runs 21:58-22:31).
StreamVGGT (bf16, no cache coding): Sintel all frames; Bonn first 110 frames per sequence (all models cut to 110).
| Sintel (mean of 23 scenes) | TAE | VDA eval_tae.py | cov-TAE | warp (occl. mask) |
|---|---|---|---|---|
| GT | 51.81 | 53.68 | 0.62 | 0.00187 |
| StreamVGGT | 52.11 | 52.53 | 7.48 | 0.00220 |
| VDA-S video | 54.30 | 55.36 | 5.86 | 0.00198 |
| VDA-L video | 54.59 | 55.72 | 6.51 | 0.00197 |
Ranking flip on Sintel: standard TAE puts StreamVGGT ahead of both VDA models (on the mean); cov-TAE and masked
warping put both VDA models ahead. Per scene, StreamVGGT lower than VDA-S/L: TAE 12/23, 12/23 (coin flip);
cov-TAE 10/23, 7/23; masked warp 5/23, 5/23. Not a clean flip per scene: state it as "TAE cannot separate
them, cov-TAE and warping both prefer VDA". delta1 (scene-aligned): StreamVGGT 0.644, VDA-S 0.580, VDA-L 0.607.
ViT-L Sintel: per-frame w1 (51.66) and 2-bit (52.17) both beat FP32 video (54.59) on mean TAE; cov-TAE ranks
FP32 first (6.51 < 7.98, 8.08).
| Bonn, 110 frames x 8 seq | TAE | cov-TAE | delta1 |
|---|---|---|---|
| GT | 12.65 | 0.80 | 1.000 |
| StreamVGGT | 2.32 | 1.24 | 0.987 |
| VDA-S video | 2.52 | 1.49 | 0.986 |
| VDA-L video | 2.57 | 1.49 | 0.986 |
Bonn: no flip; StreamVGGT lower than VDA-S under both (7/8 scenes), VDA-L TAE 7/8, cov 6/8.

## Bonn ViT-L, all frames (rerun of the failed 19:18 analysis)
GT 13.79 / cov 0.84; FP32 3.35 / 1.85; per-frame 3.74 / 2.43 (worse on both, TAE 7/8, cov 8/8);
2-bit 3.10 / 1.85 (delta1 0.896 vs 0.924): TAE prefers 2-bit in 8/8 scenes, cov-TAE prefers FP32 in 5/8 (means tied).
Same at 110 frames (TAE 0/8 for FP32, cov 5/8). Honest reading: mild quantisation smoothing lowers TAE;
cov-TAE does not reward it but barely separates the two.

## Kinect-hole check (all 8 Bonn sequences, S and L identical for GT)
GT: TAE 13.79, TAE excluding clipped/hole pixels (tae_noclip) 4.05, target-valid only 13.77, cov-TAE 0.84.
FP32 ViT-S 3.22 / 3.22 / 3.00 / 1.84. GT still worse than the model with holes removed in 7/8 sequences.
Remaining GT error sits on non-co-visible pixels (GT 60.3% vs model 7.5%; co-visible fraction 0.78).
So holes explain most of GT's standard TAE but not the ranking; occlusion handling does.

## ScanNet (VDA's own TAE benchmark), 2026-11-15
Source: results/2026-11-15_covtae_scannet/analysis_scannet_{vits,vitl}.json. 20 test scenes (scene0707_00..0726_00),
frames 10..179, depth/1000, range 0.1-10 m, full frame with depth intrinsics (VDA's script crops 8/11 px but keeps
uncropped K, so absolute values are not comparable to VDA's published ScanNet TAE).
| ScanNet | TAE | VDA eval_tae.py | tae_noclip | cov-TAE | delta1 |
|---|---|---|---|---|---|
| GT | 11.01 | 14.26 | 2.02 | 0.368 | 1.000 |
| VDA-S w32 FP32 | 1.016 | 1.018 | 1.016 | 0.753 | 0.848 |
| VDA-L w32 FP32 | 0.922 | 0.925 | 0.922 | 0.643 | 0.862 |
| VDA-S per-frame (w1) | 1.887 | | | 1.691 | 0.843 |
| VDA-S blur 8 | 0.898 | | | 0.746 | 0.845 |
| VDA-S 2-bit | 2.402 | | | 2.303 | 0.696 |
| VDA-S jitter 5% | 3.394 | | | 3.185 | 0.842 |
| constant depth (S) | 0.746 | | | 0.736 | 0.556 |
Per scene (S / L): GT worse than model under TAE 20/20 and 20/20; better under cov-TAE 19/20 and 19/20;
holes excluded (tae_noclip) GT still worse 19/20 and 20/20. Per-frame worse under both 20/20.
2-bit worse than FP32: S TAE 19/20, cov 19/20; L TAE 15/20, cov 20/20. Bit sweep monotone under both.
Blur lowers TAE (1.016 -> 0.898), cov-TAE flat (0.753 -> 0.746). Constant depth fools both (report with delta1).
