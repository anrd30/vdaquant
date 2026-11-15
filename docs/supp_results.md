# Supplement experiments (server run 2026-11-15, commit 7b5389c)

Source: results/2026-11-16_supp (supp.tgz), summarised by scripts/supp_summary.py. All runs OK.
Deviation = relative deviation of the depth output from the FP32-cache model (%). Bits = 32 x cache / FP32 cache.

## 1. Frame stride (W2, Q1)
Stride 1 = the paper's runs (results/2026-11-13_vda_extra2). Bonn here = all 8 sequences.
| | s1 | s2 | s4 | s8 |
|---|---|---|---|---|
| Sintel S: INT2 g64 (2.50 b) | 1.62 | 1.64 | 1.59 | - |
| Sintel S: TCC-1 K8 (~2.0 b) | 1.40 | 1.80 | 2.00 | - |
| Sintel S: TCC-2 (~2.85 b) | 0.44 | 0.57 | 0.63 | - |
| Sintel L: INT2 / TCC-1 / TCC-2 | 1.62 / 1.68 / 0.32 | 1.73 / 2.28 / 0.41 | 1.74 / 2.57 / 0.43 | - |
| Bonn S: INT2 / TCC-1 / TCC-2 | 1.10 / 0.52 / 0.14 | 1.10 / 0.75 / 0.23 | 1.11 / 1.07 / 0.35 | 1.14 / 1.50 / 0.53 |
| Bonn L: INT2 / TCC-1 / TCC-2 | 1.40 / 0.50 / 0.09 | 1.43 / 0.79 / 0.15 | 1.46 / 1.25 / 0.25 | 1.55 / 1.77 / 0.33 |
- INT2 is flat in stride (codes each frame alone); TCC degrades as consecutive cached frames differ more.
- TCC-1 at ~2 bits falls behind INT2 at 2.50 bits from stride 2 on Sintel (fast motion), and at stride 8 on Bonn.
- TCC-2 stays 2.6-16x below INT2 at every stride (but at ~2.8-3.25 bits vs 2.50).
- delta1 stays within 0.003 of FP32 for TCC at every stride.
- win16 at stride >1 on Sintel is not meaningful: scenes become shorter than 16 frames, so it keeps everything at INT4
  (3.06 / 4.50 bits).

## 2. Abrupt cuts (W3, Q1): ScanNet, 10 streams of 85 frames of scene A then 85 of scene B
Mean deviation per frame around the cut (frame 85), ViT-S:
| | before | frames 86-88 | after (95-170) | bits |
|---|---|---|---|---|
| INT2 g64 | 1.15 | ~1.4 | 1.23 | 2.50 |
| TCC-1 K8 | 0.43 | 2.5-2.7 (peak per stream mean 2.81) | 0.54 | 2.04 |
| TCC-1 K8 + adaptive keyframe a0.5 | 0.42 | 0.23-0.44 | 0.46 | 2.21 |
| TCC-2 | 0.10 | 0.5-0.7 | 0.12 | 2.89 |
ViT-L: TCC-1 0.55 before, 1.4-1.8 for ~9 frames (back below 1.5x pre-cut in median 9 frames, i.e. by the next
keyframe), adaptive a0.5 removes the spike (peak 0.69) at 2.57 bits.
- A cut costs fixed-K TCC-1 a 4-6x deviation spike for up to one keyframe interval; still about INT2's level at the
  peak, and lower after.
- The adaptive keyframe (residual energy > 0.5 x entry variance) removes it for +0.17 bits (S) / +0.53 bits (L).

## 3. Fixed K vs adaptive keyframes at equal bits (Q2), ViT-S
Fixed K: Sintel d2 3.11b/0.74%, d4 2.39/1.15, d8 2.04/1.40, d16 2.07/1.54, d32 1.93/1.58;
Bonn d2 3.11/0.28, d4 2.46/0.43, d8 2.21/0.52; ScanNet d2 3.11/0.29, d4 2.39/0.41, d8 2.04/0.48, d32 1.75/0.52.
Adaptive (max interval 32): Sintel a0.75 3.64b/0.97%, a0.5 4.50/0.70; Bonn a0.25 2.57/0.36, a0.35 2.43/0.48;
ScanNet a0.35 3.14/0.48, a0.75 = d32.
- On continuous video, adaptive keyframes are not better than fixed K at the same bits (Sintel and ScanNet: worse;
  Bonn a0.25: slightly better than the K line). Fixed K is the right default; adaptive is the safeguard for cuts (2).
- Bits are not monotone in K beyond 8 (the residual chain kept for the oldest attended frame grows with K).

## 4. Error after each temporal module vs depth deviation (W4, Q4); modules in forward order
Sintel S: INT2 sym [8.7 6.0 5.6 5.5] -> 2.74; INT2 g64 [6.2 4.5 3.8 3.8] -> 1.62; TCC-1 [4.9 2.5 2.3 2.5] -> 1.40;
TCC-2 [3.6 1.6 1.1 1.3] -> 0.44. Bonn L: INT2 g64 [5.8 4.3 3.0 2.7] -> 1.40; TCC-1 [5.2 1.6 0.9 1.1] -> 0.50;
TCC-2 [2.8 0.7 0.3 0.4] -> 0.09.
- Module errors rank the formats as the depth deviation does, and the error shrinks through the head
  (depth deviation < last-module error) in every run: no amplification.
- Exception to the ranking: Sintel L, TCC-1 has lower module errors than INT2 g64 [6.7 2.7 1.9 2.3 vs 6.8 3.6 2.7 3.2]
  but a slightly higher depth deviation (1.68 vs 1.62).

## 5. Redundancy per layer, dataset, stride (W2; W1 cross-layer)
Energy the previous frame leaves, % of the entry variance (8 layers, pooled over scenes):
Sintel S s1 16.5-46.2 (mean 27.2); Bonn S s1 6.5-27.2 (13.1); ScanNet S s1 4.7-30.1 (12.2); ViT-L similar.
-> the paper's "4-46%" range (intro). Grows with stride: s2 means 22-49, s4 38-74, s8 61-97 (some layers > 100%).
- Best per-channel affine temporal predictor fitted on the same scene (in-sample): mean 25.5 vs 27.2 (Sintel S s1),
  12.3 vs 13.1 (Bonn), 11.6 vs 12.2 (ScanNet): a fitted temporal predictor gains little at the native frame rate.
- Cross-layer oracle (other layer of the same module, same frame, per-channel affine, in-sample; applies to 4 of 8
  layers): first module 20-32%, other modules 41-68%. At stride 1 temporal beats it on 3 of 4 modules
  (e.g. ScanNet S temporal [27.0 6.9 4.7 9.0] vs cross-layer [25.1 54.9 45.0 48.5]); the first module is a near-tie
  (Sintel: cross-layer 25.5 vs temporal 48.0). At stride >= 4 cross-layer wins on most modules.

## 6. Latency vs K (Q5), A100, median of 3 repeats (ms/frame)
ViT-S: FP32 36.8, INT2 g64 40.7, TCC-1 K=2/4/8/16/32: 42.1/41.8/41.7/41.7/41.9.
ViT-L: FP32 94.4, INT2 g64 96.8, TCC-1 K=2/4/8/16/32: 97.7/98.2/97.3/97.0/97.4.
- Latency does not depend on K (within 1 ms); TCC costs ~1 ms more than plain INT2.
- Cache MB (ViT-S): K2 35.0, K4 26.9, K8 22.9, K16 23.3, K32 21.7.
