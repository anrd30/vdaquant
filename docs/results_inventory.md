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
