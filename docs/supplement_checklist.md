# CVPR 2027 supplement checklist (due Nov 23 AoE)

Everything the main paper (main_12nov_v9.tex) promises or relies on, and where the data is.

## Explicitly promised ("in the supplement")
- [ ] Table 1 caption: delta1 and AbsRel for every backbone x dataset x format pair.
      Source: results/fidelity_{sintel,bonn}.json -> tables/tab_full.tex (TAE and cov-TAE columns removed 2026-11-12).

## Claims in the main text that a reviewer will want backed
- [ ] Bootstrap table: ALL comparisons with ratio, 95% CI, wins, per-scene SD -> results/paper_bootstrap.json (scripts/paper_bootstrap.py, 20000 resamples, seed 0, scene means). Borderline (lower bound 1.00): TCC-1 vs XStreamVGGT-12000 Sintel, TCC-2 vs INT3 Sintel; TCC-1 K8 vs INT2 not significant on Sintel S/L - paper does not claim these.
- [ ] Per-scene wins: "18-19 of 23 Sintel scenes", "21 of 23 / 5 of 5", "every scene" for TCC-2 - per-scene tables.
- [ ] XStreamVGGT: authors' released code vs our implementation, "deviates more in three of four settings" - table.
- [ ] Long videos: per-scene Table 6 values, 495-frame curves (crowd2, crowd3), Sintel camera-centre numbers (13.3% vs 38.5-43.1%).
- [ ] Camera "extent" definition and per-scene trajectory extents (reviewer asked).
- [ ] Frames-until-OOM table for all formats (282 / 960 / 924 / 1144 / 2287 / 1847 / 1460), both sequences.
- [ ] Speed: StreamVGGT clean interleaved timing (svg_time2) + full GPU profile breakdown (svg_prof3); VDA speed table hardware (rerun on A100 if possible).
- [ ] Residual statistics behind the bound: contraction 35.5% mean / 65.6% max, key outlier ratios.
- [ ] Open-loop control on all VDA backbones (12 ratios, results/2026-11-02_streaming/streaming_vda/*_open.json).
- [ ] Lower frame rates (every 2nd/4th frame) and scene-cut details (moved out of main text in v14: TCC-1 0.58% before / 0.47% after the cut; adaptive keyframes -0.07..0.10 points at 1.93 vs 1.88 bits).
- [ ] Frame dropping (StreamVGGT, and VDA once outputs/stream/*_drop.json finish).
- [ ] 3D reconstruction: main paper has the dense-setting paragraph (v12). Supplement: full table incl. sparse (official stride) setting, official Acc/Comp/NC, per-scene deviations, adaptive bits. Sources: svg_recon logs, recon_fidelity2.json.
- [ ] Implementation details: bit packing (INT3 two codes per byte), scale index grid, fused kernel description, resolutions (VDA 518, Sintel 392; StreamVGGT runner vs official preprocessing).
- [ ] Hyperparameter choice: set on VDA, reused unchanged on StreamVGGT.

## Open decisions
- [x] TAE: dropped everywhere (main.tex + tab_full.tex), 2026-11-12. Do not reintroduce without verifying the TAE code.
- [ ] "Where consistency lives" analysis moved out of the paper - include or not.
- [ ] XStreamVGGT+TCC at 2048 tokens does NOT shrink the cache (49 vs 54 MB Sintel, 57 vs 54 Bonn) - report honestly; queue5 tests 8192.
- [ ] Moved out of main text in v15: Bonn AbsRel within 0.0007 (2-bit/KIVI 0.0618-0.0645); Bonn flicker 1.9-2.4x lower; Sintel camera centres 13.3% vs 38.5-43.1%; K=2 flicker 1.21->0.72; 3D raw means (TCC-2 2.75/1.01 vs KIVI 3.90/3.71); frame-dropping memory details; first contribution bullet (memory measurement).
- [ ] KITTI table (dev, flicker, AbsRel, delta1, peak) all 6 caches + STAC paper row; bootstrap kitti rows in results/paper_bootstrap.json.
- [ ] STAC table: full/default/recommended x Bonn/Sintel(14) dev vs official and vs STAC-full, peak MB, seconds; note own-implementation offset.
- [ ] XStreamVGGT budget sweep table (fid_xs_*.json) incl. non-monotone Bonn behaviour; XStream+TCC.
- [ ] Contraction table TCC-1/TCC-2 keys/values mean/max (svg_queue5/contraction).
- [ ] 755-frame crowd2 per-100 curves for all 4 caches (TCC-2 OOM at 697 on shared GPU).
- [ ] Moved out in v18: lower-frame-rate paragraph (TCC-2 0.42-0.54% at every 2nd/4th frame vs 1.43-1.72% INT2/KIVI; scene cut 0.58/0.47%, adaptive -0.07..0.10 at 1.93 bits); full GPU-profile breakdown; TCC-2 peak memory in Table 2 (inverted vs TCC-1 in the accuracy runs - explain or omit); "code will be released" sentence (give anonymous code link in supplement).
- [ ] Bonn: per-scene tables instead of bootstrap intervals (paper now promises this).
- [ ] XStreamVGGT authors' code vs ours (paper says ours is the stronger baseline in 3 of 4 settings).
- [ ] STAC "full" mode is OFFLINE (no causal mask: mode name lacks "causal", causalvggt/layers/block.py create_attn_mask -> mask None => every frame attends to future frames). Its deviation from official StreamVGGT (Bonn 3.7, Sintel 12.0, KITTI 17.6%) is offline-vs-streaming, NOT an anomaly. Paper v20 no longer compares STAC against it. If a STAC-own streaming reference is wanted: run STAC --mode causal (check it streams without compression).
- [ ] STAC reproduction check (STAC's own eval_depth.py, --align scale), proves our STAC runs match its paper:
      Bonn 0.0595/97.11 (paper 0.060/97.1); KITTI 0.1919/67.41 (paper 0.192/67.6);
      Sintel on STAC's 14 scenes: STAC 0.3415/59.71 vs uncompressed on same 14 0.3404/59.58 (paper, 23 scenes: 0.327/65.2 vs 0.323/65.7).
      Note: STAC GT metrics ~unchanged on Sintel while deviation is 8.92% (TCC-1 4.15%) -> illustrates GT-metric blindness.

## Added with v23 (per-sequence ground truth in Table 1)
- Flicker deviation table for VDA (moved out of Table 1; text still cites 2.2-4.6x lower flicker for TCC-2).
- Per-frame-aligned delta1/AbsRel for every VDA format (old Table 1 numbers, tab_full.tex).
- Five Bonn sequences (bonn5, full length) table: delta1_seq/AbsRel_seq/Dev for all formats incl. win16/win8 (text: TCC-1 within 0.002, window loses up to 0.013).
- Short-window baselines win16 and win8 on all sets (win8: cache 44-47% of TCC-1, delta1 up to -0.027).
- Zero cache under per-sequence alignment: on Bonn it RAISES delta1_seq (0.966 -> 0.984, ViT-S) while deviation is 24-38%; on Sintel it lowers it by 0.014-0.029. Explain neutrally (streaming FP32 drifts in scale over long Bonn videos; one fit per video penalises drift). Kept out of main Table 1 on purpose.
- Note the scale-only per-sequence variant (delta1_seq_scale ~0.45) is not meaningful in disparity; do not report.
- Table 1 sources: delta1/AbsRel from results/2026-11-13_vda_extra2 (A100 rerun, Bonn restricted to balloon/crowd/kidnapping_box), Dev from the main runs; rerun reproduces per-frame delta1 to 1.1e-4 (audit_paper.py).
- VDA KITTI (kittiseq, 13 drives): DONE 11-13, in text (TCC-1 dev 0.54-0.64 vs INT2 1.12-1.59; window up to -0.007 delta1). Add full table to supplement.
