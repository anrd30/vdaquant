# Supplement plan (due Nov 23 AoE). Build with scripts/make_supp.py; every number from results/.
Current build: paper/supp_v2.pdf (sections marked DONE below). Excluded on purpose: frame-stride results,
TempoKV/CORAL-KV comparisons, cov-TAE. Each section: 2-4 sentences saying what is tested, how, and what it means
for the design; then the table/figure. Overview table at the top maps sections to paper claims.

## Part A. Method details (what a reader needs to reimplement)
A1 Formats and bit accounting: INT4 keyframe g=16 with 8-bit scale index on a uniform grid (largest group scale/255);
   1-bit residual = sign + 8-bit scale per 16 values (1.5 bits); TCC-2 residual = asym INT2 g=64, 16-bit scale+zero;
   packing (INT3 two codes per byte etc.); "bits" includes keyframes, scales, chains kept for decoding.
   Source: research/models/stream_cache.py, paper Sec. 3. No new runs.
A2 Decoder: chain walk, one layer in full precision at a time, fused Triton kernel (one launch per layer, writes into
   the attention input, fixed size -> no fragmentation), bit-identical to PyTorch on 7 configs; encoder reuses its codes
   (bit-identical, checked on 5 configs). Complexity O((N+K)C) + Table timing vs K (DONE, keep).
A3 StreamVGGT integration: what is cached (K/V per layer), KIVI and INT baselines in the same interface,
   XStreamVGGT reimplementation. Source: tcc/streamvggt_cache.py.
A4 Residual statistics behind the bound: contraction 35.5% mean / 65.6% max (TCC-1), TCC-2 19-23% mean / 47% max,
   keys vs values. Source: results/2026-11-09_svg_queue4 or svg_queue5 contraction (check).

## Part B. Video Depth Anything
B1 DONE  Flicker + per-frame alignment, Bonn and Sintel (tab_full_*). Source: fidelity runs via make_paper_assets.
B2 Short windows win16/win8 on all sets (win8 cache 44-47% of TCC-1, delta1 up to -0.027). Source: 2026-11-13_vda_extra2.
B3 DONE  Bonn5, KITTI, ScanNet tables.
B4 Open-loop control on all backbones (12 ratios; paper says 1.7-2.7x). Source: 2026-11-02_streaming/streaming_vda/*_open.json.
B5 Statistics: bootstrap ratios with 95% CI, wins per scene, per-scene SD for every comparison in Sec. 5.1;
   Bonn per-scene tables (paper promises "number of scenes each method wins"). Source: results/paper_bootstrap.json.
   Do not claim the non-significant ones (TCC-1 K8 vs INT2 Sintel S/L).
B6 DONE  Scene cuts + adaptive keyframes (fig_cut, tab_splice).
B7 DONE  Error through the temporal modules (fig_modules, tab_modules).
B8 DONE  Redundancy per layer (fig_redundancy, tab_redundancy).
B9 Speed: A100 table protocol, encoder/decoder split, before/after fast encoder (vda_timing_a100*.json).
B10 Qualitative: depth and difference maps on more frames/scenes (stream_grid from cached predictions; extend).

## Part C. StreamVGGT
C1 Official benchmark, all formats (Sintel 23, Bonn 5, KITTI 13): cache MB, bits, Dev, Flk, AbsRel, delta1,
   per-scene wins and bootstrap CIs. Source: results/fidelity_{sintel,bonn,kitti}.json, paper_bootstrap.json.
C2 Frames until out of memory on a 40 GB GPU, all formats, both sequences (282 / 960 / 924 / 1144 / 2287 / 1847 / 1460).
   Source: svg queue runs (streamvggt_oom). Check which folder.
C3 Long videos: per-100-frame deviation curves (crowd2 755 frames, all caches), 495-frame curves (crowd2, crowd3),
   camera-centre error on Sintel (13.3% vs 38.5-43.1%), per-scene values of the long-video table.
C4 Speed: clean interleaved timing (svg_time2) and GPU profile breakdown (svg_prof3): decoding 18-23 ms/frame.
C5 XStreamVGGT: authors' released code vs our implementation (paper: ours deviates less in 3 of 4 settings - PROMISED);
   token-budget sweep (fid_xs_*). Omit XStream+TCC (does not shrink the cache at 2048 tokens).
C6 STAC: released code, modes, our reproduction of its paper numbers (Bonn 0.0595/97.11 vs 0.060/97.1; KITTI
   0.1919/67.41 vs 0.192/67.6). Do not compare against STAC "full" (offline, no causal mask).
C7 3D reconstruction (7-Scenes, NRGBD): dense and official-stride settings, Acc/Comp/NC, point deviation per scene,
   adaptive-keyframe bits. Source: svg_recon logs, recon_fidelity2.json (2026-11-11_timing has recon_fidelity.json).

## Part D. Reproducibility
D1 Hardware, software versions, input sizes (VDA 518, Sintel 392), data versions and splits (ScanNet scenes, Bonn sequences).
D2 Commands for every table (README of the anonymous release) and the result-file -> table mapping.

## Checks before submitting
- scripts/audit_supp.py: every number in the supplement text traced to results/ (as audit_paper.py).
- Anonymity: no names, server paths, IPs in PDF text or metadata; \pdfinfoomitdate and empty trailer id (as main.tex).
- Page flow: each section text -> its table/figure; S-numbering; cross-references to paper sections.
