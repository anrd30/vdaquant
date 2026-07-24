# Related Work Notes — DG-4 kill-shot literature review (2026-07-25)

Executed per docs/phase4_a_star_plan.md §5. Purpose: decide whether the paper's
novelty claims survive. **Verdict: the method-level novelty does NOT survive; a
narrow domain-level slice does.** Threat levels: HIGH = directly claims something
we wanted to claim; MED = adjacent/overlapping; LOW = background/citation.

Two most consequential papers were fetched and confirmed directly (not just
title/abstract): Quant VideoGen and the 33-method study. Others are from search
snippets and should be read in full before the related-work section is written.

## Bucket 1 — lattice / vector quantization on KV cache  → HEAVILY CLAIMED

| Paper | arXiv | Threat | Note |
|---|---|---|---|
| Leech Lattice VQ for Efficient LLM Compression | 2603.11021 | **HIGH** | Explicitly beats QuIP#, QTIP, PVQ. Our "push to Leech" idea is already published, and it's SOTA. |
| Learning Grouped Lattice VQ for Low-Bit LLM | 2510.20984 | **HIGH** | Grouped lattice VQ — overlaps our grouped-scale framing. |
| FibQuant: Universal VQ for Random-Access KV-Cache | 2605.11478 | **HIGH** | Random-access VQ KV cache — the exact deployment concern we raised as "lattice breaks random access". Solved by others. |
| Hurwitz Quaternion Multiplicative Quant for KV | 2605.27646 | MED | Exotic algebraic-structure VQ for KV. |
| GSRQ: Gain-Shape Residual Quant, sub-1-bit KV | 2607.01065 | MED | Residual + gain-shape VQ, sub-1-bit. |
| VecInfer: Outlier-Suppressed VQ KV cache | 2510.06175 | MED | Outlier suppression (~ our rotation motivation) + VQ KV. |

Conclusion: **E8-lattice-on-KV-cache is not novel.** Lattice/VQ KV quantization is a
crowded, fast-moving area with SOTA already past E8 (Leech). This is the DG-4 kill
condition for the *method* firing exactly as the plan feared.

## Bucket 2 — video-GENERATION KV quantization (our "standardize to video gen" target) → CLAIMED, active

| Paper | arXiv | Threat | Note |
|---|---|---|---|
| **Quant VideoGen** (ICML 2026) | 2602.02958 | **HIGH (confirmed by fetch)** | Auto-regressive video gen, training-free, 2–4 bit KV, **progressive RESIDUAL quantization + semantic-aware smoothing exploiting spatiotemporal redundancy**, 7× KV reduction. This IS the temporal-residual idea, for the exact generalization target the boss named. |
| **33-Method Empirical Study, self-forcing video gen** | 2603.27469 | **HIGH (confirmed by fetch)** | The "honest empirical KV-quant benchmark" niche — for video gen. Even states "nominal compression alone is insufficient" (our honest-accounting angle) and "methods appear stable while drifting structurally" (adjacent to our TAE-gameability). |
| Quantized Keys Steal Attention: bias correction, video diffusion | 2605.26266 | **HIGH** | QJL-style attention-bias correction, for video diffusion. Our QJL angle, claimed for video. |
| Attend Locally, Remember Linearly (linear attn cross-frame memory) | 2605.16579 | LOW | Different mechanism (linear attention), background. |

Conclusion: **the video-generation KV-quant space is already active at ICML 2026**,
and it already contains (a) residual KV quant, (b) an honest empirical benchmark,
(c) bias correction. The "generalize/standardize to video generation" ambition is
3–6 months behind multiple groups.

## Bucket 3 — video-DEPTH quantization → NARROW SLICE SURVIVES

| Paper | Threat | Note |
|---|---|---|
| Depth Anything V2 INT8 via OpenVINO/NNCF (Medium/engineering) | LOW | INT8 weight quant of the *image* model, engineering not research. No KV cache, no video temporal path. |
| Video-Depth-Anything (CVPR 2025, the base model) | — | The model we quantize; no quantization of its KV cache exists. |

Conclusion: **KV-cache quantization of a video DEPTH (discriminative dense-prediction)
model appears genuinely unclaimed.** This is the one surviving novel slice.

## Bucket 4 — temporal-redundancy KV compression → CLAIMED (incl. 3D/streaming)

| Paper | arXiv | Threat | Note |
|---|---|---|---|
| AttentionPredictor (NeurIPS 2025) | 2502.04077 | MED | Learns temporal attention patterns for KV compression. |
| STAC: Spatio-Temporal Aware Cache, streaming 3D recon | 2603.20284 | **HIGH** | Spatio-temporal cache compression for streaming 3D reconstruction — closest to our geometry/depth domain. Read in full. |
| PureKV: spatial-temporal sparse attn, VLMs | 2510.25600 | MED | Temporal-redundancy KV purification for vision-language. |
| GUI-KV (projects old frames into present key subspace) | — | MED | Temporal-redundancy scoring across frames. |

Conclusion: temporal-redundancy KV compression is also an active, claimed area,
now reaching into 3D/streaming-geometry (STAC).

## Overall DG-4 verdict

- **Method novelty (lattice / rotation / residual / QJL on KV cache): DEAD.** Every
  one of these has a 2025–2026 paper, several in the last few months, some SOTA past
  ours (Leech), some in the exact video target (Quant VideoGen).
- **"Set a new standard for video-KV compression": not available.** The window is
  occupied by groups publishing at ICML/NeurIPS 2026.
- **What genuinely survives:**
  1. KV-cache quantization *specifically for a video-DEPTH / dense-geometry model*
     (discriminative, geometric-accuracy eval — distinct from all the generation work).
  2. The **geometric-TAE-gameability finding** (F16): a compression method winning a
     geometric temporal-consistency metric while collapsing on accuracy. The video-gen
     33-method study noticed the *general* "appear stable while drifting" phenomenon
     but did NOT formalize metric-gaming or do it for geometric reprojection TAE.
- **Honest positioning:** this is a narrow, defensible **workshop / borderline-conference
  paper in the efficient-video-vision niche**, framed as "the first careful, honestly-
  accounted study of KV quantization for video depth, and a demonstration that geometric
  temporal-consistency metrics are gameable under compression." It is NOT a standard-setter,
  and any framing that implies method novelty will be desk-checked against the papers above.

TODO before writing related-work: read in full 2603.11021 (Leech), 2602.02958
(Quant VideoGen), 2603.27469 (33-method), 2603.20284 (STAC), 2605.26266 (bias
correction video diffusion). Confirm none has already done video-DEPTH specifically.
