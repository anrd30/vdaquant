# Third model, XStreamVGGT baseline, and prior-art check (2026-11-02)

Research notes for the CVPR 2027 submission (deadline Nov 16, 2026). Everything below was
read from code or fetched from the cited page on 2026-11-02. Items marked **[UNVERIFIED]**
were not checked against a primary source, or are my inference from code and not an
executed result. No model checkpoints or datasets were downloaded.

Code clones (read-only references, all under 100 MB, shallow):
`C:\Users\aniru\third_party\` -> `OnlineVideoDepthAnything`, `StreamVGGT`, `STream3R`,
`XStreamVGGT` (main + fetched `prune_and_quantize` branch), `STAC`, `Tempokv`.

---------------------------------------------------------------------------------------

## 0. Recommendation (read this first)

### 0.1 Third model: integrate STream3R, not oVDA

Reasons, in order of weight:

1. **STream3R has the same disease as StreamVGGT.** Its global-attention KV cache grows
   linearly with frames: README table, 518x384, H200: 5.49 GB at 1 frame, 45.41 GB at 200
   frames, i.e. ~0.2006 GB/frame. Arithmetic check: 24 layers x 2 (K,V) x 16 heads x 1004
   tokens x 64 dim x 4 B (fp32; the model runs under `torch.autocast(dtype=torch.float32)`,
   `stream3r/models/stream3r.py:81`) = 197.5 MB/frame. So the cache really is fp32 and
   our 8x-16x compression claim has a large headroom story there.
2. **oVDA's cache is tiny, so the memory story is weak.** At 280x924 (the paper's Tab. 2
   resolution) the whole cache at c=16 is ~51 MB in fp16 (computation in section 1.3) out
   of 0.45 GB total VRAM. Only ViT-S (29 M params) is released. Compression would be
   "11% of VRAM -> ~2%". It is nearly the same code path as the VDA streaming mode we
   already have, so it is cheap, but it adds little new evidence.
3. **Integration patch point is the same as StreamVGGT.** STream3R
   `stream3r/models/components/layers/attention.py::Attention.forward` takes `kv_cache=[k,v]`
   and does `torch.cat([k_cache, k], dim=2)`. One difference from StreamVGGT that helps us:
   STream3R (like XStreamVGGT's repo) caches **post-q/k-norm, post-RoPE K in a flat
   `[B,H,N_tokens,D]` layout**. Since RoPE here is 2-D patch position only (no frame
   index), the same patch index has the same rotation in every frame, so frame-to-frame
   deltas on post-RoPE K are well defined. Reshape `(B,H,S*P,D)` -> `(B,H,S,P,D)` with
   P = 1004 (1 camera + 4 register + 999 patch tokens at 518x378; **[UNVERIFIED]** exact P
   depends on dataset crop) and the existing `research/models/streamvggt_cache.py`
   compressor applies almost unchanged.
4. **Eval scripts and a ready wrapper exist.** `eval/video_depth/run.sh` runs Sintel,
   Bonn, KITTI with `StreamSession(model, mode="causal")`, and the STAC repo
   (MIT, `Rainzor/STAC`) already wraps both StreamVGGT and STream3R behind one
   `StreamSession` and ships window/H2O baselines. This lets us also report STAC-style
   window baselines on both backbones cheaply.
5. **Same protocol as StreamVGGT/XStreamVGGT**, so one harness and one results table
   cover both models (per-sequence scale alignment, same datasets, see section 2.4).

Estimated effort (one person, assumes datasets from the MonST3R/Spann3R prep are
available; the authors host them at `huggingface.co/datasets/yslan/pointmap_regression_evalsets`):

| Step | Time |
|---|---|
| Env + 4.76 GB checkpoint + FP baseline on Sintel/Bonn/KITTI | 0.5-1 day |
| Adapter: flat 4-D cache <-> (B,H,S,P,D) view, hook `StreamSession` (compressed object in `aggregator_kv_cache_list[i][k]`) | 1 day |
| Delta codec runs (INT3/INT4 keyframe K=8, 1-bit residual) + plain INT2/3/4 + KIVI-INT4 | 1-1.5 days |
| XStreamVGGT-style prune+quant port (nearly copy-paste, same cache layout) | 0.5-1 day |
| Plots / table / debugging buffer | 1-2 days |
| **Total** | **~4-6 days** |

GPU note: the fp32 uncompressed baseline needs ~0.2 GB/frame. Bonn (110 frames in
`rgb_110`) is ~27 GB by the README slope, which does not fit a 24 GB RTX 4090
**[UNVERIFIED, extrapolated from the H200 table]**; run the FP baseline on an A100
(40/80 GB) and the compressed runs wherever. Sintel (<=50 frames) fits a 4090.

### 0.2 If time allows: oVDA as a cheap fourth row (1-2 days)

Because oVDA caches exactly what our VDA streaming mode caches (pre-K/V hidden states
of the temporal attention, section 1.3), the integration is ~100 lines. Frame it as
"edge / Jetson-class memory budget", not as a memory-reduction headline. Check the
license first (section 1.2): NC-SA-UHDV1.0 is share-alike, so keep our patch outside their
files (monkeypatch/wrapper) so no derived file has to be re-licensed.

### 0.3 Re-implementing XStreamVGGT faithfully (details in section 3)

Port the authors' two code paths, not just the paper text:

- **Pruning** = `Attention._prune_fast_vggt_protected` in
  `XStreamVGGT/src/streamvggt/layers/attention.py` (budget 2048 tokens, pool 16).
- **Quantization** = the `KV_QUANT_MODE=KCVT` block on branch `prune_and_quantize`
  (INT4, group 64, K per-channel with groups over the token axis, V per-token).
- Both operate on **post-norm, post-RoPE K** in a flat `[B,H,T,D]` cache. Stock
  StreamVGGT caches raw pre-norm, pre-RoPE K in `[B,H,S,N,D]`. Our existing
  `research/models/xstream_baseline.py` stores raw pre-RoPE K and re-applies norm/RoPE
  after dequantization, so its K quantization sees different statistics than the
  authors' code. Run both variants and label them (faithful = post-RoPE; ours-strong =
  pre-RoPE).
- **The "2K" budget is about 2 frames, not 2000 frames.** With ~1000 tokens/frame the
  rule `middle_keep = max(0, 2048 - N_new - N_new)` leaves ~40 middle tokens on Bonn, so
  XStreamVGGT at its default budget is effectively "first frame + current frame (+ the
  previous frame inside the current step)". Report budgets in frames as well as tokens,
  and include a quantization-only row (`budget=inf`) so the comparison with ours is
  apples to apples (we keep every frame; they drop almost all of them).
- The released quantization is **simulated** (fake-quant, tensors stay fp). Count real
  bytes including scale/zero (INT4, group 64, fp16 min+scale = 4.5 bits/elt).

---------------------------------------------------------------------------------------

## 1. Online Video Depth Anything (oVDA)

### 1.1 Basics
- Paper: arXiv 2510.09182, "Online Video Depth Anything: Temporally-Consistent Depth
  Prediction with Low Memory Consumption", Feiden, Kuchler, Zavadski, Savchynskyy,
  Rother (Heidelberg). 3DV 2026 per the task statement (**[UNVERIFIED]**, I did not see
  the venue on the pages I fetched).
- Site: https://friedfeid.github.io/oVDA-website/
- **Code is public:** https://github.com/FriedFeid/OnlineVideoDepthAnything (GitHub
  pushed_at 2026-03-09; README news: "2026-03-09: Code and model weights released").
- **Checkpoints are public:** https://huggingface.co/FriedFeid/oVDA (not gated).
  - `oVDA_c16.pth` 116,345,367 B (~116 MB)
  - `oVDA_c8.pth` 116,299,380 B (~116 MB)
  - **Only the ViT-S model (29.0 M params) is released**, in two cache sizes (c16, c8).
    README: "We provide the smallest model, based on vits (29.0M parameters)."
    No ViT-L release.
- **License:** `LICENSE` = "Non-Commercial Software License - Heidelberg University
  Version 1.0 (NC-SA-UHDV1.0)", with Apache-2.0 only for portions derived from VDA that
  are marked in files. Non-commercial, ShareAlike (derivatives must carry the same
  license), attribution required. HF card says `license: other / nc-sa-uhdv1.0`.
  Academic use is fine; keep our changes outside their files.
- Repo contents (read): `run.py`, `src/models/video_depth.py`,
  `src/models/modules/dpt_temporal.py`, `src/build_ONNX/*` (ONNX/TensorRT build for
  Jetson), `configs/oVDA_c16.yaml`, `configs/oVDA_c8.yaml`. **There is no evaluation
  script** (no Sintel/Bonn/KITTI loaders); we must reuse our own harness.

### 1.2 What the cache is (from the code)
Defined in `src/models/modules/dpt_temporal.py::TemporalAttention.forward`
(called through `TemporalTransformerBlock` -> `TemporalTransformer3DModel` ->
`TemporalModule` -> `DPTHeadTemporalCrossAtt.forward`).

- It caches the **normalized hidden state that enters the temporal attention** (output
  of `norm(hidden_states)` per attention block), i.e. the *pre-K/V-projection input*,
  returned as `curr_hidden_states` (`return hidden_states, curr_hidden_states`). K and V
  are **recomputed from the whole cache every frame** (`key = self.to_k(encoder_hidden_states)`,
  `value = self.to_v(encoder_hidden_states)`), after adding sinusoidal APE to all slots
  (`self.pos_encoder`, config `pe: ape`).
- Only the **current frame is the query** (`pos_enc_curr_hidden_states` is one frame);
  the cache slots are keys/values. A frame never re-computes the past.
- Four motion modules (`self.motion_modules[0..3]`), each with `num_transformer_block=1`,
  `num_attention_blocks=2`, `num_attention_heads=8`. One cache tensor per module:
  `{'c1','c2','c3','c4'}` with shape `[1, 2, tokens, cache_size, C]`
  (`DPTHeadTemporalCrossAtt`/`setup_cache` in `src/models/video_depth.py`;
  `src/build_ONNX/compile_utils.py::load_dummy_data` is the same). Dim 1 (=2) is the two
  attention blocks.

  | cache | channels C | tokens (H/14 = hp, W/14 = wp) |
  |---|---|---|
  | c1 | 192 | hp*wp |
  | c2 | 384 | ceil(H/28)*ceil(W/28) |
  | c3 | 64 | hp*wp |
  | c4 | 64 | ceil(2H/14)*ceil(2W/14) |

- Cache length: `cache_size` 16 (default, `oVDA_c16.yaml`) or 8 (`oVDA_c8.yaml`, which
  also sets `cross_att: true`, `freeze_backbone: true`). The last slot is the scratch
  slot for the current frame (`input_position=[cache_size, cache_size]`), so history is
  `cache_size-1` frames. `mask_indices` masks unfilled slots with -10000 during warm-up.
- dtype: fp16 by default (`infer_video_depth(..., fp32=False)` calls `.half()` on the
  cache and model); `--fp32` optional.
- Update rule (`onlineVideoDepthAnything.infer_video_depth`, ~lines 99-113): once
  full, `torch.cat([cache[..., 1:cache_size, :], output_cache, zeros_like(output_cache)], dim=3)`
  (shift-left sliding window); before that, write at slot `cache_size`.
- Training: causal ("lower-triangular") frame masking so past influences future only
  (paper). 130k fine-tune steps at lr 1e-6 (paper).

### 1.3 Size of the cache (my arithmetic)
Values per slot, summed over c1..c4 and the 2 attention blocks, at 280x924
(20x66 patches): c1 = 1320x192, c2 = 330x384, c3 = 1320x64, c4 = 5280x64 -> 802,560 per
attention block -> 1,605,120 per slot. At c=16, fp16: 16 x 1,605,120 x 2 B = **~51 MB**
(c=32 would be ~103 MB). The paper reports 0.45 GB total VRAM at c=16 and 1.36 GB at
c=32, so the cache is a minority of VRAM and the jump from 0.45 to 1.36 GB is **not
explained by cache size alone [UNVERIFIED why]**.

### 1.4 Evaluation protocol and reported numbers (arXiv HTML, 2510.09182)
- Zero-shot KITTI (avg 311 frames/seq), Bonn (avg 1118), Sintel (avg 46); full-length
  sequences; depth clipped at 80 m. Scale and shift are fit **on the first frame only**
  and applied to the whole sequence (online-friendly); Tab. 8 (supplement) reports global
  alignment over the first 500 frames or all frames.
- Metrics: **AbsRel and delta1 only.** No TAE; scale drift is shown as a plot (Fig. 4).
  No ScanNet/NYUv2. (This supports our "TAE is gameable" angle: the oVDA paper does not
  rely on TAE.)
- Tab. 1 (AbsRel / delta1): oVDA KITTI 0.140/0.809, Bonn 0.118/0.871, Sintel 0.380/0.548;
  FlashDepth 0.165/0.760, 0.131/0.789, 0.370/0.564; FlashDepth-s 0.156/0.774,
  0.116/0.848, 0.355/0.532; CUT3R 0.238/0.690, 0.116/0.850, 0.635/0.400.
- Tab. 2, A100, 280x924: oVDA **42 FPS, 0.45 GB**; FlashDepth-s 60 FPS/0.69 GB;
  FlashDepth 30 FPS/2.72 GB; CUT3R 10 FPS/6.70 GB; DAv2-s 72 FPS/0.25 GB.
- Tab. 3, **Jetson Orin NX**, KITTI, 280x924: FP32 9 FPS/0.64 GB;
  FP16 **20 FPS/0.49 GB**, AbsRel 0.1398, delta1 0.8093.
- Tab. 6 (cache length): c=8 0.146/0.800, 43 FPS, 0.37 GB; c=16 0.140/0.809, 42 FPS,
  0.45 GB; c=32 0.136/0.811, 39 FPS, 1.36 GB.

### 1.5 Integration plan for a compressed cache
Wrap `TemporalAttention.forward` (single choke point, one call per attention block per
module per frame):

- Compress at the *write* in `onlineVideoDepthAnything.infer_video_depth`'s
  "Update Cache" block (and the `lazy_forward` copy), operating on `output_cache[key]`
  (`[1,2,tokens,1,C]`), and decode at the *read* just before `forward()` (or inside
  `TemporalAttention.forward` at `encoder_hidden_states = input_cache.clone()`).
- Delta coding is natural here: the cached quantity is a per-frame feature map at fixed
  token positions, so closed-loop delta vs. the previous *reconstructed* slot works as in
  our VDA streaming code. Keyframe every K=8 matches c=8/c=16 cache lengths (c8 history = 7
  slots, so K=8 means a single keyframe in-window; consider K=4 for oVDA).
- ONNX/TensorRT export (`src/build_ONNX`) passes the cache as explicit I/O tensors
  (`cache_dict.c1..c4`), so a decode step would have to be fused into the graph; skip for
  the paper, evaluate in PyTorch only.
- Effort: ~1-2 days given the existing VDA streaming code (same tensor semantics:
  pre-projection hidden states), plus an eval loop (they ship none).

---------------------------------------------------------------------------------------

## 2. STream3R and the other streaming models

### 2.1 STream3R basics
- Paper: arXiv 2508.10893, "STream3R: Scalable Sequential 3D Reconstruction with Causal
  Transformer", ICLR 2026 (README header). Authors Lan, Luo, et al. (NTU S-Lab).
- Code: https://github.com/NIRVANALAN/STream3R (pushed 2026-05-08). Inference, eval
  and training code are released. A DUSt3R-based metric-scale variant lives on the
  `alpha` branch (README news, Mar 9 2026); the main branch is the VGGT-based "beta".
- Checkpoint: https://huggingface.co/yslan/STream3R `model.safetensors` 4,762,545,592 B
  (~4.76 GB, fp32). HF license "other"; repo `LICENSE` = S-Lab License 1.0
  (non-commercial redistribution terms). Auto-downloaded by `STream3R.from_pretrained`.
- Dependencies: python 3.11, torch 2.8 / CUDA 12.6 in README; FlashAttention optional.

### 2.2 Cache implementation (read from code)
- `stream3r/stream_session.py::StreamSession(model, mode)`; `mode` in
  {"causal","window"}. Holds `aggregator_kv_cache_list[depth][2]` (depth = 24 global
  layers; entry `[i][0]`=K, `[i][1]`=V) and `camera_head_kv_cache_list[4 iters][trunk_depth][2]`.
- `stream3r/models/components/layers/attention.py::Attention.forward(x, pos, attn_mask, kv_cache)`:
  computes qkv, **applies `q_norm`/`k_norm` and RoPE, then**
  `k = torch.cat([k_cache, k], dim=2)`, `kv_cache = [k, v]`; SDPA over the whole cache.
  So cached K is **post-norm, post-RoPE**, layout `[B, heads=16, N_tokens_total, 64]`.
- Aggregator path: `streamaggregator.py::_process_global_attention(... kv_cache=...)`
  returns the updated cache per global layer; `stream3r.py::forward` threads
  `aggregator_kv_cache_list`/`camera_head_kv_cache_list` through.
- Window mode (`StreamSession._update_cache`): keeps the first frame's P tokens
  (`anchor_token = cache[:, :, :P]`) plus the last `window_size*P` tokens, window_size=5
  (camera head: 1 anchor + 5 tokens). This is the "STream3R-W[5]" configuration in the
  paper, and an in-repo example of bounded cache we can use as a baseline.
- Memory: ~0.2 GB per frame at 518x384 fp32 (section 0.1). README timing, causal: 0.12 s
  (1 frame) -> 1.17 s (200 frames) per streamed frame; window: ~0.15 s flat; VRAM window
  mode flat at 6.53 GB.

### 2.3 Where to hook a compressed cache
Smallest change: in `StreamSession._update_cache`/`_get_cache`, store a compressed
object per `[layer][k]`, and in `Attention.forward` decode it before the `torch.cat`
(or pass a decoder in via the `kv_cache` argument). The aggregator's
`is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None` test and
`slice_expand_and_flatten` need the "cache is empty" state preserved. The camera-head
cache is tiny (1 token per frame per iter); leave it uncompressed (state this in the paper).

### 2.4 Evaluation protocol (README + `eval/video_depth/`)
- Datasets Sintel, Bonn, KITTI (`run.sh`: `datasets=('sintel' 'bonn' 'kitti')`), images
  loaded with `load_images_for_eval(size=518, patch_size=14)`, `StreamSession(model,
  mode="causal")`, `eval_depth.py --align scale` (per-sequence scale; `scale&shift` and
  `metric` also available). Depth caps in `eval_depth.py`: `max_depth=70` for the first
  two branches shown, `None` for the third (the script is multi-branch; **check which
  branch Sintel/Bonn/KITTI hit before reporting**, not resolved here).
- Paper Tab. 2 (Abs Rel / delta<1.25, per-sequence scale):
  STream3R-beta Sintel 0.264/70.5, Bonn 0.069/95.2, KITTI 0.080/94.7, 12.95 FPS (KITTI);
  beta-W[5] 0.279/68.6, 0.064/96.7, 0.083/95.2, 32.93 FPS; CUT3R 0.421/47.9, 0.078/93.7,
  0.118/88.1, 16.58 FPS. Scale-and-shift (Tab. 7): STream3R 0.356/58.6, 0.068/95.7,
  0.099/91.0. The paper does **not** put StreamVGGT in its video-depth table (only NRGBD,
  Tab. 8).
- Note: StreamVGGT's own numbers (Sintel 0.328/65.8, Bonn 0.058/95.9, KITTI 0.094/94.4,
  as quoted in the XStreamVGGT paper) are a different checkpoint/eval; do not mix tables
  across papers.

### 2.5 Other streaming models: does a growing per-frame cache exist?

| Model | Memory | Compressible growing cache? |
|---|---|---|
| **StreamVGGT** (wzzheng, 1.26 B params, 5.03 GB fp32 `lch01/StreamVGGT`, CC BY-NC-SA 4.0) | Global-attn KV, `[B,H,S,N,D]` raw pre-norm/pre-RoPE K, V per layer, never evicted | Yes (our main target) |
| **STream3R** | Global-attn KV, grows | Yes |
| **CUT3R / TTT3R** | Fixed recurrent state (CUT3R `model.py` has `state_size` arg default 324 plus `local_mem_size=256` pose memory; released config value **[UNVERIFIED]**, the paper uses a fixed number of state tokens) | **No growing cache.** State is fixed-size and overwritten, so delta/KV compression has nothing to compress. Confirmed. |
| **Point3R** (arXiv 2507.02863, NeurIPS 2025, github.com/YkiWu/Point3R) | Explicit "spatial pointer memory" that **expands as exploration proceeds**, with a fusion mechanism keeping it spatially uniform (abstract/v2 HTML). Memory is indexed by 3D position, not by frame. | Grows with scene extent, not frame index; the per-pointer features are not frame-aligned, so frame-to-frame delta coding does not apply directly. Skip. |
| **FlashDepth** (arXiv 2504.07093, ICCV 2025, Eyeline-Research/FlashDepth) | Mamba recurrent module on DPT features (hidden state fixed-size). README: Mamba layers moved to after the first DPT layer in the Full model; ablation variants Hydra, xLSTM, and a CUT3R-style transformer RNN | **No growing cache** (recurrent hidden state). Confirmed. |
| **oVDA** | Sliding window of pre-projection features, 7-15 history frames | Fixed-length, small (section 1) |

---------------------------------------------------------------------------------------

## 3. XStreamVGGT and related cache compressors

### 3.1 Identity and availability
- Title: "XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded
  Transformer with KV Cache Compression", Su, Ye, Feng, Fan, Zhang, Yu, Liu, Wong.
- **Two arXiv IDs with identical title and authors:** 2601.01204 (v1, 3 Jan 2026) and
  2602.21780 (v1, 25 Feb 2026). Cite 2601.01204 (the repo README links it). The second is
  probably a re-submission **[UNVERIFIED]**. Also published at SID Display Week 2026 /
  J. SID (DOI 10.1002/jsid.70086, via search result).
- Code: https://github.com/ywh187/XStreamVGGT (CC BY-NC-SA 4.0). **Branch `main` = pruning
  only; branch `prune_and_quantize` = pruning + simulated quantization.** The README's
  own "Code" link is a placeholder (`your-repo/XStreamVGGT`).
- Weights: same as StreamVGGT (`lch01/StreamVGGT`).

### 3.2 Exact method, as implemented
File: `XStreamVGGT/src/streamvggt/layers/attention.py` (`Attention`), env vars
`KV_PRUNE_MODE` (XStreamVGGT / SlidingWindow / Random), `KV_CACHE_SIZE` (default
constructor arg 8192, README command uses 2048), `KV_POOL_SIZE` (default 16), and on the
quant branch `KV_QUANT_MODE` (default "KCVT").

**Cache layout.** Changed from stock StreamVGGT. `forward` applies `q_norm`/`k_norm`
and RoPE to the *new* frame's q and k, then concatenates onto a flat `[B,H,T,D]` cache,
so the cache holds **post-norm, post-RoPE K**. (The aggregator was patched to match:
`_, _, S_true, _ = past_key_values[0][0].shape`, so `S_true` is now a token count, only
its last element is used.) Both global blocks and the camera-head blocks use this
`Attention` class, so the camera-head cache is subject to the same rule, which never
triggers below 2048 tokens (1 token/frame/iter) at our sequence lengths.

**When pruning happens.** Per layer, **after** the attention of the current frame, on
the concatenated cache (`new_kv = self.prune_kv_cache(new_kv, q)`), only if
`T > cache_size` where T includes the current frame. So the current frame's attention
always sees the previous step's pruned cache plus its own K/V; pruning decides what is
carried to the next frame.

**Criterion** (`_prune_fast_vggt_protected`), N = number of current-frame tokens:
- protected = first N cache tokens (frame 0), window = last N tokens (current frame);
  `middle_keep = max(0, cache_size - N - N)`; prunable = tokens in `[N, T-N)`.
- Query side: current-frame q (post-norm, post-RoPE); first 5 tokens (camera + 4 register)
  kept unpooled; remaining patch queries average-pooled in groups of `pool_size`=16
  (leftover pooled into one extra token); then **mean over heads**.
- Key side: prunable keys **mean over heads**.
- Score = `(q_score @ k_score^T).mean(dim=1)` (no softmax); `topk(middle_keep)`; keep set
  = protected + topk + window, sorted, gathered identically for all heads.
- Matches the paper's Eqs. 2-6 (g=16, head average, mean over pooled queries).
  Paper says budget L_max = 2K, ablated over 2K/4K/6K/8K on Sintel (figure only,
  "minimal change"), selected 2K.

**Consequence of the 2K budget [my derivation, not executed].** With long side 518 and
14-px patches, Bonn (640x480) gives ~27x37 = 999 patches + 5 = ~1004 tokens/frame
(the code comments show 1041 = 28x37+5 at another crop), so
`middle_keep = 2048 - 2008 = ~40` tokens; Sintel (1024x436) ~560 tokens/frame gives
~928 middle tokens (~1.7 frames); KITTI (cropped, ~412 tokens/frame) ~1224 (~3 frames).
I.e. the default operating point retains roughly 2-5 frames of context, and the
reported 4.42x memory / 5.48x speedup come from that. The paper only shows exact
length-vs-memory curves in figures (Fig. 1, Fig. 5), so I cannot quote the per-length
values.

**Quantization (branch `prune_and_quantize`, block inside `Attention.forward`).**
- Constants hard-coded: `k_bit = v_bit = 4`, `gs = 64`, `preserve_tokens = 0`,
  `clip_ratio = 1`. **Asymmetric** min/max quantization per group
  (`find_params_groupwise_lastdim_safe`: `scale=(max-min)/(2^b-1)`, `zero=round(-min/scale)`).
- `KCVT` (default): K is transposed so the last dim = tokens; **groups of 64 consecutive
  tokens per channel** (per-channel K, KIVI-style); V quantized along the last dim = head
  dim 64, i.e. **one group = the whole head vector of one token** (per-token V). Modes
  KTVT/KTVC/KCVC also exist.
- Applied to the **entire concatenated cache (old + current frame) at every step, before
  pruning**, so already-dequantized entries are re-quantized each frame, and pruning /
  scoring then operate on the quantized K/V. The current step's SDPA uses the
  unquantized local `k, v` (the quantized copy only goes into the returned cache).
- **No full-precision residual window, no outlier handling, no calibration.**
- **Fake quantization only**: tensors stay in the input dtype; there are also unconditional
  `print(f"kvtc=...")` calls per layer/frame (slow). Real memory savings from quantization
  are therefore not realized by their code; the 4.42x number is accordingly the paper's
  claim, **not reproduced by released code [UNVERIFIED which component yields it]**.

**Their reported numbers (arXiv HTML v1; Point3R protocol, max edge 518; efficiency on a
single A100-80GB, 50-1000 frames; video depth with per-sequence scale as in
StreamVGGT's `run.sh`).** Abs Rel / delta<1.25 (StreamVGGT -> XStreamVGGT):
Sintel 0.328/65.8 -> 0.341/61.9; Bonn 0.058/95.9 -> 0.077/97.1; KITTI 0.094/94.4 ->
0.098/94.3. 7-Scenes Acc mean 0.132 -> 0.142; NRGBD Acc mean 0.085 -> 0.085. Camera pose
(90-frame truncation): TUM ATE 0.062 -> 0.068, ScanNet ATE 0.160 -> 0.171. Headline
4.42x memory, 5.48x speed. Note the "negligible loss" claim is generous: Sintel Abs Rel
+4%, delta1 -3.9 pts, Bonn Abs Rel +33% (delta1 improves, which is odd).

### 3.3 Cross-check of our existing baseline (`research/models/xstream_baseline.py`)
Compared against the authors' actual code (the file docstring says it was written from a
"text summary"):

| Item | Authors' code | Ours | Verdict |
|---|---|---|---|
| Score (pooling 16, head mean, 5 unpooled specials, no softmax) | as above | `importance_scores` | matches |
| `middle_keep = budget - N_first - N_cur`, protect first frame + current | yes | `select_keep` | matches |
| Attention over pruned-cache + current, prune after | yes | yes | matches |
| Quant: KCVT INT4 g64 asymmetric, re-quantize whole cache each step | yes | `_KiviTensor` token/channel | matches (we also bit-pack and store fp16 min/scale) |
| **Tensor quantized** | post-norm, **post-RoPE** K | raw **pre-norm, pre-RoPE** K (`cache.update(k.detach(), ...)`), norm+RoPE re-applied after decode | **differs**; ours is likely friendlier to per-channel K quant. Add a post-RoPE variant for the faithful row |
| Order quant vs prune | quantize concat cache, then prune (gather of already-quantized values) | prune (gather decoded), then quantize | minor: group boundaries differ |
| Real bit-packing | no (fake quant) | yes | ours is stricter (reports real bytes) |
| Pre-RoPE position bookkeeping | not needed (RoPE already applied) | stores per-token positions | correct for the stock-StreamVGGT layout |

### 3.4 StreamCacheVGGT and STAC
- **STAC** (arXiv 2603.20284, CVPR 2026 Highlight; Wang, Song, Cai, Liu, USTC; v1 18 Mar
  2026). Code: https://github.com/Rainzor/STAC (MIT, pushed 2026-06-15), project page
  https://stac-3r.github.io/. Training-free; backbones STream3R and StreamVGGT via
  `--base_model`. Three parts: working temporal token cache (decayed cumulative attention,
  H2O-like, `stac/h2o.py`, `kv_manager.py`), long-term spatial cache that merges evicted
  tokens into voxels (`stac_voxel.py`), chunked multi-frame optimization. Budget (paper):
  first frame fully kept, rest = 8x per-frame tokens split 50% temporal window (s=4),
  25% anchors, 25% retrieved merged tokens; fp16 KV; 518x392 input; one 40 GB A100.
  Headline: "nearly 10x" memory, 4x speed (arXiv v1/v2 abstract); the CVPR listing says
  8.5x / 3.5x (check the camera-ready before citing). NRGBD Tab. 1: STream3R 19.75 GB ->
  2.20 GB, StreamVGGT 19.75 GB -> 2.57 GB. Video depth is in an appendix (Sintel, KITTI,
  ReFusion cited; Bonn not named in the part I read, ~22 kB of the HTML unread).
  No quantization. XStreamVGGT is not compared. This is the strongest *eviction/merge*
  baseline we can run on both our backbones with their released code.
- **StreamCacheVGGT** (arXiv 2604.15237, 16 Apr 2026; Liu, Yu, Ji, et al.). Training-free;
  Cross-Layer Consistency-Enhanced Scoring + three-tier Hybrid Cache Compression (evict /
  merge into anchors by nearest-neighbor on keys / keep). Evaluated on 7-Scenes, NRGBD,
  ETH3D, Bonn, KITTI. **No code link found** (arXiv page; GitHub repo search returned
  nothing). No headline numbers on the abstract page.
- Others in the eviction/selection cluster (all token-level, no bit-width): Evict3R
  (2509.17650), OVGGT (2603.05959), FrameVGGT (2603.07690; frame-wise KV segments
  summarized by key prototypes + sparse anchors; repo `ZhisongXu/FrameVGGT` exists),
  GHOST (2605.15852), RetrieveVGGT (2605.09644, NeurIPS 2026, repo `zzctmd/RetrieveVGGT`),
  HorizonStream (2605.23889).

---------------------------------------------------------------------------------------

## 4. TempoKV (arXiv 2610.15324)

- Title: "TempoKV: Predictive KV Cache Quantization for Autoregressive Video Generation",
  Hu, Mo, Cai, Liu, Ren, Zhang, Tu, Zhang (Linfeng Zhang group). **Submitted 11 Oct 2026
  (v1, 03:59:25 UTC).** So it is ~3 weeks old at the time of writing and ~5 weeks before
  our deadline; check the CVPR 2027 author guidelines on concurrent work **[UNVERIFIED,
  I did not read the policy]**.
- **Code:** the arXiv page says "will be released"; however a repository
  https://github.com/jiahui1021/Tempokv exists (created 2026-08-05, last push
  2026-09-30, no license, README in Chinese, "TempoKV compresses video-generation KV
  caches via temporal prediction and CrossKV prediction; integrates Self-Forcing, Causal
  Forcing, LongCat"). It matches the paper's model list; **whether it is the official
  repo is [UNVERIFIED]** (no arXiv link in the README; the git author is "jiahui").
- **Predictor details (paper HTML + repo, partly different from our notes):**
  - **Closed-loop: confirmed.** Regression target is the current full-precision value;
    predictor inputs are the *quantized-then-reconstructed* cache.
  - K: element-wise (Hadamard) coefficients on the previous **h_K = 2** reconstructed key
    frames (`A (.) K_hat`), per element/dimension; fit per layer and head in the "native"
    protocol, **shared across layers in the "legacy" default** of the repo.
  - V: predicted from the **current reconstructed K through a dense D x D "CrossKV"
    matrix plus 1 previous reconstructed V frame** (h_V = 1). This cross-K->V term is a
    difference from a pure temporal predictor and is the paper's asymmetry finding
    (keys follow temporal history; values follow the current key).
  - Ridge: yes (Eq. 5 L2 penalty; lambda not reported). Calibration: 8 BF16 prompt
    traces by default (ablated 1-32), no backprop.
  - **Anchor: only the first latent frame**, kept for the whole generation, quantized at
    `max(4, b_K, b_V)` bits; no periodic refresh described in the paper. The repo
    exposes an anchor-reset period (`stride`/"span", `ANCHOR_BITS` 1-16).
  - Residual quantizer: RTN affine, written k{N}v{M} (k2v2, k4v2, k5v3, k6v4). Repo
    `config.py` defaults: `group_size=64`, `scale_dtype=bfloat16`,
    `residual_mode="asymmetric"`, residual bits 1-8 selectable. Paper does not state
    group size.
  - The repo also ships **fixed (training-free) predictors** `identity`, `taylor`,
    `taylor2` (`FIXED_TEMPORAL_PREDICTOR_MODES`) and 1-bit residuals are selectable
    (`BITS` 1-8). `identity` + closed loop is essentially our "previous reconstruction"
    delta. **This is the closest prior-art risk**: a reviewer can say TempoKV-identity
    = our codec. Our differentiators must be domain (perception/depth/3D, where the
    output is a dense geometry metric not a video), the keyframe-every-K design,
    sign + 8-bit scale at 1.x bits, no calibration, and measured depth-quality results.
- Models/eval: Self-Forcing-Wan-1.3B, Causal Forcing-Wan-1.3B, LongCat-Video-13B;
  first 128 MovieGen prompts, 832x480, 717 frames; PSNR/SSIM/LPIPS/FID + six VBench-Long
  dimensions. Baselines: QVG, QVG-Pro, RTN, KIVI, QuaRot (Forcing-KV only in a pruning
  comparison, App. A.6). Ratios: k2v2 6.71x/6.71x/6.66x, k4v2 4.73x, k5v3 ~3.65x, k6v4
  2.97x (Self-Forcing/Causal-Forcing/LongCat), headline "up to 6.71x".
- **What to mirror:** (a) report compression ratio with scales/anchor included and name
  configs kNvM-style next to ours; (b) include RTN, KIVI and a rotation baseline
  (QuaRot-style) as quantization comparators, plus an `identity`-predictor closed-loop
  row to pre-empt the obvious reviewer question; (c) K/V asymmetry ablation (they claim
  V depends on the current key; test whether it holds for depth/3D); (d) calibration-free
  is a selling point versus their 8-prompt ridge fit; (e) separate "quality vs
  full-precision" metrics from end-task metrics.

---------------------------------------------------------------------------------------

## 5. Prior art: temporal / delta KV or feature-cache compression for depth or 3D

Searched (arXiv 2025-2026, CVPR/ICCV/ECCV/NeurIPS/ICLR via web search; several
phrasings: temporal delta/residual/predictive coding x KV/feature cache x depth/3D/VGGT,
1-bit sign residual, keyframe quantization). **I found no paper that applies
frame-to-frame predictive (delta) coding to the KV or feature cache of a video-depth or
3D-reconstruction/perception model.** The search is not exhaustive (no full-text
search of every venue); close hits:

| Paper | Venue / ID | Domain | Difference vs ours |
|---|---|---|---|
| **TempoKV** | arXiv 2610.15324, 11 Oct 2026 | AR video **generation** KV | Closest. Closed-loop predictive residual quantization; ridge-fit linear predictors; CrossKV for V; first-frame anchor only; repo has identity/Taylor fixed predictors and 1-8-bit residuals. Not perception, no depth/3D. |
| XStreamVGGT | 2601.01204 | StreamVGGT | Prune + KIVI INT4; **no temporal prediction** (quantizes each token independently). |
| STAC | 2603.20284 (CVPR 2026) | StreamVGGT/STream3R | Token eviction + voxel merging; no quantization, no delta. |
| StreamCacheVGGT | 2604.15237 | VGGT | Cross-layer scoring + merge/evict; no quantization. |
| 3DTurboQuant | 2604.05366 | DUSt3R-L (pairwise, not streaming) | Training-free KV quant, 2->3 bit cliff, 4-bit "lossless"; **per-token independent, no temporal axis**. (Already in `docs/related_work_notes.md`; its PDF text was not fully read there, still to verify manually.) |
| FrameVGGT / Evict3R / OVGGT / GHOST / RetrieveVGGT / HorizonStream | 2603.07690, 2509.17650, 2603.05959, 2605.15852, 2605.09644, 2605.23889 | VGGT-style streaming | Eviction, selection, or prototype summarization; none stores residuals. |
| Quant VideoGen (QVG) | 2602.02958 (ICML 2026) | AR video generation | Semantic-aware smoothing (k-means centroid subtraction over tokens) + progressive residual quantization; residual is w.r.t. cluster centroids, not previous frame. |
| KV-cache quantization for Self-Forcing (33-method study) | 2603.27469 | Video generation | Empirical study; abstract does not mention temporal-delta methods (**[UNVERIFIED]**, full text not read). |
| ResQ: Residual Quantization for Video Perception | ICCV 2023, 2308.09511 | Video perception (segmentation, pose) | Quantizes *activation residuals between frames* with keyframes and adaptive bit-width, for compute savings; activations not KV caches, no depth/3D (abstract names segmentation and pose only). Important to cite; our novelty is KV/feature cache memory for streaming geometry. |
| DeltaCNN | CVPR 2022 | CNN video | Sparse frame-difference compute reuse; not caches or quantization. |
| CacheGen | 2310.07240 | LLM KV streaming | Delta vs anchor token inside a group for KV *bitstream* compression (video-codec inspired); text, not video/depth. |
| ERTACache | ICLR 2026 | Diffusion video | Caches residual to skip compute; generation. |
| InfiniPot-V | 2025 | Streaming video LLM | Temporal-redundancy token removal in fixed memory; understanding. |
| oVDA / VDA streaming | 2510.09182 | Video depth | Caches pre-projection features in an fp16 sliding window; **no compression**. |
| MAMo | 2307.14336 | Video depth | Memory + attention for depth; learned memory, unrelated to quantization. |

Net positioning: temporal predictive residual KV coding exists for **generation**
(TempoKV, concurrent) and for **activations in perception** (ResQ, 2023); streaming-3D
cache compression exists as **eviction/merge** (STAC, StreamCacheVGGT, XStreamVGGT) or
**per-token quantization** (XStreamVGGT, 3DTurboQuant). The combination (closed-loop
temporal delta on the KV/feature cache of streaming depth/3D models with keyframes) is
unclaimed as far as this search can tell. Re-run a search on arXiv listing pages in the
week before submission (TempoKV appeared only 3 weeks ago; the field moves weekly).

---------------------------------------------------------------------------------------

## 6. Suggested experiment grid for the third-model integration

Per model (StreamVGGT, STream3R), datasets Sintel/Bonn/KITTI with per-sequence scale
alignment, report AbsRel, delta1 and real bytes/token (scales + keyframes included):

1. FP32 full cache (reference) and fp16 cache.
2. Plain per-group INT4 / INT3 / INT2 (ours, `plain:*`).
3. KIVI-INT4 g64 (`xstream:inf:4:4`) - quantization only, no pruning.
4. Ours delta, K=8, INT3/INT4 keyframe + 1-bit residual (`delta:*`).
5. XStreamVGGT faithful: budget 2048 tokens, pool 16, INT4 KCVT, **post-RoPE** variant.
6. XStreamVGGT-strong: same but pre-RoPE K (our existing implementation).
7. Window baselines: STream3R-W[5] (in repo) and STAC (MIT repo, both backbones).
8. Optional: XStreamVGGT-prune + our delta (`xstream+delta`) to show orthogonality.
9. Budgets expressed in frames (4, 8, 16 frames) alongside the literal 2048 tokens.
10. TempoKV-style `identity` closed-loop predictor with their quantizer settings as an
    ablation row (can be a re-implementation of the idea inside our code; their repo is
    tied to Wan/Self-Forcing).
