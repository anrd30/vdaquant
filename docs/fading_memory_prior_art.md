# Prior-art check: "fading memory" (age-dependent progressive re-quantization of streaming perception caches)

Search date: 2026-11-02. Method: web search plus primary-source fetches (arXiv abstract/HTML pages, a few PDFs converted to text, GitHub README). Status tags used below:

- [V] = I read the arXiv abstract page, HTML or PDF text (or README) in this session.
- [S] = seen only in search-result snippets / secondary summaries (blogs, aggregators). Treat details as unverified.
- [B] = cited from background knowledge only, not re-fetched this session. Verify before citing.

Quotes are kept under 15 words. Numbers are the authors' own claims, not reproduced.

---

## VERDICT

**Partially anticipated. The ingredients are all published; the specific combination in streaming perception caches is, as far as I could find, not.**

Found no paper (arXiv 2023 to Oct 2026, plus CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR/AAAI/MLSys/ACL listings reached via search) that does all of: (i) per-frame, age-indexed precision ladder (e.g. 4 -> 2 -> 1 bit) (ii) applied by re-quantizing already-stored entries as they age (iii) with a memory bound for unbounded streams (iv) in a streaming 3D / depth / perception model. Streaming-3D cache papers do eviction, merging, retrieval or a single uniform INT4. The LLM literature already has each mechanism in isolation, and one grey-literature package (FADE-KV) plus one workshop paper (LogQuant) come uncomfortably close at the mechanism level.

**Do not claim** "first age-dependent mixed-precision KV cache", "first progressive re-quantization", or "first fading memory". Those are all taken (SKVQ, KVmix, PM-KVQ, LogQuant, FADE-KV, TTKV, FadeMem for the "fade" idea itself).

### The 3 closest works (mechanism level)

1. **PM-KVQ** (arXiv 2505.18610, ICLR 2026) [V]. Starts the cache at 16-bit and, once the memory budget is full, shrinks the *existing* cache 16 -> 8 -> 4 -> 2 by an "Equivalent Right Shift" that is equivalent to dequantize + requantize. That is progressive re-quantization of stored entries under a memory budget. Differences from ours: the trigger is memory occupancy, not token age; the shrink is applied to the whole cache (per transformer block) at once, not as a staggered per-age ladder; first token and last 128 tokens stay 16-bit; LLM long-CoT reasoning only; no error feedback analysis.
2. **FADE-KV** (PyPI `fade-kv`, GitHub Omodaka9375/fade, 2026; software, no paper, not peer reviewed) [V via README fetch]. Five tiers: FP16 for sink + recent window, INT4 for the middle, optional INT2 / product-quantized, then eviction. README says tokens are placed "based on age and attention importance", and tiers are re-assigned periodically (every 64 decode steps in the example). This is the closest thing to an age-tiered precision ladder with a terminal eviction tier. Differences: LLM only, attention-importance policies (h2o/ema/adaptive) mixed in, benchmark on a 0.5B model, no paper, no perception/3D, no analysis of error accumulation across re-quantizations.
3. **LogQuant** (arXiv 2503.19950, ICLR 2025 SLLM workshop) [V HTML]. Keeps at most 3W tokens at 16-bit and everything else at 2-bit; when the buffer fills, every other token in the older span is demoted to 2-bit, producing a log-spaced density of high-precision tokens (denser near the present). That is position/age-dependent precision with one-way demotion. Differences: only two precisions (16 / 2), selection by log-spaced position rather than a multi-step ladder, LLM only, memory still grows (2-bit tokens are never dropped).

### Closest by concept / domain (not by mechanism)

- **FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion** (arXiv 2606.10671) [V]. Dense-near / sparse-far memory under a fixed budget, adjacent older KV blocks merged with a power-law temporal schedule. Same philosophy as ours (graceful aging instead of hard eviction, bounded memory) but it merges tokens/blocks, never lowers bit-width, and is for video generation. **Name collision risk**: avoid the names "FadeMem" and "FADE" for our method (there is also an unrelated agent-memory paper called FadeMem, arXiv 2601.18642, and the FADE-KV package).
- **XStreamVGGT** (arXiv 2602.21780 / 2601.01204; JSID 2026 poster) [V]. The only streaming-3D paper with KV quantization: importance pruning to a fixed budget, then uniform KIVI INT4 (group 64) on what remains. Precision does not depend on frame age. Any reviewer will say "XStreamVGGT already quantizes StreamVGGT's cache", so we must position against it explicitly.
- **Compressive Transformer** (Rae et al., arXiv 1911.05507, ICLR 2020) [V]. Conceptual ancestor, see section 5.

### Exact differentiators we could claim honestly (only if our experiments back them)

1. **Domain.** First age-graded precision schedule for streaming 3D / depth feature caches (StreamVGGT, STream3R, Video-Depth-Anything-style caches), evaluated with geometry metrics (AbsRel, pose ATE, Chamfer/accuracy) rather than LLM perplexity. Nothing found that does this. XStreamVGGT is uniform INT4; STAC / StreamCacheVGGT / OVGGT / Evict3R / GHOST / IncVGGT / InfiniteVGGT / FrameVGGT / RetrieveVGGT are all eviction/merge/retrieve.
2. **Re-quantize instead of evict, as the primary memory-control knob, for perception.** The 3D papers all assume the unit of memory control is "keep the token or drop it". Our claim would be: for geometry caches, lowering bits is a strictly gentler knob than dropping tokens. Support in the LLM literature exists (MiKV, "More Tokens, Lower Precision", ARKV) so we can cite them as motivation, but must show it empirically for geometry (loop closure / long-range correspondence is the natural test).
3. **Per-frame age ladder with a geometric schedule and an explicit memory accounting.** The LLM schemes are per-token with 2 tiers (KIVI, SKVQ, KVmix, LogQuant) or memory-triggered (PM-KVQ). A frame-granular, age-indexed ladder with a closed-form memory bound is not published (see caution 1 below).
4. **Error handling across repeated re-quantization.** PM-KVQ re-quantizes by right shift and explicitly does not analyze accumulated error; QVG says it avoids re-compression drift by quantizing once per chunk; GEAR's residual correction is open-loop on new blocks. A closed-loop / error-feedback re-quantization (or provably lossless nested bit truncation) for cached features is not published in the KV-cache literature I found. Claim only if we actually measure drift vs. number of requantizations.
5. **Joint with temporal delta/residual coding on perception caches.** TempoKV (video generation, arXiv 2610.15324) predicts keys from temporal history and stores low-bit residuals; CacheGen uses cross-token deltas for transport. Combining age-graded precision with delta coding inside a streaming 3D cache is not published. (Both TempoKV and CacheGen are single-precision w.r.t. age, so this is combination novelty only.)

### Cautions that affect how we write the claim

1. **"Bounded memory" is not free.** With integer bits >= 1 and no terminal tier, memory is still O(T) (e.g. a 1-bit floor is 1/16 of FP16 per frame, but grows linearly). With tier widths doubling and bits halving (tier k holds 2^k frames at B/2^k bits) every tier costs the same, so total memory grows as O(log T) only if you can go below 1 bit/element, which requires token merging, pruning, or vector quantization, i.e. not plain scalar re-quantization. A true constant bound needs a terminal evict/merge tier (as FADE-KV, FadeMem, ARKV do). State the bound with the terminal tier included, and do not claim "unbounded stream, bounded memory" from re-quantization alone. PM-KVQ's bound is simply the fixed final bit-width times max context.
2. **Recency is not always the right signal.** AKV (grey literature) and ARKV/MiKV/ZipCache/QAQ argue attention importance beats recency; the video-LLM literature (DyCoke observation relayed via VidKV) says old frames can become relevant again; Dual-Signal/TriAxialKV hint at modality and role axes. Include an age-vs-importance ablation, and note that 3D streaming has a particular reason to keep old frames (global anchors, loop closure; cf. InfiniteVGGT immutable anchor frame set, XStreamVGGT keeps first frame).
3. **Heavy reuse of the word "progressive"**: PM-KVQ (progressive over the cache fill), Quant VideoGen (progressive residual quantization = coarse-to-fine stages, NOT age), Lynx (progressive speculative quantization for KV transfer), ReadKV (progressive/nested code read at query time). Avoid titles that make reviewers conflate these.

---

## 1. LLM KV-cache: mixed precision by recency / age / importance

Columns: (a) age/recency-dependent precision, (b) re-quantizes stored entries over time, (c) bounds memory for unbounded streams, (d) perception/3D.

| Work | ID / venue / date | Link | Mechanism (1-2 lines) | a | b | c | d |
|---|---|---|---|---|---|---|---|
| KIVI [V] | arXiv 2402.02750, ICML 2024, Feb 2024 | https://arxiv.org/abs/2402.02750 | 2-bit asymmetric (K per-channel, V per-token). Newest R=128 tokens kept in full precision; when the residual buffer fills, the oldest group is quantized once and appended. | Yes (2 tiers: recent FP16 vs old 2-bit) | No (quantized once) | No (all tokens kept) | No |
| SKVQ [V] | arXiv 2405.06219, COLM 2024 (venue per acknowledgments, not confirmed), v3 Nov 2024 | https://arxiv.org/abs/2405.06219 | Sliding window of last w=128 tokens in full precision; token leaving the window is quantized (channel reorder + clipped group quant). Argues attention to old tokens decays. Sink tokens kept FP. | Yes (2 tiers) | No (one-shot at window exit) | No | No |
| KVmix [V] | arXiv 2506.08018, AAAI 2026, v3 Feb 2026 | https://arxiv.org/abs/2506.08018 | Layer-importance (gradient) bit allocation plus "Recent Pivotal Context" kept full precision; older / less critical KV compressed. RPC count is a ratio of the current cache, shrinks dynamically. | Partly (recent window FP, older quantized) | Implied, not explicit | No | No |
| PM-KVQ [V] | arXiv 2505.18610, ICLR 2026 | https://arxiv.org/abs/2505.18610 | Progressive bit-width shrink 16->8->4->2 of the existing cache once memory budget is exhausted, via equivalent right shift (dequant+requant). Block-wise (layer) bit allocation; first token and last 128 tokens kept INT16. | Not by age: trigger is memory fill, applied cache-wide (recent window protected) | **Yes** | Yes (fixed final bit-width x max context) | No (long-CoT LLM) |
| LogQuant [V] | arXiv 2503.19950, ICLR 2025 SLLM workshop | https://arxiv.org/abs/2503.19950 | <=3W tokens at 16-bit, rest 2-bit. Periodically every-other token in the older span demoted to 2-bit, giving log-spaced density of high-precision tokens (dense near, sparse far). | **Yes** (position-dependent, log schedule) | **Yes** (16 -> 2, one-way) | No (2-bit tokens kept forever) | No |
| FADE-KV [V README only] | PyPI fade-kv / GitHub Omodaka9375/fade, 2026, no paper | https://github.com/Omodaka9375/fade | Tiers FP16 (sink + recent) / INT4 (middle) / optional INT2 or PQ / evicted; policies h2o, ema, adaptive, position, learned; tier re-assignment every N steps. Self-reported tests on a 0.5B model. | **Yes** (age + attention) | **Yes** | Yes via eviction tier | No |
| AKV [S] | PyPI `akv-cache` 1.0.0, 2026, no paper | https://pypi.org/project/akv-cache/ | Hot (FP16 GPU) / warm (INT4) / cold (INT2, CPU) tiers; claims importance-aware migration beats FIFO demotion. Page failed to load; details from search snippet only. | Partly (FIFO baseline is age; main method is importance) | Yes (tier migration) | Cold tier unbounded on CPU | No |
| TTKV [V HTML] | arXiv 2604.19769, IJCAI 2026 template (acceptance unconfirmed), Apr 2026 | https://arxiv.org/abs/2604.19769 | Two temporal tiers: recent KV in HBM at FP16; older blocks (128 tokens) demoted FIFO to DRAM with keys 8-bit / values 4-bit; top-k block retrieval. Single compression step at eviction. | **Yes** (recent = fast/high precision) | No (one demotion step, no further re-quantization) | Fast tier bounded; slow tier grows | No |
| TriAxialKV [V HTML] | arXiv 2605.17170, May 2026 | https://arxiv.org/abs/2605.17170 | Static per-tag INT2/INT4 map calibrated offline; tags include temporal recency (`older`, `turn_m2`, `turn_m1`, `current`), modality, role. Agentic VLM inference. | Partly (recency is one tag axis; assignment from calibration, not a schedule) | No (no re-quantization as token ages) | Budgeted average bits | Partly (image tokens, computer-use agent) |
| ARKV [V PDF] | arXiv 2603.08727, Feb 2026 | https://arxiv.org/abs/2603.08727 | Tri-state per token (Original / Quantized / Evicted) by heavy-hitter score under a strict budget; last W tokens protected at full precision; layer-wise OQ ratio. | Window protected; main signal is importance | Decode-time re-ranking | Yes (budget) | No |
| MiKV [V] | arXiv 2402.18096, Feb 2024 | https://arxiv.org/abs/2402.18096 | Keeps would-be-evicted KV in low precision, important KV in high precision. Importance policy not stated on abstract page. | No (importance) | Not stated | Not stated | No |
| "More Tokens, Lower Precision" [V abs] | arXiv 2412.12706, Dec 2024 | https://arxiv.org/abs/2412.12706 | Study: under a fixed budget, more tokens at lower precision beats fewer at 16-bit (quantized pruning). Useful motivation for re-quantize-not-evict. | No | No | Budgeted | No |
| Cocktail [S] | arXiv 2503.23294, Mar 2025 | https://arxiv.org/abs/2503.23294 | Chunk-level bit-width from query-chunk similarity; reorders chunks by bit-width. | No (query similarity) | No | No | No |
| ZipCache [V abs] | arXiv 2405.14256, NeurIPS 2024 (per my background; abs page lists none), May 2024 | https://arxiv.org/abs/2405.14256 | Saliency via normalized attention score; salient tokens higher precision. | No | No | No | No |
| QAQ [V abs] | arXiv 2403.04643, Mar 2024 | https://arxiv.org/abs/2403.04643 | Attention-aware quantization with dedicated outlier handling; separate K/V strategies. | No | No | No | No |
| WKVQuant [V abs] | arXiv 2402.12065, Feb 2024 | https://arxiv.org/abs/2402.12065 | PTQ of weights + KV ("past-only quantization" refers to attention computation, not token age). | No | No | No | No |
| KVQuant [V abs] | arXiv 2401.18079, NeurIPS 2024 | https://arxiv.org/abs/2401.18079 | Per-channel / pre-RoPE key quant, non-uniform datatypes, dense-and-sparse outliers. | No | No | No | No |
| KVTuner [V abs] | arXiv 2502.04420, ICML 2025 | https://arxiv.org/abs/2502.04420 | Offline sensitivity-aware layer-wise mixed precision (K/V pairs). | No (layer axis) | No | No | No |
| DiffKV [S] | arXiv 2412.03131, SOSP 2025 | https://arxiv.org/abs/2412.03131 | Differentiated K vs V precision, token-importance precision, per-head budgets, paged memory manager. | No (importance) | Per-request adaptation | No | No |
| MiniCache, PyramidKV, PyramidInfer [B] | arXiv 2405.14366 (NeurIPS 2024), 2406.02069, 2405.12532 | n/a | Layer-wise: cross-layer KV merging; pyramid layer budgets. Not token-age or precision schedules. | No | No | Budgeted | No |
| ReadKV [V abs] | arXiv 2610.11245, Oct 2026 | https://arxiv.org/abs/2610.11245 | Stored entries in a progressive (nested) code; precision chosen per query at read time ("Stored entries remain unchanged"). | No (query-adaptive) | No | No | No |
| Lynx [S] | arXiv 2607.01831, Jul 2026 | https://arxiv.org/abs/2607.01831 | Split KV into MSB anchor stream and residual stream for KV transfer in disaggregated serving. | No | No | n/a | No |
| Matryoshka Quantization [S] | arXiv 2502.06786, ICML 2025 | https://arxiv.org/abs/2502.06786 | Nested int8/int4/int2 for model weights. Relevant technique: truncating MSBs gives lower precision without a second quantization pass. Not KV. | No | n/a | n/a | No |
| Technical Disclosure 11299 [V] | tdcommons.org DPS 11299, Aug 7 2026 | https://www.tdcommons.org/dpubs_series/11299 | Defensive publication: per-token attention decay-rate quantification, decay-class-driven KV segment compression, tier residency. Abstract does not mention bit-width. | Decay of attention, not age schedule | n/a | n/a | No |

Takeaway: (a) is done in 2-tier form (KIVI/SKVQ/KVmix/TTKV), in log form (LogQuant), and in 3-tier form in grey literature (FADE-KV, AKV). (b) is done by PM-KVQ (memory-triggered, whole-cache), LogQuant and FADE-KV. (c) is only solved with a terminal evict tier or a fixed final bit-width.

## 2. Streaming video understanding (video-LLM) caches

| Work | ID / venue / date | Link | Mechanism | a | b | c | d |
|---|---|---|---|---|---|---|---|
| StreamingTOM [V] | arXiv 2510.18269, CVPR 2026 | https://arxiv.org/abs/2510.18269 | Causal Temporal Reduction (fixed per-frame token budget) + Online Quantized Memory: each incoming group stored once at 4-bit, no revisiting of historical groups; top-k group retrieval + dequantization. | No (uniform 4-bit at write time) | No (explicitly "without revisiting historical groups") | Active KV bounded by retrieval; stored KV still grows O(T G d)/4 | Video-LLM |
| HERMES [V abs + S] | arXiv 2601.14724, ACL 2026 Main | https://arxiv.org/abs/2601.14724 | KV cache as hierarchical memory; shallow layers use an exponential forgetting curve over temporal distance for token importance; evicted tokens folded into summary tokens. Secondary summaries disagree on tier compression ratios (8x/4x vs 16-32x): unverified. No quantization found. | Age-dependent token importance, not precision | Merge into summary | Yes (fixed budget) | Video-LLM |
| MuKV [V abs] | arXiv 2605.22269, CVPR 2026 | https://arxiv.org/abs/2605.22269 | Patch / frame / segment granularity caches, self-attention + frequency dual-signal compression, semi-hierarchical retrieval. Abstract does not say older = coarser; no quantization. | Not stated | No | Not stated | Video-LLM |
| StreamKV [V abs] | arXiv 2511.07278, Nov 2025 | https://arxiv.org/abs/2511.07278 | Segment-based KV retrieval and compression guided by a prompt; no quantization. | No | No | Yes-ish | Video-LLM |
| StreamMem [S] | arXiv 2508.15717, Meta/NYU, Aug 2025 | https://arxiv.org/abs/2508.15717 | Query-agnostic: frame filtering, chat-template tokens as proxy queries, prototype merging; compresses after each clip, bounded memory. No quantization in summaries. | No | Merge | Yes | Video-LLM |
| LiveVLM [S] | arXiv 2505.15269, DAC 2026 | https://arxiv.org/abs/2505.15269 | Streaming-oriented KV: Vision Sink Bucketing (vision-to-vision attention) + retrieval. | No | No | Partly | Video-LLM |
| InfiniPot-V [S] | arXiv 2506.15745, NeurIPS 2025 | https://arxiv.org/abs/2506.15745 | Memory-constrained: at a threshold, drop temporally redundant tokens, keep by value-norm. | No | Re-compress on threshold (eviction) | Yes | Video-LLM |
| ReKV [S] | arXiv 2503.00540, ICLR 2025 | https://arxiv.org/abs/2503.00540 | Sliding-window attention, full KV offloaded to RAM/disk, retrieval of relevant parts. | No | No | No (offload) | Video-LLM |
| Flash-VStream [S] | arXiv 2506.23825 (extended), ICCV 2025 | https://arxiv.org/abs/2506.23825 | Fixed-size memory of Context Synopsis + Detail Augmentation memories. | No | No | Yes | Video-LLM |
| StreamingVLM, Dispider, VideoLLM-online [B] | arXiv 2510.09608; 2501.03218; 2406.11816 | n/a | Attention-sink + window with contiguous RoPE; decoupled perception/decision/reaction; streaming narration. No age-dependent precision known to me. | No | No | Window | Video-LLM |
| VidKV [S] | arXiv 2503.16257, Mar 2025 | https://arxiv.org/abs/2503.16257 | 1.x-bit KV quantization for video-LLMs; mixed precision by channel (K) and salient tokens (V). Importance-based, not age. | No | No | No | Video-LLM |
| WindowQuant [V abs] | arXiv 2605.02262, ACM TACO | https://arxiv.org/abs/2605.02262 | Window-level mixed-precision by similarity between visual-token windows and text prompt. | No | No | No | VLM |

Takeaway: no streaming video-LLM cache uses age-dependent precision. The Ebbinghaus-style "forgetting curve" in HERMES is the same philosophical idea applied to token importance and token merging, not to bit-width; it should be cited as the closest video-LLM analogue of "fading".

## 3. Streaming 3D / video depth caches

| Work | ID / venue / date | Link | Mechanism | a | b | c | d |
|---|---|---|---|---|---|---|---|
| XStreamVGGT [V] | arXiv 2602.21780 (also 2601.01204), v1 25 Feb 2026; JSID 2026 poster | https://arxiv.org/abs/2602.21780 | Per-layer importance pruning to a 2K-token budget (first and current frame always kept), then KIVI-style INT4 (K per-channel, V per-token, group 64). Reports 4.42x memory, 5.48x speed. Efficiency tests only to 1,000 frames. | No (uniform on pruned cache) | No | Yes (budget) | Yes |
| STAC [V abs] | arXiv 2603.20284, CVPR 2026 | https://arxiv.org/abs/2603.20284 | Decayed-cumulative-attention working token cache + voxel-merged long-term spatial cache + chunked multi-frame optimization. Note "decayed" refers to attention scores, not precision. | No | Merge | Not clearly strict | Yes |
| StreamCacheVGGT [V abs] | arXiv 2604.15237, Apr 2026 | https://arxiv.org/abs/2604.15237 | Cross-layer consistency scoring + three-tier triage; mid-importance tokens merged into anchors by nearest-neighbor on key manifold. | No | Merge | Constant budget | Yes |
| OVGGT [V abs] | arXiv 2603.05959, Mar 2026 (v3 Apr 2026) | https://arxiv.org/abs/2603.05959 | Self-Selective Caching via FFN residual magnitude + Dynamic Anchor Protection; FlashAttention compatible. | No | Evict | Yes (O(1)) | Yes |
| Evict3R [V abs] | arXiv 2509.17650, Sep 2025 | https://arxiv.org/abs/2509.17650 | Training-free per-layer budgeted token eviction for StreamVGGT. | No | Evict | Yes | Yes |
| GHOST [V abs] | arXiv 2605.15852, May 2026 | https://arxiv.org/abs/2605.15852 | Geometry-hierarchical online token eviction on StreamVGGT, privilege mechanism, layer-wise budget by cosine similarity. | No | Evict | Yes | Yes |
| IncVGGT [S] | ICLR 2026 poster (arXiv id not confirmed) | https://iclr.cc/virtual/2026/poster/10010837 | Merge overlapping frames into composite views; fixed top-k slots + latest frame (S=k+1). Runs 10k frames on 80 GB per authors. | No | Evict | Yes | Yes |
| InfiniteVGGT [S] | arXiv 2601.02281, Jan 2026 | https://arxiv.org/abs/2601.02281 | Bounded rolling memory, immutable first-frame anchors, diversity-scored per-layer/head pruning. | No | Evict | Yes | Yes |
| FrameVGGT [S] | arXiv 2603.07690, Mar 2026 | https://arxiv.org/abs/2603.07690 | Frame-level rolling explicit memory: each frame's KV contribution summarized as a prototype, fixed-capacity mid-term bank, optional anchor tier. | No (frame-grained tiers, prototypes) | Summarize | Yes | Yes |
| RetrieveVGGT [S] | arXiv 2605.09644, May 2026 (NeurIPS 2026 listing) | https://arxiv.org/abs/2605.09644 | Retrieve a fixed number of relevant frames per step by first-layer QK similarity + pose-aware spatial memory; constant memory. | No | No | Yes | Yes |
| StreamVGGT [B], STream3R [S] | StreamVGGT arXiv 2507.11539 (from memory); STream3R arXiv 2508.10893, ICLR 2026 | https://arxiv.org/abs/2508.10893 | Causal-attention variants of VGGT with full past-frame KV cache (STream3R also offers sliding-window mode). The unbounded-cache baselines everyone compresses. | No | No | No | Yes |
| CUT3R [B], TTT3R [S], Point3R [S], FILT3R [S] | arXiv 2501.12387; 2509.26645; 2507.02863; 2603.18493 | https://arxiv.org/abs/2509.26645 | Fixed-size recurrent state / explicit pointer memory (Point3R) / Kalman-filtered latent state (FILT3R). These are learned or gated state-update memories (forgetting by gating), not per-frame caches; no bit-width control. | Implicit forgetting via state update | Yes (state overwritten) | Yes | Yes |
| QuantVGGT [S], QVGGT [S] | arXiv 2509.21302; arXiv 2605.31124 | https://arxiv.org/abs/2509.21302 | PTQ of VGGT weights/activations (W4A4 etc.). Model quantization, not cache aging. Worth citing as "quantization in 3D foundation models". | No | No | No | Yes |

Takeaway: confirmed for 12 streaming-3D papers: none ages precision; XStreamVGGT is the only one quantizing the cache, and uniformly. I did not find any quantized cache work for Video-Depth-Anything / oVDA (oVDA uses a fixed-size temporal cache of 8 or 16 frames per the Voxel51 docs [S]).

## 4. Video generation caches

| Work | ID / venue / date | Link | Mechanism | a | b | c | d |
|---|---|---|---|---|---|---|---|
| Quant VideoGen (QVG) [V HTML] | arXiv 2602.02958, v5 May 2026 (Xi ... Stoica, Han, Keutzer; ICML keyword on page, acceptance not stated) | https://arxiv.org/abs/2602.02958 | Semantic-Aware Smoothing (k-means centroid subtraction) + Progressive Residual Quantization (multi-stage residual refinement). "Progressive" = coarse-to-fine stages, not age. Quantizes once per chunk, "avoids re-compression drift". | No | No (explicitly avoids) | Constant bit-width (INT2/INT4) | Video gen |
| QuantWM [V abs] | arXiv 2609.26425, Sep 2026 | https://arxiv.org/abs/2609.26425 | 2-bit KV for video world models: sensitivity-aware clustering of key centroids + principal-subspace attention compensation. Uniform 2-bit. | No | No | No | Video gen |
| TempoKV [V abs] | arXiv 2610.15324, Oct 2026 | https://arxiv.org/abs/2610.15324 | Predict keys from temporal history, values from current key + short history; store low-bit residuals; predictors fit once offline. Up to 6.71x. | No | No | Not stated | Video gen |
| Forcing-KV [V abs] | arXiv 2605.09681, May 2026 | https://arxiv.org/abs/2605.09681 | Static-head structured pruning + dynamic-head similarity pruning; no quantization. | No | Prune | Partly | Video gen |
| TempCache [S] | arXiv 2602.01801, ICML 2026 | https://arxiv.org/abs/2602.01801 | Merge near-duplicate keys across frames (ANN); bounds cache growth. | No | Merge | Yes | Video gen |
| FadeMem (video) [V HTML] | arXiv 2606.10671, Jun 2026 | https://arxiv.org/abs/2606.10671 | Distance-aware consolidation: adjacent older KV blocks merged by span-weighted average, power-law temporal warp u(d)=d^0.3, M=12 entries; newest and first protected. Never mentions quantization. | **Yes (distance), but resolution/merge not precision** | Yes (repeated merging) | Yes | Video gen |
| DeCoPrune, Focused Forcing, VideoMLA [S] | arXiv 2609.39096; 2605.18346; 2605.30351 | n/a | Position remapping/pruning, per-frame selection, low-rank latent KV cache. No age-graded precision in snippets. | No | No | Partly | Video gen |
| 33-method study [V abs] | arXiv 2603.27469, Mar 2026 | https://arxiv.org/abs/2603.27469 | Benchmarks 33 KV quantization / cache-policy variants on Self-Forcing (incl. PRQ_INT4, QuaRot, soft-prune INT4). Abstract does not mention age-tiered precision. | Not stated | Not stated | n/a | Video gen |

Takeaway: video-gen KV quantization (QVG, QuantWM, TempoKV) is uniform in bit-width over frame age. The only age-graded memory in that area is FadeMem (merge-based). Our idea would be the precision-based counterpart.

## 5. Classic fading / hierarchical memory (and how Compressive Transformer differs)

**Compressive Transformer** (Rae, Potapenko, Jayakumar, Lillicrap; arXiv 1911.05507, ICLR 2020) [V via ar5iv]. Per layer, a FIFO "memory" of n_m recent activations plus a second FIFO "compressed memory" of n_cm slots. When the window advances, the n_s oldest memories are not discarded; a compression function f_c maps them to floor(n_s/c) compressed vectors (c = compression rate) that enter the compressed FIFO. f_c options: max/mean pooling, 1D conv, dilated conv, "most-used" (keep by average attention). The conv compressors are trained with an auxiliary loss (attention-reconstruction worked best; auto-encoding also tried) with gradients stopped from flowing into the main network. Temporal range is roughly l x (n_m + c x n_cm). Algorithm 1 keeps only the last n_cm compressed entries, so the oldest compressed memories are still discarded: compression delays forgetting, it does not make memory unbounded. [For the quotable idea: its key principle is to compress old memories instead of discarding them.]

How ours differs, precisely:
- **What is reduced.** CT reduces the *number of slots* (time axis) by factor c via a learned/pooling map, at full numerical precision. Fading memory reduces *bits per element* of every stored slot, keeps all slots (until a terminal tier), and needs no trained compressor.
- **Training.** CT needs a trained compressor plus an auxiliary loss. Ours is training-free post-hoc quantization.
- **Schedule.** CT has exactly two tiers (memory, compressed memory) with a single rate c. Ours is a multi-step ladder by age.
- **Information semantics.** CT mixes neighbouring time steps (pooling/conv), which blurs temporal detail but keeps full-precision values; ours keeps temporal resolution (per-frame tokens remain individually addressable) but coarsens amplitude. The two are complementary, and a combined scheme (merge then quantize, as FADE-KV/ARKV/StreamingTOM-style hybrids hint) is plausible.
- **Domain.** CT: language modelling, speech, RL memory. Not a streaming perception cache.

Other ancestors (all slot-count or state-based, none bit-precision-based):
- **Infini-attention** (Munkhdalai et al., arXiv 2404.07143) [V abs]: folds old KV into a fixed-size linear-attention compressive memory per head; bounded memory, lossy associative state, no per-token addressability.
- **Memformer** (arXiv 2010.06891) [S]: fixed-size external memory with read/write and a forgetting mechanism.
- **Recurrent Memory Transformer** (arXiv 2207.06881, NeurIPS 2022) [B/S]: memory tokens passed between segments.
- **FadeMem (agent memory)** (arXiv 2601.18642) [S]: Ebbinghaus-style exponential decay with dual-layer memory and fusion for LLM agents; text-memory level, not KV precision.
- **HERMES** (above): Ebbinghaus-style forgetting curve over temporal distance for KV token importance in streaming video.
- **CacheGen** (arXiv 2310.07240, SIGCOMM 2024) [V abs]: KV streaming with adaptive compression levels across parts of the cache; abstract page does not confirm delta encoding or layer-wise levels (my earlier snippet said it does; check full paper before citing that detail).

Suggested citation sentence (honest framing): "Compressive Transformer compresses old memories into fewer slots; we instead keep every slot but age its precision."

---

## Items I could not verify / open ends

- AKV (`akv-cache`) and FADE-KV are PyPI/GitHub packages with self-reported results; PyPI pages failed to load for fetches, so AKV's demotion rule is from search snippets only.
- HERMES' exact tier compression ratios (two blogs disagree) and whether any quantization appears in the full text: full PDF exceeded the fetch size limit. Abstract-level only.
- IncVGGT arXiv id not confirmed (ICLR 2026 poster page only). InfiniteVGGT, FrameVGGT, RetrieveVGGT, LiveVLM, InfiniPot-V, ReKV, Flash-VStream, Cocktail, DiffKV, Lynx, TempCache are snippet-level [S].
- StreamVGGT (2507.11539), CUT3R (2501.12387), StreamingVLM, Dispider, VideoLLM-online, MiniCache, PyramidKV, PyramidInfer, RMT ids are from memory [B].
- XStreamVGGT appears under two arXiv ids (2601.01204 and 2602.21780); cite one and check which is canonical.
- HorizonStream was mentioned in one summary as constant-memory streaming 3D; no source found.
- Did not find "RetrieveVGGT/GHOST precision decay" claims in either abstract; full texts not read.
- Search tool is US-only and summary-based: absence of a hit is not proof of absence. Before submission, re-run on arXiv listing search for: "age-aware KV quantization", "recency bit-width", "frame-age precision", "tiered precision streaming VGGT", and check CVPR 2026 / ICLR 2027 submissions as they appear.
