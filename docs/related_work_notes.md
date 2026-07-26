# Related Work — FULL literature survey (DG-4 round 2, 2026-07-26)

Supersedes the 2026-07-25 first pass. Round 1 found the *method* novelty dead.
Round 2 went after the two SURVIVING claims — (a) "first KV-quant for a dense-
geometry/video-depth model", (b) "temporal-consistency metrics are gameable" —
and found **prior art against BOTH**. A narrower, still-defensible contribution
survives. Read this before writing a single line of the paper.

Threat: HIGH = claims something we wanted to claim. MED = adjacent. LOW = cite-only.

---

## KILL 1 — "First KV-cache quantization of a dense-geometry model" is DEAD

**3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction
Models** (arXiv 2604.05366) — **HIGH**.

Per search-result summary (⚠ full PDF text extraction FAILED — details below are
from a search snippet and MUST be verified by a manual read before submission):
  * Training-free **KV-cache quantization** of **DUSt3R ViT-Large** (a dense-geometry
    vision transformer).
  * Reports a **phase transition between 2 and 3 bits** (pointmap PSNR 16.5 dB @ b=2
    → 29.3 dB @ b=3) — i.e. **the same cliff as our F13**.
  * At **b=4, 7.9× KV compression, "depth structure indistinguishable from the
    unquantized baseline"** — i.e. **the same headline as our F23** (4.0 eff bits, 8×,
    lossless).

**This paper is already cited in our own `research/models/rotated_attention.py`
docstring as reference [3].** It was in the repo from the start and neither DG-4
round 1 nor any earlier session positioned against it. That is a process failure
worth naming.

Surviving distinction (NARROW, must be verified): DUSt3R is **multi-view / pairwise**
3D reconstruction — its attention is spatial cross-view, not **temporal attention
across video frames**. Our work quantizes a *video* model's temporal KV cache and
evaluates *temporal consistency*, which 3DTurboQuant (static reconstruction) cannot.
But **"first KV quant for dense geometry" can no longer be claimed in any form.**

### Adjacent cluster — streaming-3D cache compression (all EVICTION, not quantization)
| Paper | arXiv | Threat | Note |
|---|---|---|---|
| STAC: Spatio-Temporal Aware Cache Compression | 2603.20284 | MED | "First systematic study of training-free spatio-temporal **KV cache compression** for causal transformer 3D recon" (VGGT). But it is **token caching/eviction + voxel consolidation**, not bit-width quantization. ~10× memory, 4× speedup. |
| XStreamVGGT | 2601.01204 | MED | Confirmed by fetch: **token eviction/pruning, NOT quantization**. VGGT streaming 3D recon; evaluates AbsRel/δ₁. |
| GHOST: token eviction for 3D recon | 2605.15852 | LOW | Eviction. |
| StreamCacheVGGT | 2604.15237 | LOW | Hybrid cache compression, VGGT. |

**Defensible axis:** eviction/pruning (which tokens to keep) and quantization (how
many bits per value) are orthogonal compression axes. This cluster is eviction;
3DTurboQuant is the quantization one, and it is the real threat.

---

## KILL 2 — "Temporal-consistency metrics are gameable" is ALREADY KNOWN (for other metrics)

**The degenerate-metric phenomenon is documented prior art in video depth.**

* **OPW** (optical-flow warping error) and **RTC** (relative temporal consistency)
  "are optimized by the degenerate case where depth is constant for all frames
  (dᵗ = k for any constant k)". Documented in *Temporally Consistent Online Depth
  Estimation Using Point-Based Fusion* (arXiv 2304.07435), which proposes the
  **TCC (temporal change consistency)** metric precisely to *prevent that degenerate
  solution*. ⚠ Full-text fetch failed (corrupt PDF); statement is from search
  summary and must be verified by manual read.
* The DAVIS temporal-consistency metric was **deprecated** for exactly this class of
  problem (limited applicability under deformation/occlusion).
* **VMAF Paradox** (arXiv 2605.18378, verified by fetch): codec encoders "can
  effectively **cheat** spatial metrics by sacrificing temporal efficiency" — I-frame
  insertion produces "a sequence of high-quality still images" scoring near-perfect
  VMAF with the worst temporal distortion. They *identify* the problem and explicitly
  **do not propose a corrected metric**.
* Goodhart's-law framing and "compression can improve a metric" (Occam's Hill;
  ~40% of W8A8 runs improved calibration, arXiv 2509.21173) are both established.

**We CANNOT claim "we discovered temporal metrics can be gamed."** A reviewer on an
Evaluation track will cite OPW/RTC-vs-TCC immediately.

### What still survives here — and it is real
1. **The known degeneracy is a *different metric* and a *different mechanism*.**
   Known: *flow-warping* metrics (OPW/RTC) are optimized by **constant depth**.
   Ours: the **geometric reprojection** metric **TAE** — adopted as the successor
   precisely because flow-based metrics were unreliable — is *also* gameable, not by
   constant depth but by **compression-induced loss of spatial structure** (depth still
   varies; it loses fine detail, so there is less to misalign). Same disease, new host,
   new vector.
2. **VDA's TAE has NO occlusion or co-visibility handling — verified by direct fetch
   of the VDA paper (arXiv 2501.12375v2):** "The paper does not discuss occlusion or
   co-visibility masking in relation to TAE. The TAE formula (Eq. 5) … contains no
   mention of occlusion handling or visibility constraints." Our **S6 co-visibility
   mask is a genuine, unclaimed addition to the TAE protocol**, and our data shows it
   FLIPS the ranking (F25) on two model scales.
3. Nobody has demonstrated metric-gaming **under model quantization** for video depth
   (the codec paper is codecs; the LLM work is perplexity/calibration).

---

## Bucket A — lattice / VQ on KV cache (from round 1) — HEAVILY CLAIMED
Leech Lattice VQ (2603.11021, beats QuIP#/QTIP/PVQ) · Grouped Lattice VQ (2510.20984) ·
FibQuant random-access VQ KV (2605.11478) · Hurwitz quaternion VQ KV (2605.27646) ·
GSRQ sub-1-bit gain-shape residual (2607.01065) · VecInfer outlier-suppressed VQ
(2510.06175) · RDKV joint eviction+quantization rate-distortion bit allocation
(2605.08317) · Spherical KV (2605.18856).
**E8-on-KV is not novel; SOTA is past E8 (Leech).**

## Bucket B — video-generation KV quant — CLAIMED, ICML-2026-active
Quant VideoGen (2602.02958, ICML 2026; training-free residual KV quant, 2–4 bit, 7×) ·
33-method empirical study (2603.27469) · Quantized Keys Steal Attention (2605.26266,
QJL-style bias correction for video diffusion) · Forcing-KV hybrid (2605.09681).
**Note (verified by fetch):** the 33-method study does **NOT** demonstrate metric
gaming ("does not demonstrate metric gaming in the sense you describe"), does **not**
use geometric reprojection or depth, and explicitly lacks geometric consistency checks.
It is therefore a *weaker* threat than round 1 feared — cite it, don't fear it.

## Bucket C — rotation / outlier suppression — CLAIMED
QuaRot, SpinQuant (rotation incl. KV cache, 3–4 bit for long video per 2026 survey),
DuQuant. Our F20 (rotation inert at 4 bits on video-ViT temporal cache, essential at
2 bits, with kurtosis mechanism) is a *negative/characterization* result against this
line — still reportable, not a novelty claim.

## Bucket D — temporal-redundancy KV compression — CLAIMED
AttentionPredictor (NeurIPS 2025) · PureKV (2510.25600) · GUI-KV · plus the
streaming-3D eviction cluster above.

---

## FINAL VERDICT (DG-4 round 2)

**DEAD — do not claim any of these:**
- Novel quantization method (lattice / rotation / QJL / residual). [round 1]
- "First KV-cache quantization of a dense-geometry model." → **3DTurboQuant**.
- "Temporal-consistency metrics can be gamed" as a new discovery. → **OPW/RTC + TCC**.
- "First honest/all-inclusive KV-quant benchmark." → 33-method study says
  "nominal compression alone is insufficient."

**SURVIVES — the honest paper:**
1. **The geometric-reprojection metric (TAE) is itself gameable under model
   compression**, by structure collapse rather than the known constant-depth
   degeneracy — the successor metric inherits the disease it was meant to cure.
2. **A co-visibility-masked TAE protocol that fixes it** — VDA's TAE provably has no
   occlusion handling (verified); our mask flips the ranking correctly and is
   monotonic in bit-rate, on **both vits and vitl** (F25).
3. **Video-*temporal* KV quantization** (vs 3DTurboQuant's static multi-view) with
   full-split CIs, a fair grouped-scalar baseline, and the dataset-dependent
   lattice crossover (F24) — as *characterization*, not as a method win.

**Venue implication:** this is a **protocol/evaluation paper**, not a method paper —
which is precisely WACV 2027's *Evaluation & Datasets* track ("analysis of benchmark
failure modes", "new evaluation protocols", "negative results and critical analyses").
The reframing is survivable *there* and would be fatal at a method track.

**Title direction (honest):** "Co-Visibility-Masked Temporal Alignment Error:
Auditing Temporal-Consistency Evaluation for Compressed Video Depth Models."

---

## MANDATORY before submission (unverified claims above)
1. **Read 3DTurboQuant (2604.05366) in full.** PDF extraction failed. Confirm: does it
   touch any *video* model or temporal attention? If it does, the domain claim narrows
   further and the paper must lead purely on the metric contribution.
2. **Read 2304.07435 in full** (OPW/RTC degeneracy + TCC). PDF extraction failed.
   Get the exact sentence and cite it as the direct antecedent of our finding.
3. Read Leech LVQ (2603.11021) and STAC (2603.20284) enough to cite correctly.
4. Verify whether any TAE variant anywhere already applies occlusion masking.
