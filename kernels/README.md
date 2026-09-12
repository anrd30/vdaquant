# kernels/ — Real int-packed KV cache and fused decode for VDA

Everything under this directory is the systems work that turns Paper 2's
**analytic** memory savings into **measured** memory savings. The Python
quantiser under `research/quantizers/` is a *simulator* — it does the
lattice math correctly and stores the reconstructed fp16 back into the KV
cache, so accuracy numbers are honest but there is zero memory reduction
on the GPU. This directory swaps that fp16 storage for real packed
integer storage plus a decode/attention path that understands the packed
format.

## Structure

```
kernels/
├── README.md                   this file
├── reference/                  pure-PyTorch Option-C step-1 reference
│   ├── bw16_codebook.py        the 32-coset BW16 codebook as a LUT
│   ├── packed_bw16.py          packed uint8 storage + PyTorch decode
│   └── packed_e8.py            packed uint8 storage for E8 (Paper 2 comparator)
├── tests/                      correctness gates
│   ├── test_bw16_roundtrip.py  packed encode -> unpack matches simulator
│   ├── test_e8_roundtrip.py    same for E8
│   └── test_full_model.py      end-to-end delta1 matches simulator (ViT-S)
└── triton/                     Triton kernels (added week 2)
    └── (to be created)
```

## Hardware plan

| Device | Where | Purpose | Time on it |
|---|---|---|---|
| **RTX 4050 6 GB** | local | kernel writing, unit tests, correctness on ViT-S/B | ~90 % |
| **Colab T4 free** | cloud | secondary correctness check on ViT-L (16 GB) | ~5 % |
| **Colab A100 paid** | cloud | final benchmarks (latency + peak memory quotes) | ~2 % |
| **University A100** | remote | full-model verification if intermittent access | ~2 % |
| **Jetson Orin** (if available) | edge | deployment demo for §5.9 bonus figure | ~1 % |

VDA memory footprint on inference — the 4050 can hold all three
backbones at batch=1 with room for kernel dev:

| Backbone | Weights (fp16) | KV cache T=32 | Working RAM |
|---|---|---|---|
| ViT-S | 56 MB | 200 MB | ~500 MB |
| ViT-B | 198 MB | 400 MB | ~1.5 GB |
| ViT-L | 764 MB | 840 MB | ~3-4 GB |

## Timeline and milestone gate

Target CVPR 2027 submission: **~2 months from 2026-09-12**.

- **Week 1** (now) — pure-PyTorch packed BW16 reference (`reference/`).
  Deliverable: packed storage that round-trips the simulator within 1e-4,
  gives real memory reduction when swapped into VDA temporal attention.
- **Week 2** — Triton decode kernel (`triton/`). Replaces the PyTorch
  unpack with a fused Triton kernel. Same interface, same tests.
- **Week 3** — fuse decode into FlashAttention v2 template. This is the
  headline: packed int → attention output, never materialise fp16 KV.
- **Week 4** — full-model integration and correctness sweep on ViT-S NYU.
  δ₁ must match simulator within 0.001.
- **Week 5** — benchmarks on ViT-L. This gives the "measured 1.4×
  speedup" number for Paper 2 Table 6.
- **Week 6 — MILESTONE (3 weeks before submission).**
  If full-model correctness on ViT-L NYU / KITTI / Sintel is passing, we
  keep going and add the Jetson deployment demo in Week 7. If it is not,
  we **pivot**: the Week-1 pure-PyTorch packed storage works fine, we
  ship it as the "measured memory" number for the paper, drop the fused-
  attention claim to future work, and take the extra time for the paper.

## Correctness gates (all four must pass before we quote a number)

1. **Roundtrip identity**: encode with the CUDA path, decode, subtract
   from simulator output. `abs(diff).max() < 1e-4`.
2. **Attention parity**: run one temporal-attention forward on random
   inputs, compare CUDA vs simulator outputs. `abs(diff).max() < 1e-3`.
3. **End-to-end δ₁**: full VDA forward on NYU eval, CUDA vs simulator.
   `abs(delta1_cuda - delta1_sim) < 0.001`.
4. **Numerical smoke** on ViT-L: 1000 random seeds through the full
   attention path, no NaN or Inf in any output.

Failing any of these => not shippable, either fix or pivot to reference.

## Reference kernels we port from (open-source, all shipped)

- **QuIP\#** (Tseng ICML 2024) — E8 lattice + Hadamard rotation, weight
  quantisation. Closest match. `https://github.com/Cornell-RelaxML/quip-sharp`
- **FlashAttention v2** (Dao 2023) — fused attention template.
  `https://github.com/Dao-AILab/flash-attention`
- **Marlin** (IST-DASLab, NeurIPS 2024) — INT4 kernel benchmarks.
  `https://github.com/IST-DASLab/marlin`
- **AQLM** (Egiazarian ICML 2024) — multi-codebook LUT layout.
  `https://github.com/Vahe1994/AQLM`
- **KIVI** (Liu ICML 2024) — cache management + prefill patterns.
  `https://github.com/jy-yuan/KIVI`

## Non-goals for this directory

- Λ24 (Leech) kernels — future work.
- Multi-GPU. VDA is single-GPU inference.
- Training with quantised KV — inference only.
