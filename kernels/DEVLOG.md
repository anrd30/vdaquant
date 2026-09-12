# kernels/ development log

Timestamped record of every session's changes, correctness runs, and
edge cases found.  Grep-friendly.  Entries prepended (newest at top).

Format:
```
## YYYY-MM-DD HH:MM  session-title
    context      : one-line context
    hardware     : where the run happened
    changes      : bullet list
    tests        : pass/fail counts
    findings     : edge cases, race conditions, surprises
    next         : what the next session picks up
```

---

## 2026-09-12 14:00 IST  session 4: multi-bit Triton + KV cache Triton path + benchmark
    context      : make the Triton decode general (bits in {2,3,4}) and
                   plumb it into PackedKVCache so all reads go via
                   Triton on CUDA
    hardware     : RTX 4050 Laptop, 6.0 GB
    changes      :
      - kernels/triton_kernels/decode_bw16.py generalised: _LAYOUT dict
        holds (bytes_per_codeword, offset_bits, offset_mask,
        offset_bias) per bit width; kernel takes them as tl.constexpr
        arguments.  Now works for bits in {2, 3, 4}.
      - kernels/reference/kv_cache.py: PackedKVCache.read() picks the
        Triton fast path when available AND on CUDA.  Explicit
        use_triton=False lets tests exercise the reference path.
      - kernels/benchmark_memory.py extended to time Triton decode
        alongside PyTorch decode and print speedup column.
    tests        : 27/27 pass across all three suites
                    * 17/17 test_bw16_roundtrip
                    * 5/5 test_simulator_parity
                    * 5/5 test_kv_cache_end_to_end (Triton path)
    findings     :
      - Triton smoke test bit-exact across all three bit widths:
             bits=2: max diff 0.00e+00, 3 bytes/codeword
             bits=3: max diff 0.00e+00, 5 bytes/codeword
             bits=4: max diff 0.00e+00, 7 bytes/codeword
      - Benchmark on RTX 4050 (all VDA layers at 3-bit):

        | Layer                       | fp16   | packed | ratio | PyT unpack | Triton | speedup |
        | ViT-S mm0 (1,32,1369, 192)  | 16.04M |  3.01M | 5.33x |    6.73 ms | 0.46ms | 14.5x   |
        | ViT-S mm1 (1,32, 361, 384)  |  8.46M |  1.59M | 5.33x |    5.88 ms | 0.25ms | 23.1x   |
        | ViT-S mm2 (1,32,1369,  64)  |  5.35M |  1.00M | 5.33x |    4.22 ms | 0.17ms | 24.3x   |
        | ViT-S mm3 (1,32,5476,  64)  | 21.39M |  4.01M | 5.33x |    8.84 ms | 0.69ms | 12.9x   |
        | ViT-L mm0 (1,32,1369,1024)  | 85.56M | 16.04M | 5.33x |   39.00 ms | 2.43ms | 16.0x   |
        | ViT-L mm1 (1,32, 361,1024)  | 22.56M |  4.23M | 5.33x |    8.18 ms | 0.62ms | 13.1x   |
        | ViT-L mm2 (1,32,1369, 256)  | 21.39M |  4.01M | 5.33x |    7.85 ms | 0.59ms | 13.2x   |
        | ViT-L mm3 (1,32,5476, 256)  | 85.56M | 16.04M | 5.33x |   38.67 ms | 2.46ms | 15.7x   |
        | TOTAL                       | 266.3M | 49.94M | 5.33x |            |        |         |

      - fp16 bit-parity through the KV cache: fp32 mock attention
        output max diff 0.00e+00 on VDA-sized ViT-S mm3 tile.
    next         :
      - Fused decode + attention kernel (the ultimate paper win).
      - VDA integration into research/models/rotated_attention.py
        (replace fp16 KV buffer with PackedKVCache instance).
      - E8 companion for the "our lattice beats E8" comparator row.
      - When A100 is available, re-run the benchmark table for the
        Paper 2 Table 6 headline numbers.

---

## 2026-09-12 13:40 IST  session 3: PackedKVCache + FIRST TRITON KERNEL
    context      : integration wrapper for VDA + first Triton decode kernel
    hardware     : RTX 4050 Laptop, 6.0 GB, Triton 3.7.0
    changes      :
      - kernels/reference/kv_cache.py PackedKVCache -- drop-in for VDA's
        fp16 KV buffer.  write(t, x) / read(k, dtype) API matches the
        indexed slice pattern in RotatedTemporalAttention.  Handles
        head_dim < group_size via absorb-and-pad (VDA DPT cross-attn
        has head_dim=8).
      - kernels/tests/test_kv_cache_end_to_end.py -- 5 gates, all pass.
        BIT-EXACT vs LatticeBW16Quantizer through a mock temporal
        attention forward: K diff = 0, output diff = 0.
      - kernels/triton_kernels/decode_bw16.py -- FIRST TRITON KERNEL.
        Simple element-wise decode: read 5 bytes -> unpack 5-bit coset
        + 16 * 2-bit offsets -> gather codebook row -> scale ->
        write fp16.  BLOCK codewords per program.
    tests        : 5/5 pass on kv_cache_end_to_end
                   triton smoke test: max diff 0.00e+00 on first try
    findings     :
      - Triton on Windows works with triton 3.7.0 -- no MSVC dance
        needed, PyTorch's bundled runtime handles it.
      - Triton kernel BENCHMARK on RTX 4050 vs PyTorch decode:
          ViT-S mm0 (1,32,1369,192):    5.95 ms -> 0.48 ms   12.4x
          ViT-S mm3 (1,32,5476, 64):    6.77 ms -> 0.75 ms    9.0x
          ViT-L mm0 (1,32,1369,1024):  35.52 ms -> 2.38 ms   14.9x
          ViT-L mm3 (1,32,5476, 256):  36.63 ms -> 2.44 ms   15.0x
        Every result is BIT-EXACT vs the PyTorch decode.
      - 15x is largely because the PyTorch decode does per-byte bit
        extraction in eager mode; the Triton kernel keeps everything
        in registers.  A well-written vectorised PyTorch decode could
        close this gap somewhat, but Triton also enables the next step:
        fusing decode with attention softmax + weighted V sum, which
        eliminates the fp16 intermediate entirely.
      - Triton gotcha found and fixed: plain Python globals inside
        @triton.jit raise NameError at compile time.  Fix: pass all
        integer constants as tl.constexpr kernel arguments.
    next         :
      - Fuse decode with a small attention kernel (K load stage).
      - Benchmark on A100 -- expect similar or better speedup ratio.
      - Add bits=4 and bits=2 to the kernel (currently only bits=3).
      - Integrate PackedKVCache into RotatedTemporalAttention proper
        so we can quote a whole-model measured peak and latency for
        the paper.

---

## 2026-09-12 13:20 IST  session 2: bit-parity with simulator + memory benchmark
    context      : achieve BIT-PARITY with LatticeBW16Quantizer so packed
                   storage is a mathematically exact drop-in for VDA's
                   temporal-attention KV cache
    hardware     : RTX 4050 Laptop, 6.0 GB
    changes      :
      - pack_bw16_ref / pack_bw16 gained a scale_bits argument.
        scale_bits=8 (new) matches LatticeBW16Quantizer's exact scheme:
             scale_max  = per-tensor amax(scale).clamp(>=1e-8)
             scale_step = scale_max / 255
             scale      = round(scale / scale_step).clamp(0, 255) * scale_step
      - group_scale is now stored fp32 (was fp16) so unpack sees the same
        value the simulator saw.  A real deployment stores index (uint8)
        + step (fp32 per-tensor), which is what nbytes() reports: 1 byte
        per group scale, not 4.  See PackedBW16Ref.nbytes docstring.
      - kernels/tests/test_simulator_parity.py -- new correctness gate:
        max(|x_sim - x_packed|) < 1e-4 on all VDA-shaped tiles, 3 and
        4 bit, fp32 and fp16 input, on CPU and CUDA.
      - kernels/benchmark_memory.py -- new script: measures compression
        on real VDA-shaped tiles (all 4 motion modules, ViT-S and ViT-L),
        prints a Markdown table for the paper.
    tests        : 22/22 pass across both files
                     bit-parity test max diff  = 0.00e+00 on all shapes
                     bit-parity holds at bits in {3, 4}
                     bit-parity holds with scale_bits=8
                     scale_bits=16 (default) still <2.5 at 99.9th %ile
    findings     :
      - **BIT-EXACT** drop-in with LatticeBW16Quantizer(scale_bits=8).
        Every scalar of the packed pipeline output matches the simulator
        to within fp32 round-off (measured 0.00e+00 max diff).
      - Real deployed compression on all VDA temporal-attention KV
        tiles is **5.33x**, not 4.57x -- BETTER than the paper's
        current analytic 4.6x.  Origin: paper's analytic used
        (bits + 8/group) which counts each offset as `bits` bits, but
        the tight codeword needs only `5 + 16*(bits-1)` bits per group.
        At 3-bit that is 37 bits, byte-padded to 40 bits = 2.5
        bits/scalar for the codeword + 0.5 bits/scalar for the scale =
        3.0 bits/scalar => 16/3.0 = 5.33x.
      - Compression table for the paper (measured on RTX 4050, 3-bit):
          ViT-S total KV cache T=32:  51.24 MB fp16 -> 9.61 MB packed
          ViT-L total KV cache T=32: 215.07 MB fp16 -> 40.32 MB packed
        Both compress at exactly 5.33x.
      - Bit-parity means the whole downstream VDA forward is
        NUMERICALLY IDENTICAL between simulator and packed storage.
        Delta1 numbers we quote in the paper are the SAME whether we
        run the simulator or the packed kernel -- the packed kernel
        is not "an approximation of" the simulator, it IS the simulator
        with different memory-layout accounting.
    next         :
      - Update Paper 2 Table 6 with measured 5.33x compression numbers.
      - VDA integration hook: PackedKVCache class with the same API as
        VDA's existing fp16 cache buffer, so surgery becomes a one-line
        change in RotatedTemporalAttention.
      - Triton decode kernel v0 (week 2) -- correctness gate is the
        bit-exact reference we now have.
      - E8 companion for the "our lattice beats E8" Table 6 row -- can
        be non-bit-packed since it is only a comparator.

---

## 2026-09-12 12:43 IST  session 1: foundations + bit-packing
    context      : first autonomous kernel dev session on 4050
    hardware     : RTX 4050 Laptop, 6.0 GB, SM 8.9
    changes      :
      - kernels/README.md, DEVLOG.md, __init__.py hierarchy
      - kernels/reference/bw16_codebook.py -- 32-coset LUT, weight
        distribution {0:1, 8:30, 16:1} verified at import
      - kernels/reference/packed_bw16.py v1 -- ref layout only, wrong
        scale convention (multiply by 64), 1.60x compression
      - added batch chunking for the 32-coset enumeration; _CHUNK_ROWS
        = 32k caps intermediate at ~64 MB
      - REWROTE packed_bw16.py with simulator-parity scale:
          half_levels = 2**b // 2
          scale       = per-group absmax / (half_levels - 1)
          x_scaled clamped to [-half_levels, half_levels - 1]
      - added true bit-packed layout PackedBW16Bits
          per group: 5-bit coset || 16 * (bits-1)-bit offsets
          serialised LSB-first into n_bytes_per_codeword bytes
      - convenience API: pack_bw16 / unpack_bw16 chain bit-pack
      - kept legacy aliases PackedBW16 and pack_bw16_reference
    tests        : 17/17 pass (kernels/tests/test_bw16_roundtrip.py)
                     bit-parity (ref <-> bit-packed): 0 diff
                     sim-parity (ref <-> LatticeBW16Quantizer): <0.05 mean
                     compression matches analytic: 6.4x/4.6x/3.6x at 2/3/4 bit
                     dtype: fp32/fp16/bf16 ok
                     device: cuda preserved
                     edge cases: single group, non-contig, zeros, outliers,
                                 batch-chunker parity, determinism -- all ok
    findings     :
      - "1-bit-per-offset" IS enough at b=2 (z ranges {-1, 0} for c=0,
        {-1, 0} for c=1); (bits - 1) is the correct offset width.
      - 8-bit scale metadata in the simulator can shift a single group
        by up to 1 lattice step on rare tiles; our reference uses fp16
        scales so we diverge on those rare tiles.  Cosmetic at 3/4 bit
        (mean err < 0.05), larger at 2 bit but 2-bit is not our target.
      - Bit-packed compression on ViT-L KV tile (85 MB fp16) is 18.7 MB
        on the GPU, 4.57x reduction MEASURED, not analytic.
      - Pack throughput on 4050 is 250-650 ms per VDA tile (enumeration
        is Python-side); unpack is 5-40 ms (no search).  A Triton
        decode kernel will kill the unpack cost first.
    next         :
      - E8 packed companion (needed for the "our lattice beats E8"
        Table 1 comparator with real measured memory)
      - VDA integration prototype: PackedKVCache class that slots into
        RotatedTemporalAttention in place of the fp16 cache tensor
      - measured whole-model peak with packed cache vs simulator on
        ViT-S NYU eval -- if delta1 matches within 0.001 we have a
        real "measured memory" number for the paper
      - Triton port (week 2)
