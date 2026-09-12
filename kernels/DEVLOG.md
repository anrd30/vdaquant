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
