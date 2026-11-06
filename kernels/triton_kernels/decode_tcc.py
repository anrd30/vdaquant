"""
Fused Triton decode for scalar-coded streaming caches (TCC and per-entry INT/1-bit).

One launch per layer decodes a whole chain of cached entries: for every element it
walks the chain in order, unpacks the entry's code from the packed bytes, scales it,
and either restarts (keyframe / independently coded entry) or adds it to the running
reconstruction (closed-loop residual with the previous-frame predictor).  Entries are
read in place through a small pointer table, and only the slots attention needs are
written, straight into a fixed-size (N, S, C) float32 output.  The output size does
not depend on the chain length, so the caching allocator reuses one block.

Bit-identical to the PyTorch path: codes and scales are the same integers / float32
values, a 1-bit residual is exactly +-scale, and floating-point fusion is disabled
so q * s + zero rounds twice, as in PyTorch.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:            # pragma: no cover
    TRITON_AVAILABLE = False

KIND = {'bin': 0, 'sym': 1, 'asym': 2}

if TRITON_AVAILABLE:
    @triton.jit
    def _decode_one(cp, sp, zp, step, e, mask, NB: tl.constexpr, KIND: tl.constexpr,
                    OFF: tl.constexpr, G: tl.constexpr, S8: tl.constexpr):
        PER: tl.constexpr = 8 // NB
        codes = cp.to(tl.pointer_type(tl.uint8))
        byte = tl.load(codes + e // PER, mask=mask, other=0).to(tl.int32)
        code = (byte >> ((e % PER) * NB)) & ((1 << NB) - 1)
        if S8:                                  # 8-bit scale index times per-tensor step
            s = tl.load(sp.to(tl.pointer_type(tl.uint8)) + e // G, mask=mask, other=0).to(tl.float32) * step
        else:                                   # fp16 scale
            s = tl.load(sp.to(tl.pointer_type(tl.float16)) + e // G, mask=mask, other=0.0).to(tl.float32)
        if KIND == 0:
            q = code.to(tl.float32) * 2.0 - 1.0
        elif KIND == 1:
            q = (code - OFF).to(tl.float32)
        else:
            q = code.to(tl.float32)
        x = q * s
        if KIND == 2:
            x = x + tl.load(zp.to(tl.pointer_type(tl.float16)) + e // G, mask=mask, other=0.0).to(tl.float32)
        return x

    @triton.jit
    def _tcc_decode_kernel(out_ptr, tab_ptr, step_ptr, E, C, F, S,
                           NB_A: tl.constexpr, KIND_A: tl.constexpr, OFF_A: tl.constexpr, G_A: tl.constexpr,
                           S8_A: tl.constexpr,
                           NB_B: tl.constexpr, KIND_B: tl.constexpr, OFF_B: tl.constexpr, G_B: tl.constexpr,
                           S8_B: tl.constexpr,
                           ACCUM: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        mask = e < E
        out_off = (e // C) * (S * C) + (e % C)
        rec = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(F):
            kind = tl.load(tab_ptr + j)               # 0: restart, 1: residual
            dst = tl.load(tab_ptr + F + j)            # output slot, or -1
            cp = tl.load(tab_ptr + 2 * F + j)
            sp = tl.load(tab_ptr + 3 * F + j)
            zp = tl.load(tab_ptr + 4 * F + j)
            step = tl.load(step_ptr + j)
            if kind == 0:
                rec = _decode_one(cp, sp, zp, step, e, mask, NB_A, KIND_A, OFF_A, G_A, S8_A)
            else:
                x = _decode_one(cp, sp, zp, step, e, mask, NB_B, KIND_B, OFF_B, G_B, S8_B)
                if ACCUM:
                    rec = rec + x
                else:
                    rec = x
            if dst >= 0:
                tl.store(out_ptr + out_off + dst * C, rec, mask=mask)


def tcc_decode(items, residual, dst, n_slots, n_rows, C, accumulate, block=1024):
    """items[j]: _ScalarTensor of chain position j; residual[j]: True if it adds to the
    running reconstruction; dst[j]: output slot or -1.  All restart entries share one
    format, as do all residual entries.  Returns (n_rows, n_slots, C) float32."""
    pa = next(p for p, r in zip(items, residual) if not r)
    pb = next((p for p, r in zip(items, residual) if r), pa)
    dev = pa.q.device
    F = len(items)
    tab = torch.tensor([[int(r) for r in residual], list(dst),
                        [p.q.data_ptr() for p in items], [p.scale.data_ptr() for p in items],
                        [p.zero.data_ptr() if p.zero is not None else 0 for p in items]],
                       dtype=torch.int64).to(dev, non_blocking=True)
    steps = torch.tensor([p.step if p.step is not None else 1.0 for p in items],
                         dtype=torch.float32).to(dev, non_blocking=True)
    out = torch.empty(n_rows, n_slots, C, device=dev, dtype=torch.float32)
    E = n_rows * C
    _tcc_decode_kernel[(triton.cdiv(E, block),)](
        out, tab, steps, E, C, F, n_slots,
        NB_A=pa.nbits, KIND_A=KIND[pa.kind], OFF_A=int(pa.off), G_A=pa.gshape[1], S8_A=pa.step is not None,
        NB_B=pb.nbits, KIND_B=KIND[pb.kind], OFF_B=int(pb.off), G_B=pb.gshape[1], S8_B=pb.step is not None,
        ACCUM=accumulate, BLOCK=block, enable_fp_fusion=False)
    return out
