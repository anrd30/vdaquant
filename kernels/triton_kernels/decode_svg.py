"""
Fused Triton decode for StreamVGGT's compressed KV cache (research/models/streamvggt_cache.py).

One launch per layer and per K or V decodes every cached frame: each GPU thread
owns one element (head, token, channel) of a frame, walks the frames in order,
unpacks its code from the bit-packed bytes, scales it, and either restarts the
reconstruction (keyframe or independently coded frame) or adds the decoded
residual to it (closed-loop TCC).  Each frame's value is written straight into
the attention buffer (B, h, S+1, N, d) in its dtype, and the newest frame's
float32 reconstruction is returned for closed-loop encoding.

Same arithmetic as streamvggt_cache.dequantize_batch: codes and scales are the
same integers and float32 values, and floating-point fusion is disabled.  Only
the order of the residual sum differs from PyTorch's cumsum, so results agree
to float32 rounding; encoder and decoder both use this path, so they stay in
closed loop.
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
    def _decode(cp, sp, zp, stp, e, mask, NB: tl.constexpr, KIND: tl.constexpr, OFF: tl.constexpr,
                G: tl.constexpr):
        PER: tl.constexpr = 8 // NB
        byte = tl.load(cp.to(tl.pointer_type(tl.uint8)) + e // PER, mask=mask, other=0).to(tl.int32)
        code = (byte >> ((e % PER) * NB)) & ((1 << NB) - 1)
        if KIND == 2:                           # asymmetric: fp16 scale and zero
            s = tl.load(sp.to(tl.pointer_type(tl.float16)) + e // G, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(zp.to(tl.pointer_type(tl.float16)) + e // G, mask=mask, other=0.0).to(tl.float32)
            return code.to(tl.float32) * s + z
        step = tl.load(stp.to(tl.pointer_type(tl.float32)))
        s = tl.load(sp.to(tl.pointer_type(tl.uint8)) + e // G, mask=mask, other=0).to(tl.float32) * step
        if KIND == 0:
            q = code.to(tl.float32) * 2.0 - 1.0
        else:
            q = (code - OFF).to(tl.float32)
        return q * s

    @triton.jit
    def _svg_decode_kernel(out_ptr, last_ptr, tab_ptr, E, ND, S1, F,
                           NB_A: tl.constexpr, KIND_A: tl.constexpr, OFF_A: tl.constexpr, G_A: tl.constexpr,
                           NB_B: tl.constexpr, KIND_B: tl.constexpr, OFF_B: tl.constexpr, G_B: tl.constexpr,
                           BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        e = pid * BLOCK + tl.arange(0, BLOCK)
        mask = e < E
        out_off = (e // ND) * (S1 * ND) + (e % ND)          # (B*h, S+1, N*d) layout
        rec = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(F):
            kind = tl.load(tab_ptr + j)                     # 0: restart, 1: add residual
            cp = tl.load(tab_ptr + F + j)
            sp = tl.load(tab_ptr + 2 * F + j)
            zp = tl.load(tab_ptr + 3 * F + j)
            stp = tl.load(tab_ptr + 4 * F + j)
            if kind == 0:
                rec = _decode(cp, sp, zp, stp, e, mask, NB_A, KIND_A, OFF_A, G_A)
            else:
                rec = rec + _decode(cp, sp, zp, stp, e, mask, NB_B, KIND_B, OFF_B, G_B)
            tl.store(out_ptr + out_off + j * ND, rec.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(last_ptr + e, rec, mask=mask)


def _fmt(kind: str):
    """(bits stored per code, kind id, offset, group) for a streamvggt_cache kind."""
    if kind == 'bin':
        return 1, KIND['bin'], 0, 16
    if kind.startswith('a'):
        b, g = kind[1:].split('g')
        return {2: 2, 3: 4, 4: 4}[int(b)], KIND['asym'], 0, int(g)
    b = int(kind[3:])
    return {2: 2, 3: 4, 4: 4}[b], KIND['sym'], 2 ** (b - 1), 16


def supported(items) -> bool:
    return TRITON_AVAILABLE and all(it['kind'] != 'raw' and not it['kind'].startswith('c') for it in items)


def svg_decode(items, residual, dst, frame_shape, block=1024):
    """items[j]: quantize() dict of frame j; residual[j]: True if it adds to the running
    reconstruction.  Writes frame j into dst[:, :, j] (dst is (B, h, S+1, N, d)) and
    returns the last frame's float32 reconstruction, shaped frame_shape (B, h, 1, N, d)."""
    B, h, _, N, d = frame_shape
    F = len(items)
    key = next(it for it, r in zip(items, residual) if not r)
    res = next((it for it, r in zip(items, residual) if r), key)
    zero = 0
    tab = torch.tensor([[int(r) for r in residual],
                        [it['packed'].data_ptr() for it in items],
                        [(it['scale'] if 'scale' in it else it['sidx']).data_ptr() for it in items],
                        [it['zero'].data_ptr() if 'zero' in it else zero for it in items],
                        [it['step'].data_ptr() if 'step' in it else zero for it in items]],
                       dtype=torch.int64).to(dst.device, non_blocking=True)
    last = torch.empty(frame_shape, dtype=torch.float32, device=dst.device)
    E, ND = B * h * N * d, N * d
    a, b = _fmt(key['kind']), _fmt(res['kind'])
    _svg_decode_kernel[(triton.cdiv(E, block),)](
        dst, last, tab, E, ND, dst.shape[2], F,
        NB_A=a[0], KIND_A=a[1], OFF_A=a[2], G_A=a[3], NB_B=b[0], KIND_B=b[1], OFF_B=b[2], G_B=b[3],
        BLOCK=block, enable_fp_fusion=False)
    return last
