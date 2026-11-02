"""
Compressed temporal cache for Video-Depth-Anything streaming inference.

In streaming mode (``VideoDepthAnything.infer_video_depth_one``) VDA keeps, for
every past frame, the input hidden states of all eight temporal-attention
blocks (shapes like (H*W, 1, C), fp32).  This module stores those entries in a
compressed form and decompresses them only when they are read, so VDA's own
streaming loop runs unchanged:

    model = VideoDepthAnything(...)            # video_depth_stream variant
    model = enable_compressed_cache(model, mode="bw16", bits=3)

Modes
-----
fp32  : unchanged (reference).
fp16  : entries stored as fp16.
bw16  : block-randomized-Hadamard (block 64 over channels) + Λ16 lattice
        (Construction A over RM(1,4)), groups of 16, 8-bit per-group scale
        stored as one uint8 index plus one fp32 step per tensor.
        Storage per 16 scalars at b=3: 5 codeword bytes + 1 scale byte.
"""
from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from kernels.reference.packed_bw16 import (  # noqa: E402
    PackedBW16Bits, pack_bw16, unpack_bw16,
)
from research.transforms.hadamard import (  # noqa: E402
    inverse_randomized_hadamard_transform, randomized_hadamard_transform,
)

RHT_BLOCK = 64


_RHT_MATS = {}


def _rht_matrix(signs: torch.Tensor) -> torch.Tensor:
    """M with x @ M == randomized_hadamard_transform(x, signs) for 64-blocks."""
    key = (signs.device, signs.data_ptr())
    if key not in _RHT_MATS:
        from research.transforms.hadamard import build_hadamard_matrix
        H = build_hadamard_matrix(RHT_BLOCK, device=signs.device).float()
        _RHT_MATS[key] = (signs.float()[:, None] * H) / math.sqrt(RHT_BLOCK)
    return _RHT_MATS[key]


def rht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):  # keep fp32
        return x.float() @ _rht_matrix(signs)


def irht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return x.float() @ _rht_matrix(signs).T


def _signs(seed: int, device) -> torch.Tensor:
    g = torch.Generator(device='cpu').manual_seed(seed)
    s = torch.randint(0, 2, (RHT_BLOCK,), generator=g) * 2 - 1
    return s.to(device=device, dtype=torch.float32)


@dataclass
class _PackedTensor:
    codeword_bytes: torch.Tensor   # uint8 (G, n_bytes)
    scale_idx: torch.Tensor        # uint8 (G,)
    scale_step: float
    shape: Tuple[int, ...]
    packed_shape: Tuple[int, ...]
    dtype: torch.dtype
    bits: int
    n_bytes: int

    def nbytes(self) -> int:
        return self.codeword_bytes.numel() + self.scale_idx.numel() + 4


def _compress(t: torch.Tensor, bits: int, signs: torch.Tensor) -> _PackedTensor:
    shape, dtype, C = tuple(t.shape), t.dtype, t.shape[-1]
    assert C % RHT_BLOCK == 0, f"channel dim {C} not a multiple of {RHT_BLOCK}"
    x = t.float().reshape(-1, C // RHT_BLOCK, RHT_BLOCK)
    x = rht(x, signs).reshape(-1, C)
    p = pack_bw16(x, bits=bits, group_size=16, scale_bits=8)
    gs = p.group_scale.float()
    step = float(gs.max().clamp(min=1e-12)) / 255.0
    idx = (gs / step).round().clamp(0, 255).to(torch.uint8)
    return _PackedTensor(p.codeword_bytes.contiguous(), idx.contiguous(), step,
                         shape, tuple(p.original_shape), dtype, bits,
                         p.n_bytes_per_codeword)


try:
    from kernels.triton_kernels.decode_bw16 import TRITON_AVAILABLE, decode_bw16_triton
    from kernels.reference.bw16_codebook import bw16_cosets
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

_CODEBOOKS = {}
USE_TRITON = True
SEARCH_CHUNK_ROWS = 4096   # nearest-point search: ~8 MB per temporary
READ_CHUNK_ROWS = 512     # positions decoded per step in read_layer


def _decompress(p: _PackedTensor, signs: torch.Tensor) -> torch.Tensor:
    C = p.shape[-1]
    if USE_TRITON and TRITON_AVAILABLE and p.codeword_bytes.is_cuda:
        dev = p.codeword_bytes.device
        if dev not in _CODEBOOKS:
            _CODEBOOKS[dev] = bw16_cosets(dtype=torch.float32, device=dev)
        # Match the reference path exactly: scale is rounded to fp16 first.
        scale = (p.scale_idx.float() * p.scale_step).to(torch.float16).float()
        y = decode_bw16_triton(p.codeword_bytes, scale, _CODEBOOKS[dev], bits=p.bits)
        x = y.reshape(-1, C // RHT_BLOCK, RHT_BLOCK).float()
        x = irht(x, signs)
        return x.reshape(p.shape).to(p.dtype)
    gs = (p.scale_idx.float() * p.scale_step).to(torch.float16)
    pb = PackedBW16Bits(codeword_bytes=p.codeword_bytes, group_scale=gs,
                        original_shape=p.packed_shape,
                        bits=p.bits, n_bytes_per_codeword=p.n_bytes)
    x = unpack_bw16(pb, dtype=torch.float32).reshape(-1, C // RHT_BLOCK, RHT_BLOCK)
    x = irht(x, signs)
    return x.reshape(p.shape).to(p.dtype)


def compress_frame(tensors: List[torch.Tensor], bits: int,
                   signs: torch.Tensor) -> List[_PackedTensor]:
    """Batched version of ``[_compress(t, bits, signs) for t in tensors]``.

    One rotation, one nearest-point search and one bit-pack for all tensors;
    the 8-bit scale step stays per tensor, so the result is byte-identical to
    the per-tensor path.
    """
    import kernels.reference.packed_bw16 as _pk
    from kernels.reference.packed_bw16 import _bytes_per_codeword, _find_nearest_bw16
    _pk._CHUNK_ROWS = SEARCH_CHUNK_ROWS      # bound the 32-candidate temporary
    dev = tensors[0].device
    half = (1 << bits) // 2
    lo, hi = -float(half), float(half - 1)
    rows = [t.numel() // RHT_BLOCK for t in tensors]
    x = torch.cat([t.float().reshape(-1, RHT_BLOCK) for t in tensors])
    x = rht(x, signs).reshape(-1, 16)                                    # (G, 16)
    groups = torch.tensor([r * (RHT_BLOCK // 16) for r in rows], device=dev)
    ids = torch.repeat_interleave(torch.arange(len(tensors), device=dev), groups)

    scale = x.abs().amax(-1).clamp(min=1e-8) / (half - 1)               # (G,)
    smax = torch.zeros(len(tensors), device=dev).scatter_reduce(0, ids, scale, 'amax')
    step = smax.clamp(min=1e-8) / 255.0
    idx = (scale / step[ids]).round().clamp(0, 255)
    scale_q = idx * step[ids]
    xs = (x / scale_q.clamp(min=1e-12)[:, None]).clamp(lo, hi)
    cosets = bw16_cosets(dtype=xs.dtype, device=dev)
    best_idx, best_z = _find_nearest_bw16(xs, cosets, lo, hi)

    off_bits = bits - 1
    shifts = 5 + off_bits * torch.arange(16, device=dev, dtype=torch.int64)
    offs = best_z.round().to(torch.int64) + (1 << off_bits) // 2
    packed = (best_idx.to(torch.int64) & 0x1F) + (offs << shifts).sum(-1)
    nb = _bytes_per_codeword(bits)
    byte_shift = 8 * torch.arange(nb, device=dev, dtype=torch.int64)
    cw = ((packed[:, None] >> byte_shift) & 0xFF).to(torch.uint8)       # (G, nb)

    out, start = [], 0
    steps = step.tolist()
    for k, t in enumerate(tensors):
        g = int(groups[k])
        out.append(_PackedTensor(cw[start:start + g].contiguous(),
                                 idx[start:start + g].to(torch.uint8).contiguous(),
                                 steps[k], tuple(t.shape),
                                 (t.numel() // t.shape[-1], t.shape[-1]),
                                 t.dtype, bits, nb))
        start += g
    return out


@dataclass
class _ScalarTensor:
    q: torch.Tensor                 # int8/uint8 codes (G, g)
    scale: torch.Tensor             # fp32 (G, 1), already rounded to stored precision
    zero: Optional[torch.Tensor]    # fp32 (G, 1) for asymmetric, else None
    shape: Tuple[int, ...]
    dtype: torch.dtype
    rot: bool
    stored_bytes: int               # bytes a bit-packed layout would use

    def nbytes(self) -> int:
        return self.stored_bytes


def parse_scalar_spec(mode: str):
    """'int4-g16-rht' -> (bits=4, group=16, rot=True, asym=False).

    g16 variants use the same 8-bit per-group scale as the lattice;
    g64 variants use an fp16 scale (and fp16 zero-point if -asym), KIVI-style.
    """
    parts = mode.split('-')
    bits = 1 if parts[0] == 'bin' else int(parts[0][3:])
    if len(parts) > 1 and parts[1] == 'gT':
        group = 0                                   # one scale for the whole tensor
    else:
        group = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith('g') else 16
    return bits, group, 'rht' in parts, 'asym' in parts


def scalar_compress(t: torch.Tensor, mode: str, signs: torch.Tensor) -> _ScalarTensor:
    bits, g, rot, asym = parse_scalar_spec(mode)
    shape, C = tuple(t.shape), t.shape[-1]
    x = t.float().reshape(-1, C)
    if rot:
        x = rht(x.reshape(-1, C // RHT_BLOCK, RHT_BLOCK), signs).reshape(-1, C)
    xg = x.reshape(1, -1) if g == 0 else x.reshape(-1, g)
    G = xg.shape[0]
    if mode.startswith('bin'):
        # 1-bit: x_hat = s * sign(x); s = mean|x| minimises ||x - s*sign(x)||^2
        scale = xg.abs().mean(-1, keepdim=True).clamp(min=1e-12)
        if g == 16:
            step = scale.max().clamp(min=1e-12) / 255.0
            scale = (scale / step).round().clamp(0, 255) * step
            meta = G * 1
        else:
            scale = scale.half().float()
            meta = G * 2
        q = torch.where(xg >= 0, 1, -1).to(torch.int8)
        stored = (xg.numel() + 7) // 8 + meta
        return _ScalarTensor(q, scale, None, shape, t.dtype, rot, stored)
    if asym:
        mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8).half().float()
        zero = mn.half().float()
        q = ((xg - zero) / scale).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
        meta = G * 4                                            # fp16 scale + fp16 zero
    else:
        h = 2 ** (bits - 1)
        scale = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / max(h - 1, 1)
        if g == 16:                                             # 8-bit scale, as the lattice
            step = scale.max().clamp(min=1e-8) / 255.0
            scale = (scale / step).round().clamp(0, 255) * step
            meta = G * 1
        else:
            scale = scale.half().float()
            meta = G * 2
        zero = None
        q = (xg / scale.clamp(min=1e-12)).round().clamp(-h, h - 1).to(torch.int8)
    stored = (xg.numel() * bits + 7) // 8 + meta
    return _ScalarTensor(q, scale, zero, shape, t.dtype, rot, stored)


def scalar_decompress(p: _ScalarTensor, signs: torch.Tensor) -> torch.Tensor:
    x = p.q.float() * p.scale + (p.zero if p.zero is not None else 0.0)
    C = p.shape[-1]
    x = x.reshape(-1, C)
    if p.rot:
        x = irht(x.reshape(-1, C // RHT_BLOCK, RHT_BLOCK), signs).reshape(-1, C)
    return x.reshape(p.shape).to(p.dtype)


class CompressedEntry:
    """One frame's cache (list of per-block tensors), stored compressed.

    Indexing returns the decompressed tensor, so VDA's
    ``torch.cat([h[i] for h in cur_list], dim=1)`` works unchanged.
    """

    def __init__(self, tensors: List[torch.Tensor], mode: str, bits: int,
                 signs: Optional[torch.Tensor]):
        self.mode, self.signs = mode, signs
        if mode in ('fp16', 'fp32'):
            dt = torch.float16 if mode == 'fp16' else torch.float32
            self.items = [t.to(dt) for t in tensors]
            self.dtypes = [t.dtype for t in tensors]
        elif mode == 'bw16':
            self.items = compress_frame(tensors, bits, signs)
        elif mode.startswith(('int', 'bin')):
            self.items = [scalar_compress(t, mode, signs) for t in tensors]
        elif mode == 'zero':          # negative control: cache content destroyed
            self.items = [(tuple(t.shape), t.dtype, t.device) for t in tensors]
        else:
            raise ValueError(mode)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.mode in ('fp16', 'fp32'):
            return it.to(self.dtypes[i])
        if self.mode == 'zero':
            return torch.zeros(it[0], dtype=it[1], device=it[2])
        if self.mode.startswith(('int', 'bin')):
            return scalar_decompress(it, self.signs)
        return _decompress(it, self.signs)

    def nbytes(self) -> int:
        if self.mode in ('fp16', 'fp32'):
            return sum(t.numel() * t.element_size() for t in self.items)
        if self.mode == 'zero':
            return 0
        return sum(p.nbytes() for p in self.items)


def cache_nbytes(model) -> int:
    """Bytes held by the streaming cache (shared entries counted once),
    including the hidden anchor chain kept for delta coding."""
    seen, total = set(), 0
    for e in list(model.frame_cache_list) + list(getattr(model, '_delta_hidden', [])):
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, CompressedEntry):
            total += e.nbytes()
        else:
            total += sum(t.numel() * t.element_size() for t in e)
    return total


def read_layer(entries: List['CompressedEntry'], i: int) -> torch.Tensor:
    """Decompress layer ``i`` of several frames in one batch.

    Equivalent to ``torch.cat([e[i] for e in entries], dim=1)`` but issues one
    decode and one inverse rotation instead of one per frame.
    """
    first = entries[0]
    if first.mode == 'zero' or any(e.mode != first.mode for e in entries):
        return torch.cat([e[i] for e in entries], dim=1)    # mixed modes: per-entry
    if first.mode.startswith(('int', 'bin')):
        return torch.cat([scalar_decompress(e.items[i], first.signs) for e in entries], dim=1)
    if first.mode in ('fp16', 'fp32'):
        return torch.cat([e.items[i] for e in entries], dim=1).to(first.dtypes[i])
    ps = [e.items[i] for e in entries]
    p0 = ps[0]
    N, C = p0.shape[0], p0.shape[-1]
    if not (USE_TRITON and TRITON_AVAILABLE and p0.codeword_bytes.is_cuda):
        return torch.cat([_decompress(p, first.signs) for p in ps], dim=1)
    dev = p0.codeword_bytes.device
    if dev not in _CODEBOOKS:
        _CODEBOOKS[dev] = bw16_cosets(dtype=torch.float32, device=dev)
    Fn, gpr = len(ps), C // 16                        # frames, groups per position
    cw_all = torch.stack([p.codeword_bytes.reshape(N * gpr, p.n_bytes) for p in ps])
    idx_all = torch.stack([p.scale_idx.reshape(N * gpr) for p in ps])
    steps = torch.tensor([p.scale_step for p in ps], device=dev, dtype=torch.float32)
    out = torch.empty(N, Fn, C, device=dev, dtype=p0.dtype)
    for r0 in range(0, N, READ_CHUNK_ROWS):
        r1 = min(r0 + READ_CHUNK_ROWS, N)
        cw = cw_all[:, r0 * gpr:r1 * gpr].reshape(-1, p0.n_bytes)
        scale = (idx_all[:, r0 * gpr:r1 * gpr].float() * steps[:, None]).to(torch.float16).float()
        y = decode_bw16_triton(cw.contiguous(), scale.reshape(-1).contiguous(),
                               _CODEBOOKS[dev], bits=p0.bits)
        x = y.reshape(Fn, r1 - r0, C // RHT_BLOCK, RHT_BLOCK).float()
        x = irht(x, first.signs)
        out[r0:r1] = x.reshape(Fn, r1 - r0, C).permute(1, 0, 2)
    return out


class LazyLayer:
    """Cache of one temporal-attention layer, decompressed only when used."""

    def __init__(self, entries, i):
        self.entries, self.i = entries, i

    def materialize(self) -> torch.Tensor:
        return read_layer(self.entries, self.i)


def _patch_lazy(attn):
    """Decompress a LazyLayer cache right before this attention layer runs,
    so only one layer's cache is ever held decompressed."""
    original = attn.forward

    def forward(*args, cached_hidden_states=None, **kwargs):
        if isinstance(cached_hidden_states, LazyLayer):
            cached_hidden_states = cached_hidden_states.materialize()
        return original(*args, cached_hidden_states=cached_hidden_states, **kwargs)

    attn.forward = forward


class DeltaLazyLayer(LazyLayer):
    """Layer cache under closed-loop delta coding.

    ``chain`` = hidden anchor prefix + frame_cache_list[1:], i.e. consecutive
    frames starting at an intra entry.  Residual entries are added to the
    previous reconstruction; intra entries reset it.  The decoder repeats the
    encoder's operations, so reconstructions match exactly.
    """

    def __init__(self, first, chain, pos1, n_last, i):
        self.first, self.chain, self.pos1, self.n_last, self.i = first, chain, pos1, n_last, i

    def materialize(self) -> torch.Tensor:
        assert getattr(self.chain[0], 'intra', True)
        vals = read_layer(self.chain, self.i).float()        # (N, len(chain), C)
        rec = torch.empty_like(vals)
        for j, e in enumerate(self.chain):
            rec[:, j] = vals[:, j] if getattr(e, 'intra', True) else rec[:, j - 1] + vals[:, j]
        f0 = self.first[self.i].float()
        return torch.cat([f0, rec[:, self.pos1:self.pos1 + 1], rec[:, -self.n_last:]], dim=1)


class KVPair:
    """Projected keys/values of cached frames, without positional encoding."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v = k, v


def _patch_temporal_attention(attn):
    """Let a VDA TemporalAttention consume a KVPair cache.

    VDA's temporal attention computes, for every cached frame,
        K = W_k (h + pe[slot]),   V = W_v (h + pe[slot])
    with no bias and no normalisation between the positional encoding and the
    projection.  By linearity K = W_k h + W_k pe[slot], so W_k h and W_v h can
    be cached once per frame and the slot-dependent term added per step.
    This is exact up to floating-point rounding.
    """
    from einops import rearrange
    original = attn.forward

    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,
                video_length=None, cached_hidden_states=None):
        if not isinstance(cached_hidden_states, KVPair):
            return original(hidden_states, encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask, video_length=video_length,
                            cached_hidden_states=cached_hidden_states)
        assert encoder_hidden_states is None and attention_mask is None
        d = hidden_states.shape[1]
        h = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
        kc, vc = cached_hidden_states.k, cached_hidden_states.v
        T = kc.shape[1] + 1
        pe = attn.pos_encoder.pe[0, :T].to(h.dtype)                 # (T, C)
        q = attn.to_q(h + pe[T - 1])
        k_cur, v_cur = attn.to_k(h), attn.to_v(h)
        key = torch.cat([kc.to(k_cur.dtype), k_cur], 1) + attn.to_k(pe).to(k_cur.dtype)
        value = torch.cat([vc.to(v_cur.dtype), v_cur], 1) + attn.to_v(pe).to(v_cur.dtype)
        out = attn._attention(attn.reshape_heads_to_batch_dim(q),
                              attn.reshape_heads_to_batch_dim(key),
                              attn.reshape_heads_to_batch_dim(value), None)
        out = attn.to_out[1](attn.to_out[0](out))
        out = rearrange(out, "(b d) f c -> (b f) d c", d=d)
        return out, (k_cur, v_cur)

    attn.forward = forward


def enable_compressed_cache(model, mode: str = 'bw16', bits: int = 3, seed: int = 0,
                            kv: bool = False, lazy: bool = True, delta: int = 0,
                            residual_mode: Optional[str] = None):
    """Patch a streaming VideoDepthAnything so its frame cache is compressed.

    Frame 0 runs VDA's own code (it reads no cache).  Later frames run a copy
    of VDA's streaming step whose only change is how the cache is read
    (batched decompression) and stored (compressed on insertion).

    delta=K>0 stores each frame as the quantized residual from the previous
    frame's reconstruction (closed loop), with an intra frame every K frames.

    kv=False caches each frame's attention input h (VDA's design: K/V of all
    31 cached frames are recomputed every frame).  kv=True caches the
    projected W_k h and W_v h instead (2x the entries, no recomputation).
    """
    if mode == 'fp32' and not kv and not delta:
        return model
    assert not (kv and delta), "delta coding is implemented for the h cache only"
    assert mode in ('fp32', 'fp16', 'bw16', 'zero') or mode.startswith(('int', 'bin')), mode
    residual_mode = residual_mode or mode
    import torch.nn.functional as F
    from video_depth_anything import video_depth_stream as vds
    INFER_LEN = vds.INFER_LEN

    original = model.infer_video_depth_one
    state = {'signs': None, 'prev': None, 'since_intra': 0}
    model._delta_hidden = []
    attns = [m for m in model.head.modules() if type(m).__name__ == 'TemporalAttention']
    if kv:
        for a in attns:
            _patch_temporal_attention(a)
    elif lazy:
        for a in attns:
            _patch_lazy(a)

    def compress(cache_list):
        if kv:
            flat = []
            for i, e in enumerate(cache_list):
                if isinstance(e, tuple):               # (k, v) from patched attention
                    flat.extend(e)
                else:                                  # raw h from VDA's first frame
                    flat.extend([attns[i].to_k(e), attns[i].to_v(e)])
            cache_list = flat
        if not delta:
            return CompressedEntry(list(cache_list), mode, bits, state['signs'])
        cur = [t.float() for t in cache_list]
        intra = state['prev'] is None or state['since_intra'] >= delta - 1
        payload = cur if intra else [c - p for c, p in zip(cur, state['prev'])]
        e = CompressedEntry(payload, mode if intra else residual_mode, bits, state['signs'])
        e.intra = intra
        dec = [e[i].float() for i in range(len(e))]
        state['prev'] = dec if intra else [p + d for p, d in zip(state['prev'], dec)]
        state['since_intra'] = 0 if intra else state['since_intra'] + 1
        return e

    def read(cur_list):
        n = len(cur_list[0])
        if kv:
            return [KVPair(read_layer(cur_list, 2 * i), read_layer(cur_list, 2 * i + 1))
                    for i in range(n // 2)]
        if delta:
            L, H = model.frame_cache_list, model._delta_hidden
            chain = list(H) + list(L[1:])
            return [DeltaLazyLayer(L[0], chain, len(H), INFER_LEN - 3, i) for i in range(n)]
        if lazy:
            return [LazyLayer(cur_list, i) for i in range(n)]
        return [read_layer(cur_list, i) for i in range(n)]

    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        if state['signs'] is None:
            state['signs'] = _signs(seed, next(self.parameters()).device)
        if self.transform is None:
            state['prev'], state['since_intra'] = None, 0
        if self.transform is None:  # first frame: VDA's own path, then compress
            out = original(frame, input_size=input_size, device=device, fp32=fp32)
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    shared = compress(self.frame_cache_list[0])
            self.frame_cache_list = [shared] * len(self.frame_cache_list)
            self._delta_hidden = []
            return out

        self.id += 1
        frame_height, frame_width = frame.shape[:2]
        assert frame_height == self.frame_height and frame_width == self.frame_width
        cur_input = torch.from_numpy(
            self.transform({'image': frame.astype('float32') / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                cur_feature = self.forward_features(cur_input)
                x_shape = cur_input.shape

        cur_list = self.frame_cache_list[0:2] + self.frame_cache_list[-INFER_LEN + 3:]
        assert len(cur_list) == INFER_LEN - 1
        cur_cache = read(cur_list)

        with torch.no_grad():
            with torch.autocast(device_type=device, enabled=(not fp32)):
                depth, new_cache = self.forward_depth(cur_feature, x_shape,
                                                      cached_hidden_state_list=cur_cache)
        del cur_cache
        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width),
                              mode='bilinear', align_corners=True)
        new_depth = depth[-1][0].cpu().numpy()
        with torch.no_grad():
            self.frame_cache_list.append(compress(new_cache))

        self.frame_id_list.append(self.id)
        if self.id + INFER_LEN > self.gap + 1:
            del self.frame_id_list[1]
            popped = self.frame_cache_list.pop(1)
            if delta:
                if getattr(self.frame_cache_list[1], 'intra', True):
                    self._delta_hidden = []
                elif getattr(popped, 'intra', True):
                    self._delta_hidden = [popped]
                else:
                    self._delta_hidden.append(popped)
        return new_depth

    model.infer_video_depth_one = types.MethodType(infer_video_depth_one, model)
    return model
