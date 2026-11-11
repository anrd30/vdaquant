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
intN/bin : scalar quantizers ('int4-g16', 'bin-g16', ...), codes bit-packed
        (1-bit 8/byte, INT2 4/byte, INT3/INT4 in 4-bit nibbles).
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


def _code_bits(bits: int, binary: bool) -> int:
    """Bits per stored code: 1-bit and INT1 -> 1, INT2 -> 2, INT3/INT4 -> 4 (nibbles)."""
    if binary or bits == 1:
        return 1
    if bits == 2:
        return 2
    return 4 if bits <= 4 else 8


def pack_codes(codes: torch.Tensor, nb: int) -> torch.Tensor:
    """Unsigned codes in [0, 2**nb) -> flat uint8, ``8 // nb`` codes per byte.

    Code k of a byte occupies bits [nb*k, nb*(k+1)).  The tail is zero padded.
    """
    flat = codes.reshape(-1).to(torch.uint8)
    if nb == 8:
        return flat.contiguous()
    per = 8 // nb
    pad = (-flat.numel()) % per
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.reshape(-1, per)
    out = flat[:, 0].clone()
    for k in range(1, per):
        out |= flat[:, k] << (nb * k)
    return out


def unpack_codes(packed: torch.Tensor, nb: int, n: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes`: flat uint8 of ``n`` codes."""
    if nb == 8:
        return packed
    per, mask = 8 // nb, (1 << nb) - 1
    out = torch.empty(packed.numel(), per, dtype=torch.uint8, device=packed.device)
    for k in range(per):
        out[:, k] = (packed >> (nb * k)) & mask
    return out.reshape(-1)[:n]


@dataclass
class _ScalarTensor:
    """Scalar-quantized tensor with bit-packed codes.

    q      : uint8 (n_bytes,), codes packed ``8 // nbits`` per byte.
    scale  : g16 -> uint8 index (G,) with one fp32 ``step`` per tensor
             (scale = index * step); otherwise fp16 (G,).
    zero   : fp16 (G,) for asymmetric, else None.
    kind   : 'bin' (code 1 -> +1, 0 -> -1), 'sym' (value = code - off),
             'asym' (value = code).
    """
    q: torch.Tensor
    scale: torch.Tensor
    zero: Optional[torch.Tensor]
    step: Optional[float]
    kind: str
    off: int
    nbits: int
    gshape: Tuple[int, int]         # (G, group size) of the unpacked codes
    shape: Tuple[int, ...]
    dtype: torch.dtype
    rot: bool

    def nbytes(self) -> int:
        """Allocated bytes of the stored tensors (+4 for the fp32 step)."""
        n = self.q.numel() * self.q.element_size()
        n += self.scale.numel() * self.scale.element_size()
        if self.zero is not None:
            n += self.zero.numel() * self.zero.element_size()
        return n + (4 if self.step is not None else 0)

    @property
    def stored_bytes(self) -> int:
        return self.nbytes()


def parse_scalar_spec(mode: str):
    """'int4-g16-rht' -> (bits=4, group=16, rot=True, asym=False).

    g16 variants use the same 8-bit per-group scale as the lattice;
    g64 variants use an fp16 scale (and fp16 zero-point if -asym), KIVI-style.
    Codes are bit-packed: 1-bit 8/byte, INT2 4/byte, INT3 and INT4 in 4-bit
    nibbles (2/byte; INT3 therefore costs 4 stored bits per code).
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
    # 8-bit scale index (as g16) for any group size with '-s8', e.g. 'bin-g32-s8'
    s8 = g == 16 or 's8' in mode.split('-')
    shape, C = tuple(t.shape), t.shape[-1]
    x = t.float().reshape(-1, C)
    if rot:
        x = rht(x.reshape(-1, C // RHT_BLOCK, RHT_BLOCK), signs).reshape(-1, C)
    xg = x.reshape(1, -1) if g == 0 else x.reshape(-1, g)
    G = xg.shape[0]
    step, zero, off = None, None, 0
    if mode.startswith('bin'):
        # 1-bit: x_hat = s * sign(x); s = mean|x| minimises ||x - s*sign(x)||^2
        kind, nb = 'bin', 1
        scale = xg.abs().mean(-1, keepdim=True).clamp(min=1e-12)
        if s8:
            step_t = scale.max().clamp(min=1e-12) / 255.0
            idx = (scale / step_t).round().clamp(0, 255)
            scale_st, step = idx.to(torch.uint8).reshape(-1), float(step_t)
        else:
            scale_st = scale.half().reshape(-1)
        codes = (xg >= 0).to(torch.uint8)
    elif asym:
        kind, nb = 'asym', _code_bits(bits, False)
        mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8).half().float()
        zero_f = mn.half().float()
        codes = ((xg - zero_f) / scale).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
        scale_st, zero = scale.half().reshape(-1), zero_f.half().reshape(-1)
    else:
        kind, nb = 'sym', _code_bits(bits, False)
        h = 2 ** (bits - 1)
        scale = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / max(h - 1, 1)
        if s8:                                                  # 8-bit scale, as the lattice
            step_t = scale.max().clamp(min=1e-8) / 255.0
            idx = (scale / step_t).round().clamp(0, 255)
            scale = idx * step_t
            scale_st, step = idx.to(torch.uint8).reshape(-1), float(step_t)
        else:
            scale = scale.half().float()
            scale_st = scale.half().reshape(-1)
        off = h
        codes = ((xg / scale.clamp(min=1e-12)).round().clamp(-h, h - 1) + h).to(torch.uint8)
    return _ScalarTensor(pack_codes(codes, nb), scale_st, zero, step, kind, off, nb,
                         (G, xg.shape[1]), shape, t.dtype, rot)


def scalar_decompress(p: _ScalarTensor, signs: torch.Tensor) -> torch.Tensor:
    G, g = p.gshape
    codes = unpack_codes(p.q, p.nbits, G * g).reshape(G, g)
    if p.kind == 'bin':
        q = codes.float() * 2.0 - 1.0
    elif p.kind == 'sym':
        q = (codes.to(torch.int16) - p.off).float()
    else:
        q = codes.float()
    scale = p.scale.float().unsqueeze(1)
    if p.step is not None:
        scale = scale * p.step
    zero = p.zero.float().unsqueeze(1) if p.zero is not None else 0.0
    x = q * scale + zero
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


BATCHED_SCALAR_DECODE = True    # one unpack per layer and mode instead of one per entry (bit-identical)
USE_TCC_KERNEL = True           # fused Triton decode of a whole chain per layer (bit-identical)
try:
    from kernels.triton_kernels.decode_tcc import TRITON_AVAILABLE as _TCC_OK, tcc_decode
except Exception:               # pragma: no cover
    _TCC_OK = False


def _kernel_decode(entries, i: int, accumulate: bool, dst=None, n_slots=None):
    """Float32 reconstruction of layer ``i`` with the fused kernel, or None if the
    chain is not supported (mixed shapes, rotation, >2 modes ...).  Intra entries
    restart the reconstruction; with ``accumulate`` the others add to it
    (previous-frame predictor), otherwise they replace it.  Chain position j goes to
    output slot ``dst[j]`` (-1: not stored); the result is (N, n_slots, C)."""
    if not (USE_TCC_KERNEL and _TCC_OK):
        return None
    items = [e.items[i] for e in entries]
    if not all(e.mode.startswith(('int', 'bin')) for e in entries):
        return None
    p0 = items[0]
    if (not p0.q.is_cuda or any(p.rot or p.gshape[1] == 0 or p.nbits not in (1, 2, 4, 8)
                                or tuple(p.shape) != tuple(p0.shape) for p in items)):
        return None
    intra = [getattr(e, 'intra', True) for e in entries]
    modes_a = {e.mode for e, f in zip(entries, intra) if f}
    modes_b = {e.mode for e, f in zip(entries, intra) if not f}
    if len(modes_a) > 1 or len(modes_b) > 1 or not intra[0]:
        return None
    if dst is None:
        dst, n_slots = list(range(len(items))), len(items)
    C = p0.shape[-1]
    return tcc_decode(items, [not f for f in intra], dst, n_slots, math.prod(p0.shape) // C, C, accumulate)


def _read_scalar_batched(entries: List['CompressedEntry'], i: int) -> torch.Tensor:
    """Layer ``i`` of scalar-coded entries (any mix of int*/bin* modes), decoded with
    one unpack per mode.  Same arithmetic per element as ``scalar_decompress``."""
    p0 = entries[0].items[i]
    N_shape, C = p0.shape, p0.shape[-1]
    Fn = len(entries)
    out = torch.empty((Fn,) + tuple(N_shape), device=p0.q.device, dtype=p0.dtype)
    groups = {}
    for j, e in enumerate(entries):
        groups.setdefault(e.mode, []).append(j)
    for mode, idx in groups.items():
        ps = [entries[j].items[i] for j in idx]
        p = ps[0]
        G, g = p.gshape
        codes = unpack_codes(torch.cat([q.q for q in ps]), p.nbits, len(ps) * G * g).reshape(len(ps), G, g)
        if p.kind == 'bin':
            q = codes.float() * 2.0 - 1.0
        elif p.kind == 'sym':
            q = (codes.to(torch.int16) - p.off).float()
        else:
            q = codes.float()
        scale = torch.stack([s.scale for s in ps]).float().unsqueeze(-1)            # (F, G, 1)
        if p.step is not None:
            steps = torch.tensor([s.step for s in ps], device=scale.device, dtype=torch.float32)
            scale = scale * steps.view(-1, 1, 1)
        if p.zero is not None:
            x = q * scale + torch.stack([s.zero for s in ps]).float().unsqueeze(-1)
        else:
            x = q * scale + 0.0
        out[idx] = x.reshape((len(ps),) + tuple(N_shape)).to(p.dtype)
    # (F, N, 1, C) -> (N, F, C), as torch.cat([...], dim=1) of (N, 1, C) entries
    return out.squeeze(2).transpose(0, 1) if out.dim() == 4 else out.transpose(0, 1)


def read_layer(entries: List['CompressedEntry'], i: int) -> torch.Tensor:
    """Decompress layer ``i`` of several frames in one batch.

    Equivalent to ``torch.cat([e[i] for e in entries], dim=1)`` but issues one
    decode and one inverse rotation instead of one per frame.
    """
    first = entries[0]
    if all(getattr(e, 'intra', True) for e in entries):
        out = _kernel_decode(entries, i, accumulate=False)
        if out is not None:
            return out.to(first.items[i].dtype)
    if (BATCHED_SCALAR_DECODE and all(e.mode.startswith(('int', 'bin')) for e in entries)
            and not any(e.items[i].rot for e in entries)
            and len({tuple(e.items[i].shape) for e in entries}) == 1):
        return _read_scalar_batched(entries, i)
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


PREDICTORS = ('prev', 'none', 'extrap2', 'linfit', 'open')   # 'open': open-loop control


def _predict(predictor: str, r1: torch.Tensor, r2: Optional[torch.Tensor], params=None):
    """Prediction of the next frame from the latest reconstructions.

    r1 = rec[t-1], r2 = rec[t-2] (None when t-1 starts a keyframe segment).
    Returns None for a zero prediction.  Encoder and decoder both call this
    function (same op order, elementwise only), so their reconstructions are
    bit-identical whatever the tensor layout.
    """
    if predictor == 'prev':
        return r1
    if predictor == 'none':
        return None
    if r2 is None:
        return r1
    if predictor == 'extrap2':
        return 2.0 * r1 - r2
    a, b, c = params                                    # linfit: (C,) each
    return (a * r1 + b * r2) + c


def _params_on(params, device):
    return None if params is None else tuple(
        torch.as_tensor(x).to(device=device, dtype=torch.float32) for x in params)


def fit_linear_predictor(cache_sequences, ridge: float = 1e-3):
    """Fit per-layer, per-channel linear temporal predictors.

    ``cache_sequences``: list (over frames) of lists (over layers) of tensors
    (N, 1, C) from an uncompressed run.  For every layer and channel,
        rec_hat[t] = a * rec[t-1] + b * rec[t-2] + c
    is fitted over all positions N and all t >= 2.  The regression is on the
    residual from persistence, ``x[t]-x[t-1] ~ (a-1) x[t-1] + b x[t-2] + c``,
    with a relative ridge ``ridge * diag(Gram)`` so that the penalty shrinks
    towards the 'prev' predictor (a=1, b=0, c=0) independently of data scale.
    Returns a list over layers of (a, b, c) float32 CPU tensors of shape (C,),
    to be passed to ``enable_compressed_cache(..., predictor_params=...)``.
    """
    T, L = len(cache_sequences), len(cache_sequences[0])
    assert T >= 4, "need at least 4 calibration frames"
    out = []
    for l in range(L):
        C = cache_sequences[0][l].shape[-1]
        X = [cache_sequences[t][l].detach().double().reshape(-1, C) for t in range(T)]
        gram = torch.zeros(C, 3, 3, dtype=torch.float64, device=X[0].device)
        rhs = torch.zeros(C, 3, dtype=torch.float64, device=X[0].device)
        n = 0
        for t in range(2, T):
            x1, x2, y = X[t - 1], X[t - 2], X[t] - X[t - 1]
            f = torch.stack([x1, x2, torch.ones_like(x1)])          # (3, N, C)
            gram += torch.einsum('knc,lnc->ckl', f, f)
            rhs += torch.einsum('knc,nc->ck', f, y)
            n += x1.shape[0]
        gram, rhs = gram / n, rhs / n
        d = torch.diagonal(gram, dim1=-2, dim2=-1)                  # (C, 3)
        gram = gram + torch.diag_embed(ridge * d + 1e-12)
        w = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)  # (C, 3)
        out.append(((1.0 + w[:, 0]).float().cpu(), w[:, 1].float().cpu(),
                    w[:, 2].float().cpu()))
    return out


class DeltaLazyLayer(LazyLayer):
    """Layer cache under closed-loop delta coding.

    ``chain`` = hidden anchor prefix + frame_cache_list[1:], i.e. consecutive
    frames starting at an intra entry.  Residual entries are added to the
    prediction made from the previous reconstruction(s) (see ``_predict``);
    intra entries reset it and the predictor history.  The decoder repeats the
    encoder's operations, so reconstructions match exactly.
    """

    def __init__(self, first, chain, pos1, n_last, i, predictor: str = 'prev',
                 params=None):
        self.first, self.chain, self.pos1, self.n_last, self.i = first, chain, pos1, n_last, i
        self.predictor, self.params = predictor, params

    def materialize(self) -> torch.Tensor:
        assert getattr(self.chain[0], 'intra', True)
        if self.predictor in ('prev', 'none'):
            # slots: 0 = first frame, 1 = oldest window frame, 2.. = the last n_last frames
            n = len(self.chain)
            if self.pos1 < n - self.n_last:
                dst = [-1] * n
                dst[self.pos1] = 1
                dst[n - self.n_last:] = range(2, 2 + self.n_last)
                out = _kernel_decode([self.first] + list(self.chain), self.i,
                                     accumulate=self.predictor == 'prev',
                                     dst=[0] + dst, n_slots=2 + self.n_last)
                if out is not None:
                    return out
        vals = read_layer(self.chain, self.i).float()        # (N, len(chain), C)
        params = _params_on(self.params, vals.device)
        if BATCHED_SCALAR_DECODE and self.predictor in ('prev', 'none'):
            rec = self._reconstruct_batched(vals)
            f0 = self.first[self.i].float()
            return torch.cat([f0, rec[:, self.pos1:self.pos1 + 1], rec[:, -self.n_last:]], dim=1)
        rec = torch.empty_like(vals)
        prev_intra = True
        for j, e in enumerate(self.chain):
            if getattr(e, 'intra', True):
                rec[:, j] = vals[:, j]
                prev_intra = True
                continue
            r2 = None if prev_intra or j < 2 else rec[:, j - 2]
            pred = _predict(self.predictor, rec[:, j - 1], r2, params)
            rec[:, j] = vals[:, j] if pred is None else pred + vals[:, j]
            prev_intra = False
        f0 = self.first[self.i].float()
        return torch.cat([f0, rec[:, self.pos1:self.pos1 + 1], rec[:, -self.n_last:]], dim=1)

    def _reconstruct_batched(self, vals: torch.Tensor) -> torch.Tensor:
        """rec[j] = rec[j-1] + vals[j] inside each keyframe segment ('prev'), or
        rec = vals ('none'): the same additions as the loop above, done for all
        segments at once, one offset at a time."""
        if self.predictor == 'none':
            return vals.clone()
        intra = [getattr(e, 'intra', True) for e in self.chain]
        starts = [j for j, f in enumerate(intra) if f]
        ends = starts[1:] + [len(intra)]
        rec = vals.clone()
        for k in range(1, max(e - s for s, e in zip(starts, ends))):
            idx = [s + k for s, e in zip(starts, ends) if s + k < e]
            ix = torch.tensor(idx, device=vals.device)
            rec[:, ix] = rec[:, ix - 1] + vals[:, ix]
        return rec


class _DeltaEncoder:
    """Closed-loop encoder state for delta coding (one per streaming model).

    Keeps the last two reconstructions of every layer (``h1`` = rec[t-1],
    ``h2`` = rec[t-2] within the current keyframe segment, else None).
    """

    def __init__(self, delta: int, mode: str, residual_mode: str, bits: int,
                 predictor: str = 'prev', params=None):
        assert predictor in PREDICTORS, predictor
        assert predictor != 'linfit' or params is not None, "linfit needs predictor_params"
        self.delta, self.mode, self.residual_mode, self.bits = delta, mode, residual_mode, bits
        self.open_loop = predictor == 'open'
        self.predictor = 'prev' if self.open_loop else predictor
        self.params = params
        self.reset()

    def reset(self):
        self.h1, self.h2, self.since_intra = None, None, 0
        self.t1 = None              # true previous entry (open-loop control only)

    def encode(self, cache_list, signs) -> CompressedEntry:
        cur = [t.float() for t in cache_list]
        intra = self.h1 is None or self.since_intra >= self.delta - 1
        if intra:
            payload, pred = cur, None
        else:
            if self.params is not None:
                assert len(self.params) == len(cur), "predictor_params: one entry per layer"
            h2 = self.h2 if self.h2 is not None else [None] * len(cur)
            pred = [_predict(self.predictor, r1, r2,
                             None if self.params is None
                             else _params_on(self.params[i], r1.device))
                    for i, (r1, r2) in enumerate(zip(self.h1, h2))]
            if self.open_loop:      # residual against the true previous entry, which
                pred = self.t1      # the decoder does not have: errors are not corrected
            payload = [c if p is None else c - p for c, p in zip(cur, pred)]
        e = CompressedEntry(payload, self.mode if intra else self.residual_mode,
                            self.bits, signs)
        e.intra = intra
        dec = [e[i].float() for i in range(len(e))]
        if intra:
            rec, self.h2, self.since_intra = dec, None, 0
        else:
            base = self.h1 if self.open_loop else pred      # what the decoder adds to
            rec = [d if p is None else p + d for p, d in zip(base, dec)]
            self.h2, self.since_intra = self.h1, self.since_intra + 1
        self.h1 = rec
        self.t1 = cur
        return e


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
                            residual_mode: Optional[str] = None,
                            predictor: str = 'prev', predictor_params=None):
    """Patch a streaming VideoDepthAnything so its frame cache is compressed.

    Frame 0 runs VDA's own code (it reads no cache).  Later frames run a copy
    of VDA's streaming step whose only change is how the cache is read
    (batched decompression) and stored (compressed on insertion).

    delta=K>0 stores each frame as the quantized residual from a prediction
    built from previous reconstructions (closed loop), with an intra frame
    every K frames.  ``predictor`` selects the prediction:
      'prev'    rec[t-1] (default);
      'none'    0, i.e. every frame coded on its own (with ``residual_mode``)
                but through the delta path (ablation);
      'extrap2' 2*rec[t-1] - rec[t-2] (rec[t-1] right after a keyframe);
      'linfit'  per-layer, per-channel a*rec[t-1] + b*rec[t-2] + c with
                ``predictor_params`` = ``fit_linear_predictor(...)`` output
                (rec[t-1] right after a keyframe).
    The decoder (``DeltaLazyLayer``) applies the same predictor to its own
    reconstructions; keyframes reset the history.

    kv=False caches each frame's attention input h (VDA's design: K/V of all
    31 cached frames are recomputed every frame).  kv=True caches the
    projected W_k h and W_v h instead (2x the entries, no recomputation).
    """
    if mode == 'fp32' and not kv and not delta:
        return model
    assert not (kv and delta), "delta coding is implemented for the h cache only"
    assert predictor in PREDICTORS, predictor
    assert predictor == 'prev' or delta, "predictor != 'prev' needs delta > 0"
    assert predictor != 'linfit' or predictor_params is not None, \
        "predictor='linfit' needs predictor_params (see fit_linear_predictor)"
    assert mode in ('fp32', 'fp16', 'bw16', 'zero') or mode.startswith(('int', 'bin')), mode
    residual_mode = residual_mode or mode
    import torch.nn.functional as F
    from video_depth_anything import video_depth_stream as vds
    INFER_LEN = vds.INFER_LEN

    original = model.infer_video_depth_one
    state = {'signs': None}
    enc = (_DeltaEncoder(delta, mode, residual_mode, bits, predictor, predictor_params)
           if delta else None)
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
        return enc.encode(cache_list, state['signs'])

    def read(cur_list):
        n = len(cur_list[0])
        if kv:
            return [KVPair(read_layer(cur_list, 2 * i), read_layer(cur_list, 2 * i + 1))
                    for i in range(n // 2)]
        if delta:
            L, H = model.frame_cache_list, model._delta_hidden
            chain = list(H) + list(L[1:])
            return [DeltaLazyLayer(L[0], chain, len(H), INFER_LEN - 3, i, 'prev' if predictor == 'open' else predictor,
                                   None if predictor_params is None else predictor_params[i])
                    for i in range(n)]
        if lazy:
            return [LazyLayer(cur_list, i) for i in range(n)]
        return [read_layer(cur_list, i) for i in range(n)]

    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        if state['signs'] is None:
            state['signs'] = _signs(seed, next(self.parameters()).device)
        if self.transform is None and enc is not None:
            enc.reset()
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
