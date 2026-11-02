"""
Compressed, delta-coded KV cache for StreamVGGT (streaming VGGT).

StreamVGGT's 24 global-attention layers cache the raw projected K and V of
every past frame (shape (B, heads, S, N, d), never evicted), so memory grows
linearly with the number of frames.  This module replaces that cache with a
bit-packed store:

  plain : every frame quantized independently (per-group scalar, 8-bit scale)
  delta : closed-loop temporal predictive coding.  Every K-th frame is a
          keyframe (e.g. INT3); other frames store the quantized residual
          against the previous frame's *reconstruction* (e.g. 1-bit sign +
          per-group scale), so quantization error does not accumulate.

No rotation and no lattice: per-group scalar quantization only.

Usage (before the model is built):
    from research.models.streamvggt_cache import install
    install("delta:int3:bin:8")        # or "plain:int2", "fp16", "none"
"""
from __future__ import annotations

import os
import types
import weakref
from typing import List, Optional

import torch

GROUP = 16

# --------------------------------------------------------------------------
# Bit-packed per-group scalar codes
# --------------------------------------------------------------------------


def _store_bits(kind: str) -> int:
    """Bits per code actually allocated (packable widths 1, 2, 4)."""
    if kind == 'bin':
        return 1
    b = int(kind[3:])
    return {2: 2, 3: 4, 4: 4}[b]


def _ideal_bits(kind: str) -> int:
    return 1 if kind == 'bin' else int(kind[3:])


def _pack(codes: torch.Tensor, w: int) -> torch.Tensor:
    """codes: uint8 in [0, 2^w) -> uint8 with 8/w codes per byte."""
    per = 8 // w
    c = codes.reshape(-1, per).to(torch.int32)
    shifts = (torch.arange(per, device=c.device, dtype=torch.int32) * w)
    return (c << shifts).sum(-1).to(torch.uint8)


def _unpack(packed: torch.Tensor, w: int, n: int) -> torch.Tensor:
    per = 8 // w
    shifts = (torch.arange(per, device=packed.device, dtype=torch.int32) * w)
    c = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << w) - 1)
    return c.reshape(*packed.shape[:-1], -1)[..., :n]


def quantize(x: torch.Tensor, kind: str) -> dict:
    """Per-group (16) symmetric scalar quantization with an 8-bit scale.
    kind: 'int2' | 'int3' | 'int4' | 'bin'."""
    shape = tuple(x.shape)
    xg = x.float().reshape(-1, GROUP)
    if kind == 'bin':
        scale = xg.abs().mean(-1, keepdim=True)          # L2-optimal 1-bit scale
        codes = (xg >= 0).to(torch.uint8)
    else:
        b = int(kind[3:])
        h = 2 ** (b - 1)
        scale = xg.abs().amax(-1, keepdim=True) / max(h - 1, 1)
    step = scale.max().clamp(min=1e-12) / 255.0
    sidx = (scale / step).round().clamp(0, 255)
    scale_q = sidx * step
    if kind != 'bin':
        q = (xg / scale_q.clamp(min=1e-12)).round().clamp(-h, h - 1)
        codes = (q + h).to(torch.uint8)
    w = _store_bits(kind)
    return {'packed': _pack(codes.reshape(-1), w), 'sidx': sidx.to(torch.uint8).reshape(-1),
            'step': step.reshape(1), 'shape': shape, 'kind': kind, 'n': xg.numel()}


def dequantize_batch(items: List[dict]) -> torch.Tensor:
    """Decode several codes of the same kind/shape at once -> (len, *shape) fp32."""
    kind, shape, n = items[0]['kind'], items[0]['shape'], items[0]['n']
    w = _store_bits(kind)
    codes = _unpack(torch.stack([it['packed'] for it in items]), w, n).float()
    codes = codes.reshape(len(items), -1, GROUP)
    scale = (torch.stack([it['sidx'] for it in items]).float()
             * torch.stack([it['step'] for it in items])).unsqueeze(-1)
    if kind == 'bin':
        vals = (codes * 2 - 1) * scale
    else:
        vals = (codes - 2 ** (int(kind[3:]) - 1)) * scale
    return vals.reshape(len(items), *shape)


def _nbytes(item: dict) -> int:
    return item['packed'].numel() + item['sidx'].numel() + 4


# --------------------------------------------------------------------------
# Per-layer cache
# --------------------------------------------------------------------------
_LIVE = weakref.WeakSet()


class _ShapeProxy:
    def __init__(self, shape):
        self.shape = shape


class CompressedKV:
    """Stands in for StreamVGGT's (past_k, past_v) tuple of one layer."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.frames: List[dict] = []
        self.frame_shape = None     # (B, h, 1, N, d)
        _LIVE.add(self)

    # aggregator reads past_key_values[0][0].shape and unpacks k, v
    @property
    def shape(self):
        B, h, _, N, d = self.frame_shape
        return (B, h, len(self.frames), N, d)

    def __getitem__(self, i):
        return _ShapeProxy(self.shape)

    def __iter__(self):
        return iter((_ShapeProxy(self.shape), _ShapeProxy(self.shape)))

    def append(self, k: torch.Tensor, v: torch.Tensor, prev=None):
        """prev: decoder's fp32 reconstruction (k, v) of the newest cached frame,
        shape (B, h, 1, N, d).  Using the decoder's own reconstruction keeps
        encoder and decoder in exact closed loop and needs no extra state."""
        cfg = self.cfg
        self.frame_shape = tuple(k.shape)
        if cfg['mode'] == 'fp16':
            self.frames.append({'key': True, 'k': k.half(), 'v': v.half()})
            return
        idx = len(self.frames)
        key = cfg['mode'] == 'plain' or prev is None or idx % cfg['K'] == 0
        kind = cfg['key'] if key else cfg['res']
        cur = (k.float(), v.float())
        payload = cur if key else (cur[0] - prev[0], cur[1] - prev[1])
        self.frames.append({'key': key, 'k': quantize(payload[0], kind),
                            'v': quantize(payload[1], kind)})

    def materialize(self):
        """All cached frames, (B, h, S, N, d) each, fp32."""
        if self.cfg['mode'] == 'fp16':
            k = torch.cat([f['k'] for f in self.frames], 2).float()
            v = torch.cat([f['v'] for f in self.frames], 2).float()
            return k, v
        S = len(self.frames)
        out = []
        for which in ('k', 'v'):
            vals = torch.empty((S,) + self.frame_shape, dtype=torch.float32,
                               device=self.frames[0][which]['packed'].device)
            groups = {}
            for i, f in enumerate(self.frames):
                groups.setdefault(f[which]['kind'], []).append(i)
            for kind, ids in groups.items():
                vals[ids] = dequantize_batch([self.frames[i][which] for i in ids])
            if self.cfg['mode'] == 'delta':
                # recon_t = sum of decoded values since the last keyframe <= t
                is_key = torch.tensor([f['key'] for f in self.frames], device=vals.device)
                pos = torch.arange(S, device=vals.device)
                last_key = torch.cummax(torch.where(is_key, pos, torch.zeros_like(pos)), 0).values
                csum = torch.cumsum(vals, 0)
                prev = torch.where((last_key > 0).view(-1, *[1] * (vals.dim() - 1)),
                                   csum[(last_key - 1).clamp(min=0)], torch.zeros_like(csum))
                vals = csum - prev
            # (S, B, h, 1, N, d) -> (B, h, S, N, d)
            out.append(vals.squeeze(3).permute(1, 2, 0, 3, 4))
        return out[0], out[1]

    def nbytes(self) -> int:
        if self.cfg['mode'] == 'fp16':
            return sum(f['k'].numel() * 2 + f['v'].numel() * 2 for f in self.frames)
        return sum(_nbytes(f['k']) + _nbytes(f['v']) for f in self.frames)

    def ideal_bits(self) -> float:
        """Bits per cached scalar for an ideally packed layout (incl. scales)."""
        if self.cfg['mode'] == 'fp16':
            return 16.0
        tot, cnt = 0.0, 0
        for f in self.frames:
            for which in ('k', 'v'):
                it = f[which]
                tot += it['n'] * _ideal_bits(it['kind']) + it['sidx'].numel() * 8
                cnt += it['n']
        return tot / max(cnt, 1)


def total_cache_bytes() -> int:
    return sum(c.nbytes() for c in list(_LIVE))


# --------------------------------------------------------------------------
# Patching StreamVGGT
# --------------------------------------------------------------------------


def parse(spec: str) -> Optional[dict]:
    """'none' | 'fp16' | 'plain:int2' | 'delta:int3:bin:8'"""
    p = spec.split(':')
    if p[0] == 'none':
        return None
    if p[0] == 'fp16':
        return {'mode': 'fp16'}
    if p[0] == 'plain':
        return {'mode': 'plain', 'key': p[1]}
    if p[0] == 'delta':
        return {'mode': 'delta', 'key': p[1], 'res': p[2], 'K': int(p[3])}
    raise ValueError(spec)


def _patched_forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
    """StreamVGGT Attention.forward with a CompressedKV cache.  Identical to
    the original except for how past K/V are stored and read."""
    import torch.nn.functional as F
    cfg = getattr(self, '_kv_cfg', None)
    if cfg is None or not use_cache:
        return self._orig_forward(x, pos=pos, attn_mask=attn_mask,
                                  past_key_values=past_key_values, use_cache=use_cache)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    k = k.unsqueeze(2)
    v = v.unsqueeze(2)
    cache = past_key_values if isinstance(past_key_values, CompressedKV) else CompressedKV(cfg)
    if cache.frames:
        pk, pv = cache.materialize()
        prev = (pk[:, :, -1:], pv[:, :, -1:])
        k_all = torch.cat([pk.to(k.dtype), k], 2)
        v_all = torch.cat([pv.to(v.dtype), v], 2)
        del pk, pv
    else:
        prev = None
        k_all, v_all = k, v
    cache.append(k, v, prev)
    a, b, c, d, e = k_all.shape
    k = k_all.reshape(a, b, c * d, e)
    v = v_all.reshape(a, b, c * d, e)
    pos_k = pos.repeat(1, c, 1) if pos is not None else None
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos_k)
    if self.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.attn_drop.p if self.training else 0.0)
    else:
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        x = self.attn_drop(attn.softmax(dim=-1)) @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj_drop(self.proj(x))
    return x, cache


def install(spec: Optional[str] = None):
    """Patch StreamVGGT so the aggregator's global-attention caches are compressed.
    Must be called after StreamVGGT/src is importable and before the model is built."""
    spec = spec or os.environ.get('KV_SPEC', 'none')
    cfg = parse(spec)
    from streamvggt.layers import attention as att
    from streamvggt.models import streamvggt as sv
    if not hasattr(att.Attention, '_orig_forward'):
        att.Attention._orig_forward = att.Attention.forward
        att.Attention.forward = _patched_forward
    orig_init = sv.StreamVGGT.__init__
    if getattr(orig_init, '_kv_wrapped', False):
        orig_init = orig_init.__wrapped__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        for blk in self.aggregator.global_blocks:
            blk.attn._kv_cfg = cfg
    __init__._kv_wrapped = True
    __init__.__wrapped__ = orig_init
    sv.StreamVGGT.__init__ = __init__
    print(f"[kv-cache] StreamVGGT global-attention cache: {spec}", flush=True)
    return cfg
