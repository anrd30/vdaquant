"""
Compressed, delta-coded KV cache for StreamVGGT (streaming VGGT).

StreamVGGT's 24 global-attention layers cache the raw projected K and V of
every past frame (shape (B, heads, S, N, d), never evicted), so memory grows
linearly with the number of frames.  This module replaces that cache with a
bit-packed store:

  plain : every frame quantized independently (per-group scalar, 8-bit scale,
          or KIVI-style asymmetric with fp16 scale and zero: 'a<bits>g<group>')
  delta : closed-loop temporal predictive coding.  Every K-th frame is a
          keyframe (e.g. INT3); other frames store the quantized residual
          against the previous frame's *reconstruction* (e.g. 1-bit sign +
          per-group scale), so quantization error does not accumulate.

No rotation and no lattice: per-group scalar quantization only.

Usage (before the model is built):
    from research.models.streamvggt_cache import install
    install("delta:int4:bin:8")        # TCC-1
    install("delta:int4:a2g64:8")      # TCC-2
    install("plain:a2g64")             # asymmetric INT2 baseline; or "fp16", "none"
    install("kivi:2:64")               # keys per channel, values per token
    install("plain:a2g64:w4")          # newest 4 frames kept at 16 bits
    install("drop:2:int4")             # keep every 2nd frame, INT4
    install("delta:int4:bin:8:s2")     # TCC-1 on every 2nd frame (drop + TCC)
    install("delta:int4:bin:8:a0.5")   # extra keyframe when the residual is large
    install("zero")                    # zero-cache control: past keys/values = 0
    install("offload")                 # exact cache in host memory: a reference
                                       # for videos too long for the GPU

Set kc.TRACE = True to record residual statistics per layer in cache.trace.
"""
from __future__ import annotations

import os
import types
import weakref
from typing import List, Optional

import torch

GROUP = 16
CHUNK = 32      # frames decoded at a time when reading a cache
TRACE = False   # record residual statistics in CompressedKV.trace

# --------------------------------------------------------------------------
# Bit-packed per-group scalar codes
# --------------------------------------------------------------------------


def _asym(kind: str):
    """'a2g64' -> (bits=2, group=64); None for symmetric kinds."""
    if not kind.startswith('a'):
        return None
    b, g = kind[1:].split('g')
    return int(b), int(g)


def _store_bits(kind: str) -> int:
    """Bits per code actually allocated (packable widths 1, 2, 4)."""
    if kind == 'bin':
        return 1
    b = _asym(kind)[0] if _asym(kind) else int(kind[3:])
    return {2: 2, 3: 4, 4: 4}[b]


def _ideal_bits(kind: str) -> int:
    if kind == 'raw':
        return 16
    kind = kind.lstrip('c')
    if kind == 'bin':
        return 1
    return _asym(kind)[0] if _asym(kind) else int(kind[3:])


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
    kind: 'int2' | 'int3' | 'int4' | 'bin', or 'a<bits>g<group>' for asymmetric
    quantization with fp16 scale and zero, as int<bits>-g<group>-asym in
    stream_cache.scalar_compress."""
    shape = tuple(x.shape)
    if _asym(kind):
        bits, g = _asym(kind)
        xg = x.float().reshape(-1, g)
        mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8).half().float()
        zero = mn.half().float()
        codes = ((xg - zero) / scale).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
        return {'packed': _pack(codes.reshape(-1), _store_bits(kind)),
                'scale': scale.half().reshape(-1), 'zero': zero.half().reshape(-1),
                'shape': shape, 'kind': kind, 'n': xg.numel()}
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


def quantize_channels(x: torch.Tensor, kind: str) -> dict:
    """KIVI-style key quantization of one frame (B, h, 1, N, d): groups run along
    the tokens of each channel.  N is padded with its last token to a whole number
    of groups."""
    g = _asym(kind)[1]
    t = x.transpose(-1, -2)
    pad = -t.shape[-1] % g
    if pad:
        t = torch.cat([t, t[..., -1:].expand(*t.shape[:-1], pad)], -1)
    item = quantize(t.contiguous(), kind)
    item.update(kind='c' + kind, tokens=x.shape[-2])
    return item


def raw_item(x: torch.Tensor) -> dict:
    """A frame kept at 16 bits inside a quantized cache."""
    return {'t': x.half(), 'kind': 'raw', 'n': x.numel()}


def dequantize_batch(items: List[dict]) -> torch.Tensor:
    """Decode several codes of the same kind/shape at once -> (len, *shape) fp32."""
    if items[0]['kind'] == 'raw':
        return torch.stack([it['t'] for it in items]).float()
    if items[0]['kind'].startswith('c'):
        vals = dequantize_batch([dict(it, kind=it['kind'][1:]) for it in items])
        return vals[..., :items[0]['tokens']].transpose(-1, -2)
    kind, shape, n = items[0]['kind'], items[0]['shape'], items[0]['n']
    w = _store_bits(kind)
    codes = _unpack(torch.stack([it['packed'] for it in items]), w, n).float()
    if _asym(kind):
        codes = codes.reshape(len(items), -1, _asym(kind)[1])
        scale = torch.stack([it['scale'] for it in items]).float().unsqueeze(-1)
        zero = torch.stack([it['zero'] for it in items]).float().unsqueeze(-1)
        return (codes * scale + zero).reshape(len(items), *shape)
    codes = codes.reshape(len(items), -1, GROUP)
    scale = (torch.stack([it['sidx'] for it in items]).float()
             * torch.stack([it['step'] for it in items])).unsqueeze(-1)
    if kind == 'bin':
        vals = (codes * 2 - 1) * scale
    else:
        vals = (codes - 2 ** (int(kind[3:]) - 1)) * scale
    return vals.reshape(len(items), *shape)


def _meta_bits(item: dict) -> int:
    """Bits of per-group side information."""
    if item['kind'] == 'raw':
        return 0
    if 'sidx' in item:
        return item['sidx'].numel() * 8
    return (item['scale'].numel() + item['zero'].numel()) * 16


def _nbytes(item: dict) -> int:
    if item['kind'] == 'raw':
        return item['t'].numel() * 2
    extra = 4 if 'sidx' in item else 0          # fp32 step of the 8-bit scales
    return item['packed'].numel() + _meta_bits(item) // 8 + extra


# --------------------------------------------------------------------------
# Per-layer cache
# --------------------------------------------------------------------------
class _Live(weakref.WeakSet):
    """Live caches.  StreamVGGT.inference drops its caches when it returns, so
    set .pin to a list to keep this inference's caches alive for reporting."""
    pin = None

    def add(self, item):
        super().add(item)
        if self.pin is not None:
            self.pin.append(item)


_LIVE = _Live()


class _ShapeProxy:
    def __init__(self, shape):
        self.shape = shape


class CompressedKV:
    """Stands in for StreamVGGT's (past_k, past_v) tuple of one layer."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.frames: List[dict] = []
        self.frame_shape = None     # (B, h, 1, N, d)
        self.seen = 0               # frames offered, including dropped ones
        self.trace = None
        self.true_prev = None
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
        self.seen += 1
        if (self.seen - 1) % cfg.get('stride', 1):
            return
        if cfg['mode'] == 'fp16':
            self.frames.append({'key': True, 'k': k.half(), 'v': v.half()})
            return
        if cfg['mode'] == 'offload':
            self.frames.append({'key': True, 'k': k.cpu(), 'v': v.cpu()})
            return
        if cfg['mode'] == 'zero':
            self.frames.append({'key': True})
            return
        window = cfg.get('window', 0)
        if window:
            self.frames.append({'key': True, 'k': raw_item(k), 'v': raw_item(v)})
            if len(self.frames) > window:
                old = self.frames[-window - 1]
                for which in ('k', 'v'):
                    old[which] = self._encode(old[which]['t'].float(), cfg['key'], which)
            return
        idx = len(self.frames)
        key = cfg['mode'] == 'plain' or prev is None or idx % cfg['K'] == 0
        kind = cfg['key'] if key else cfg['res']
        cur = (k.float(), v.float())
        payload = cur if key else (cur[0] - prev[0], cur[1] - prev[1])
        if not key and cfg.get('adaptive') and any(
                float(r.pow(2).sum() / (c - c.mean()).pow(2).sum().clamp(min=1e-20)) > cfg['adaptive']
                for r, c in zip(payload, cur)):
            key, kind, payload = True, cfg['key'], cur      # scene change: new keyframe
        self.frames.append({'key': key, 'k': self._encode(payload[0], kind, 'k'),
                            'v': self._encode(payload[1], kind, 'v')})
        if TRACE:
            self._record(cur, payload, key)

    def _encode(self, x: torch.Tensor, kind: str, which: str) -> dict:
        if which == 'k' and self.cfg.get('chan_k'):
            return quantize_channels(x, kind)
        return quantize(x, kind)

    def _record(self, cur, payload, key: bool):
        """Per stream: energy kept by the residual quantizer, and how the residual
        and the coding error compare with the true frame-to-frame change."""
        if self.trace is None:
            self.trace = {w: {'frames': 0, 'quant_err_ratio': 0.0, 'quant_err_ratio_max': 0.0,
                              'change_ratio': 0.0, 'err_vs_change': 0.0} for w in 'kv'}
        for i, which in enumerate('kv'):
            h = cur[i]
            if not key and self.true_prev is not None:
                r = payload[i]
                e = r - dequantize_batch([self.frames[-1][which]])[0]
                change = (h - self.true_prev[i]).pow(2).sum()
                ratio = float(e.pow(2).sum() / r.pow(2).sum().clamp(min=1e-20))
                t = self.trace[which]
                t['frames'] += 1
                t['quant_err_ratio'] += ratio
                t['quant_err_ratio_max'] = max(t['quant_err_ratio_max'], ratio)
                t['change_ratio'] += float(change / (h - h.mean()).pow(2).sum().clamp(min=1e-20))
                t['err_vs_change'] += float(e.pow(2).sum() / change.clamp(min=1e-20))
        self.true_prev = cur

    def materialize(self):
        """All cached frames, (B, h, S, N, d) each, fp32."""
        if self.cfg['mode'] == 'zero':
            z = torch.zeros(self.shape)
            return z, z
        if self.cfg['mode'] in ('fp16', 'offload'):
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

    def _read_one(self, which: str, cur: torch.Tensor):
        """Past frames plus the current one in cur's dtype, (B, h, S+1, N, d), and
        the newest past frame's fp32 reconstruction.  Decodes CHUNK frames at a
        time, so fp32 temporaries do not grow with the length of the video."""
        S = len(self.frames)
        B, h, _, N, d = self.frame_shape
        dst = torch.empty((B, h, S + 1, N, d), dtype=cur.dtype, device=cur.device)
        dst[:, :, S:S + 1] = cur
        run = None
        for s0 in range(0, S, CHUNK):
            part = self.frames[s0:s0 + CHUNK]
            n = len(part)
            if self.cfg['mode'] == 'fp16':
                vals = torch.stack([f[which] for f in part]).float()
            elif self.cfg['mode'] == 'offload':
                vals = torch.stack([f[which] for f in part]).to(cur.device)
            elif self.cfg['mode'] == 'zero':
                vals = torch.zeros((n,) + self.frame_shape, dtype=cur.dtype, device=cur.device)
            else:
                vals = torch.empty((n,) + self.frame_shape, dtype=torch.float32, device=cur.device)
                groups = {}
                for i, f in enumerate(part):
                    groups.setdefault(f[which]['kind'], []).append(i)
                for kind, ids in groups.items():
                    vals[ids] = dequantize_batch([part[i][which] for i in ids])
            if self.cfg['mode'] == 'delta':
                # recon_t = reconstruction carried in + decoded values since the last keyframe
                is_key = torch.tensor([f['key'] for f in part], device=vals.device)
                if run is not None and not part[0]['key']:
                    vals[0] += run
                pos = torch.arange(n, device=vals.device)
                last_key = torch.cummax(torch.where(is_key, pos, torch.zeros_like(pos)), 0).values
                csum = torch.cumsum(vals, 0)
                base = torch.where((last_key > 0).view(-1, *[1] * (vals.dim() - 1)),
                                   csum[(last_key - 1).clamp(min=0)], torch.zeros_like(csum))
                vals = csum - base
                del csum, base
            run = vals[-1]
            dst[:, :, s0:s0 + n] = vals.squeeze(3).permute(1, 2, 0, 3, 4)
        return dst, run

    def read(self, k: torch.Tensor, v: torch.Tensor):
        """Cached frames with the current k, v appended, and the newest cached
        frame's fp32 reconstruction for closed-loop encoding."""
        k_all, pk = self._read_one('k', k)
        v_all, pv = self._read_one('v', v)
        return k_all, v_all, (pk, pv)

    def nbytes(self) -> int:
        if self.cfg['mode'] == 'zero':
            return 0
        if self.cfg['mode'] in ('fp16', 'offload'):
            return sum(f['k'].numel() * 2 + f['v'].numel() * 2 for f in self.frames)
        return sum(_nbytes(f['k']) + _nbytes(f['v']) for f in self.frames)

    def ideal_bits(self) -> float:
        """Bits per cached scalar for an ideally packed layout (incl. scales)."""
        if self.cfg['mode'] == 'zero':
            return 0.0
        if self.cfg['mode'] in ('fp16', 'offload'):
            return 16.0
        tot, cnt = 0.0, 0
        for f in self.frames:
            for which in ('k', 'v'):
                it = f[which]
                tot += it['n'] * _ideal_bits(it['kind']) + _meta_bits(it)
                cnt += it['n']
        return tot / max(cnt, 1)


def total_cache_bytes() -> int:
    return sum(c.nbytes() for c in list(_LIVE))


# --------------------------------------------------------------------------
# Patching StreamVGGT
# --------------------------------------------------------------------------


def parse(spec: str) -> Optional[dict]:
    """'none' | 'fp16' | 'offload' | 'plain:int2[:w<recent>]' | 'kivi:2:64[:w<recent>]' |
    'drop:<stride>:int4' | 'delta:int3:bin:8' | 'xstream:2048:4:4' |
    'xstream+delta:2048:int3:bin:8'"""
    p = spec.split(':')
    if p[0] == 'none':
        return None
    if p[0] == 'fp16':
        return {'mode': 'fp16'}
    if p[0] == 'offload':
        return {'mode': 'offload'}
    if p[0] == 'zero':
        return {'mode': 'zero'}
    if p[0] == 'plain':
        cfg = {'mode': 'plain', 'key': p[1]}
        if len(p) > 2:
            cfg['window'] = int(p[2][1:])
        return cfg
    if p[0] == 'kivi':             # kivi:<bits>:<group>[:w<recent>]
        cfg = {'mode': 'plain', 'key': f'a{p[1]}g{p[2]}', 'chan_k': True}
        if len(p) > 3:
            cfg['window'] = int(p[3][1:])
        return cfg
    if p[0] == 'drop':             # drop:<stride>:<kind>
        return {'mode': 'plain', 'key': p[2], 'stride': int(p[1])}
    if p[0] == 'delta':            # delta:<key>:<res>:<K>[:s<stride>][:a<threshold>]
        cfg = {'mode': 'delta', 'key': p[1], 'res': p[2], 'K': int(p[3])}
        for opt in p[4:]:
            if opt[0] == 's':
                cfg['stride'] = int(opt[1:])
            elif opt[0] == 'a':
                cfg['adaptive'] = float(opt[1:])
            else:
                raise ValueError(spec)
        return cfg
    if p[0] == 'xstream':          # xstream:<budget>:<kbits>:<vbits>[:<group>[:<pool>]]
        from research.models.xstream_baseline import parse_xstream
        return parse_xstream(p)
    if p[0] == 'xstream+delta':    # xstream+delta:<budget>:<keykind>:<reskind>:<K>
        from research.models.xstream_baseline import parse_xstream_delta
        return parse_xstream_delta(p)
    raise ValueError(spec)


def _patched_forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
    """StreamVGGT Attention.forward with a CompressedKV cache.  Identical to
    the original except for how past K/V are stored and read."""
    import torch.nn.functional as F
    cfg = getattr(self, '_kv_cfg', None)
    if cfg is None or not use_cache:
        return self._orig_forward(x, pos=pos, attn_mask=attn_mask,
                                  past_key_values=past_key_values, use_cache=use_cache)
    if cfg['mode'] in ('xstream', 'xstream+delta'):
        from research.models.xstream_baseline import xstream_forward
        return xstream_forward(self, x, pos, attn_mask, past_key_values, cfg)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    k = k.unsqueeze(2)
    v = v.unsqueeze(2)
    cache = past_key_values if isinstance(past_key_values, CompressedKV) else CompressedKV(cfg)
    if cache.frames:
        k_all, v_all, prev = cache.read(k, v)
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
