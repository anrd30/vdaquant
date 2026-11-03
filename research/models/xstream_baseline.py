"""
XStreamVGGT-style KV-cache baseline (arXiv 2601.01204) and its combination
with temporal delta coding, for StreamVGGT's global-attention layers.

XStreamVGGT ("Extremely Memory-Efficient Streaming VGGT with KV Cache
Compression") is tuning-free.  Per global-attention layer it keeps the cache
at a fixed token budget L_max and quantizes it:

  * Pruning (paper S3.2, Eqs. 2-6).  Whenever |cache| + |current frame| exceeds
    L_max the cache is pruned.  The first frame and the current frame are always
    kept; the remaining k = max(0, L_max - T_first - T_current) tokens are the
    top-k "middle" tokens by an importance score: current-frame queries
    (camera/register queries kept as is, patch queries average-pooled in groups
    of g=16, then averaged over heads) dotted with head-averaged middle keys,
    averaged over the pooled queries.  One index set is shared by all heads.
  * Quantization (paper S3.3, Eqs. 7-10).  KIVI-style asymmetric uniform INT4,
    group size 64; keys per-channel (groups run over tokens), values per-token
    (groups run over channels).  Applied to the whole post-pruning cache
    including first and current frame; no full-precision recent window.

Details the paper does not state were taken from the authors' released code
(github.com/ywh187/XStreamVGGT, attention.py, read via a text summary):
  * attention of the current frame runs over the FULL (unpruned) cache plus the
    current K/V; pruning only decides what is carried to the next step;
  * "first N_new" and "last N_new" cache entries are protected (N_new = tokens
    of the current frame), the first 5 queries are unpooled "special" tokens;
  * the quantization is simulated there; here the codes are really bit-packed.

RoPE.  This repo's StreamVGGT applies RoPE to the cached raw K *after* reading
the cache, with pos repeated per cached frame.  Once tokens are pruned that
repeat is wrong, so these caches store the 2-D position of every kept token and
the patched forward (xstream_forward) passes the matching positions to RoPE.

Frame count.  The aggregator derives the number of cached frames from
past_key_values[0][0].shape[2]; these caches keep an explicit frame counter
(n_frames) and report it as shape[2] regardless of how many tokens survive.

Modes (see streamvggt_cache.parse):
  xstream:<budget>:<kbits>:<vbits>[:<group>[:<pool>]]
        budget in tokens ('inf' = never prune), bits in {2,4,8,16}; 16 = fp16
  xstream+delta:<budget>:<keykind>:<reskind>:<K>
        same pruning, but kept tokens are stored with the closed-loop temporal
        delta code of streamvggt_cache (see XStreamDeltaKV).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from research.models import streamvggt_cache as kc
from research.models.streamvggt_cache import (GROUP, CompressedKV, _ideal_bits,
                                              _pack, _store_bits, _unpack, _LIVE)

INF_BUDGET = 10 ** 12


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------
def _budget(s: str) -> int:
    return INF_BUDGET if s in ('inf', 'none') else int(s)


def parse_xstream(p: List[str]) -> dict:
    """['xstream', budget, kbits, vbits, (group), (pool)]"""
    return {'mode': 'xstream', 'budget': _budget(p[1]), 'kbits': int(p[2]),
            'vbits': int(p[3]), 'group': int(p[4]) if len(p) > 4 else 64,
            'pool': int(p[5]) if len(p) > 5 else 16, 'n_special': 5}


def parse_xstream_delta(p: List[str]) -> dict:
    """['xstream+delta', budget, keykind, reskind, K]"""
    return {'mode': 'xstream+delta', 'budget': _budget(p[1]), 'key': p[2],
            'res': p[3], 'K': int(p[4]), 'pool': 16, 'n_special': 5}


# --------------------------------------------------------------------------
# Importance score (paper Eqs. 2-6)
# --------------------------------------------------------------------------
@torch.no_grad()
def importance_scores(q: torch.Tensor, k_mid: torch.Tensor, pool: int = 16,
                      n_special: int = 5) -> torch.Tensor:
    """q: (B, h, N, d) current-frame queries; k_mid: (B, h, M, d) prunable keys
    (both after q/k-norm and RoPE).  Returns (B, M) scores."""
    B, h, N, d = q.shape
    ns = min(n_special, N)
    special, normal = q[:, :, :ns], q[:, :, ns:]
    parts = [special]
    L = normal.shape[2]
    if L > 1:
        full = (L // pool) * pool
        if full > 0:
            parts.append(normal[:, :, :full].reshape(B, h, full // pool, pool, d).mean(3))
        if L - full > 0:                      # leftover tokens pooled into one extra token
            parts.append(normal[:, :, full:].mean(2, keepdim=True))
    elif L == 1:
        parts.append(normal)
    q_pool = torch.cat(parts, 2).mean(1)      # head average -> (B, Nq, d)
    k_bar = k_mid.mean(1)                     # head average -> (B, M, d)
    return torch.einsum('bqd,bmd->bqm', q_pool.float(), k_bar.float()).mean(1)


def select_keep(scores: torch.Tensor, n_first: int, n_old: int, n_cur: int,
                budget: int) -> Optional[torch.Tensor]:
    """Indices (sorted, into the old cache) to keep, or None if nothing is pruned.
    The first n_first old tokens are always kept; of the middle region
    [n_first, n_old) the top-k by score are kept, k = max(0, budget - n_first - n_cur).
    Selection is shared across the batch (scores averaged over it)."""
    if n_old + n_cur <= budget:
        return None
    k = max(0, budget - n_first - n_cur)
    n_mid = n_old - n_first
    first = torch.arange(n_first, device=scores.device)
    if k >= n_mid:
        return None
    if k == 0:
        return first
    top = scores.mean(0).topk(k).indices.sort().values + n_first
    return torch.cat([first, top])


# --------------------------------------------------------------------------
# KIVI-style asymmetric quantization (per-channel K, per-token V)
# --------------------------------------------------------------------------
def _aq(x: torch.Tensor, bits: int):
    """x: (..., G) -> asymmetric uniform codes over the last dim.  Parameters are
    stored as fp16 (min, scale) and the codes are computed against the rounded
    fp16 parameters so decode is exact w.r.t. what is stored."""
    mn = x.amin(-1, keepdim=True)
    mx = x.amax(-1, keepdim=True)
    scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8)
    mn16, s16 = mn.half(), scale.half()
    s = s16.float().clamp(min=6e-8)
    q = ((x - mn16.float()) / s).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
    return q, mn16, s16


class _KiviTensor:
    """One quantized (B, h, T, d) tensor.  axis='token': groups of `group` tokens
    per channel (keys).  axis='channel': groups of `group` channels per token (values)."""

    def __init__(self, x: torch.Tensor, bits: int, group: int, axis: str):
        self.shape = tuple(x.shape)
        self.bits, self.group, self.axis = bits, group, axis
        B, h, T, d = x.shape
        if bits == 16:
            self.raw = x.half()
            return
        x = x.float()
        if axis == 'token':
            Tp = -(-T // group) * group
            if Tp > T:                                       # edge-replicate: keeps min/max honest
                x = torch.cat([x, x[:, :, -1:].expand(B, h, Tp - T, d)], 2)
            xg = x.reshape(B, h, Tp // group, group, d).transpose(3, 4)     # (B,h,Tg,d,G)
        else:
            g = min(group, d)
            assert d % g == 0
            xg = x.reshape(B, h, T, d // g, g)
        self.gshape = tuple(xg.shape)
        q, self.mn, self.sc = _aq(xg, bits)
        self.w = _store_bits('int%d' % bits) if bits != 8 else 8
        self.packed = _pack(q.reshape(-1), self.w)

    def dequant(self) -> torch.Tensor:
        if self.bits == 16:
            return self.raw.float()
        n = 1
        for s in self.gshape:
            n *= s
        q = _unpack(self.packed, self.w, n).reshape(self.gshape).float()
        xg = q * self.sc.float().clamp(min=6e-8) + self.mn.float()
        B, h, T, d = self.shape
        if self.axis == 'token':
            x = xg.transpose(3, 4).reshape(B, h, -1, d)[:, :, :T]
        else:
            x = xg.reshape(B, h, T, d)
        return x

    def nbytes(self) -> int:
        if self.bits == 16:
            return self.raw.numel() * 2
        return self.packed.numel() + (self.mn.numel() + self.sc.numel()) * 2

    def scalars(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def ideal_bits_total(self) -> float:
        if self.bits == 16:
            return 16.0 * self.scalars()
        n = 1
        for s in self.gshape:
            n *= s
        return n * self.bits + (self.mn.numel() + self.sc.numel()) * 16.0


# --------------------------------------------------------------------------
# Shared cache base
# --------------------------------------------------------------------------
class _PrunedKVBase(CompressedKV):
    """Per-layer cache with a fixed token budget.  Exposes the same surface the
    aggregator / launch scripts use: .shape, [i], iteration, .frames (len =
    number of frames seen), nbytes(), ideal_bits()."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_frames = 0
        self.n_first = 0                 # tokens of frame 0 (always protected)
        self.frame_shape = None          # (B, h, 1, N, d)
        _LIVE.add(self)

    @property
    def frames(self):                    # only len() is used by callers
        return range(self.n_frames)

    @property
    def shape(self):
        B, h, _, N, d = self.frame_shape
        return (B, h, self.n_frames, N, d)

    @property
    def n_tokens(self) -> int:
        raise NotImplementedError

    def append(self, *a, **kw):
        raise RuntimeError('pruned caches are updated through update()')

    def decode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update(self, k, v, pos, keep, dec):
        raise NotImplementedError

    def materialize(self):
        raise RuntimeError('use decode(): pruned caches have no (B,h,S,N,d) layout')


def _pos_to_store(pos: torch.Tensor) -> torch.Tensor:
    return pos[:1].to(torch.int16)                       # identical across the batch


# --------------------------------------------------------------------------
# XStreamVGGT: prune + KIVI INT4
# --------------------------------------------------------------------------
class XStreamKV(_PrunedKVBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.kq: Optional[_KiviTensor] = None
        self.vq: Optional[_KiviTensor] = None
        self.pos: Optional[torch.Tensor] = None          # (1, T, 2) int16
        self.fid = self.tid = None                       # (T,) provenance, for tests/analysis
        self._B = 1

    @property
    def n_tokens(self) -> int:
        return 0 if self.kq is None else self.kq.shape[2]

    def decode(self):
        return (self.kq.dequant(), self.vq.dequant(),
                self.pos.long().expand(self._B, -1, -1))

    def update(self, k, v, pos, keep, dec):
        cfg = self.cfg
        B, h, N, d = k.shape
        self._B = B
        self.frame_shape = (B, h, 1, N, d)
        fid_c = torch.full((N,), self.n_frames, dtype=torch.int32, device=k.device)
        tid_c = torch.arange(N, dtype=torch.int16, device=k.device)
        k, v = k.float(), v.float()
        if dec is None:
            K, V, P, fid, tid = k, v, _pos_to_store(pos), fid_c, tid_c
            self.n_first = N
        else:
            pk, pv, _ = dec
            if keep is not None:
                pk, pv = pk[:, :, keep], pv[:, :, keep]
                po, fo, to = self.pos[:, keep], self.fid[keep], self.tid[keep]
            else:
                po, fo, to = self.pos, self.fid, self.tid
            K, V = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            P = torch.cat([po, _pos_to_store(pos)], 1)
            fid, tid = torch.cat([fo, fid_c]), torch.cat([to, tid_c])
        self.kq = _KiviTensor(K, cfg['kbits'], cfg['group'], 'token')
        self.vq = _KiviTensor(V, cfg['vbits'], cfg['group'], 'channel')
        self.pos, self.fid, self.tid = P, fid, tid
        self.n_frames += 1

    def nbytes(self) -> int:
        if self.kq is None:
            return 0
        return self.kq.nbytes() + self.vq.nbytes() + self.pos.numel() * 2

    def ideal_bits(self) -> float:
        if self.kq is None:
            return 16.0
        tot = self.kq.ideal_bits_total() + self.vq.ideal_bits_total()
        return tot / (self.kq.scalars() + self.vq.scalars())


# --------------------------------------------------------------------------
# Row-wise symmetric group-16 quantizer with per-row fp16 step (used by the
# delta variant so that rows can be gathered / concatenated independently)
# --------------------------------------------------------------------------
class RowStore:
    """Quantized rows of shape (b, h, R, d); every (b,h,row) is coded independently:
    d/16 groups, 8-bit scale index per group, one fp16 step per row."""

    def __init__(self, kind, packed, sidx, step):
        self.kind, self.packed, self.sidx, self.step = kind, packed, sidx, step

    @property
    def R(self) -> int:
        return self.packed.shape[2]

    @staticmethod
    def empty(kind, b, h, d, device):
        w = _store_bits(kind)
        return RowStore(kind, torch.zeros(b, h, 0, d * w // 8, dtype=torch.uint8, device=device),
                        torch.zeros(b, h, 0, d // GROUP, dtype=torch.uint8, device=device),
                        torch.zeros(b, h, 0, dtype=torch.float16, device=device))

    @staticmethod
    def encode(x: torch.Tensor, kind: str) -> 'RowStore':
        b, h, n, d = x.shape
        if n == 0:
            return RowStore.empty(kind, b, h, d, x.device)
        xg = x.float().reshape(b, h, n, d // GROUP, GROUP)
        if kind == 'bin':
            scale = xg.abs().mean(-1, keepdim=True)
        else:
            hh = 2 ** (int(kind[3:]) - 1)
            scale = xg.abs().amax(-1, keepdim=True) / max(hh - 1, 1)
        step = (scale.amax((-2, -1), keepdim=True) / 255.0).clamp(min=1e-7)
        step = step.half().float().clamp(min=6e-8)
        sidx = (scale / step).round().clamp(0, 255)
        sq = sidx * step
        if kind == 'bin':
            codes = (xg >= 0).to(torch.uint8)
        else:
            q = (xg / sq.clamp(min=1e-12)).round().clamp(-hh, hh - 1)
            codes = (q + hh).to(torch.uint8)
        w = _store_bits(kind)
        packed = _pack(codes.reshape(-1), w).reshape(b, h, n, -1)
        return RowStore(kind, packed, sidx.to(torch.uint8).reshape(b, h, n, -1),
                        step.reshape(b, h, n).half())

    def decode(self, d: int) -> torch.Tensor:
        b, h, R = self.packed.shape[:3]
        if R == 0:
            return torch.zeros(b, h, 0, d, device=self.packed.device)
        w = _store_bits(self.kind)
        codes = _unpack(self.packed, w, d).float().reshape(b, h, R, d // GROUP, GROUP)
        scale = (self.sidx.float() * self.step.float().unsqueeze(-1)).unsqueeze(-1)
        if self.kind == 'bin':
            vals = (codes * 2 - 1) * scale
        else:
            vals = (codes - 2 ** (int(self.kind[3:]) - 1)) * scale
        return vals.reshape(b, h, R, d)

    def take(self, idx: torch.Tensor) -> 'RowStore':
        return RowStore(self.kind, self.packed[:, :, idx], self.sidx[:, :, idx], self.step[:, :, idx])

    @staticmethod
    def cat(stores: List['RowStore']) -> 'RowStore':
        s0 = stores[0]
        return RowStore(s0.kind, torch.cat([s.packed for s in stores], 2),
                        torch.cat([s.sidx for s in stores], 2), torch.cat([s.step for s in stores], 2))

    def nbytes(self) -> int:
        return self.packed.numel() + self.sidx.numel() + self.step.numel() * 2

    def ideal_bits_total(self, d: int) -> float:
        n = self.packed.shape[0] * self.packed.shape[1] * self.R
        return n * (d * _ideal_bits(self.kind) + (d // GROUP) * 8 + 16)


class _DeltaState:
    """Tokens in ascending (frame, token) order.  fid/tid/is_key/row are (T,);
    row indexes into kst (key-coded rows) or dst (residual-coded rows).  K and V
    share the stores by being stacked on the leading dim (2B, h, R, d)."""

    def __init__(self, fid, tid, is_key, row, kst, dst, pos):
        self.fid, self.tid, self.is_key, self.row = fid, tid, is_key, row
        self.kst, self.dst, self.pos = kst, dst, pos

    @property
    def T(self) -> int:
        return self.fid.shape[0]

    def keys(self, P: int) -> torch.Tensor:
        return self.fid.long() * P + self.tid.long()

    def refs(self, P: int):
        """index of each token's temporal reference (frame-1, same token) in this
        state, and whether it exists."""
        keys = self.keys(P)
        tgt = (self.fid.long() - 1) * P + self.tid.long()
        idx = torch.searchsorted(keys, tgt).clamp(max=max(self.T - 1, 0))
        ok = (self.fid > 0) & (keys[idx] == tgt) if self.T else (self.fid > 0)
        return idx, ok

    def decode(self, d: int, P: int, Kmax: int) -> torch.Tensor:
        """-> (2B, h, T, d) fp32 closed-loop reconstruction."""
        b, h = self.kst.packed.shape[:2]
        T = self.T
        dev = self.fid.device
        recon = torch.zeros(b, h, T, d, device=dev)
        kidx = self.is_key.nonzero().squeeze(1)
        didx = (~self.is_key).nonzero().squeeze(1)
        if kidx.numel():
            recon[:, :, kidx] = self.kst.decode(d)[:, :, self.row[kidx].long()]
        if didx.numel():
            res = torch.zeros(b, h, T, d, device=dev)
            res[:, :, didx] = self.dst.decode(d)[:, :, self.row[didx].long()]
            ref, ok = self.refs(P)
            assert bool(ok[didx].all()), 'delta token without reference'
            done = self.is_key.clone()
            pending = ~self.is_key
            for _ in range(Kmax + 1):
                ready = (pending & done[ref]).nonzero().squeeze(1)
                if ready.numel() == 0:
                    break
                recon[:, :, ready] = recon[:, :, ref[ready]] + res[:, :, ready]
                done[ready] = True
                pending[ready] = False
            assert not bool(pending.any()), 'unresolved delta chain'
        return recon


# --------------------------------------------------------------------------
# xstream+delta: XStreamVGGT pruning + temporal delta coding of the kept tokens
# --------------------------------------------------------------------------
class XStreamDeltaKV(_PrunedKVBase):
    """Pruning exactly as XStreamVGGT (same score, budget, protected tokens).  The
    kept tokens are stored with the closed-loop temporal delta code of
    streamvggt_cache ('key' kind for keyframes, 'res' kind for residuals).

    Why this is not a plain per-frame delta code: after pruning, frame t's cache
    holds an arbitrary subset of its tokens and frame t-1 holds a different one,
    so a token is only delta-codable if its same-index token of frame t-1 is
    still in the cache (the decoder needs that reconstruction).  Rules:
      * the current frame's token i is a residual against the reconstruction of
        (t-1, i) iff (t-1, i) survives this step's pruning and t % K != 0;
        otherwise it is keyframe-coded (key kind);
      * a stored residual whose reference is pruned later is re-coded as a
        keyframe from its current reconstruction ("orphan conversion"; one extra
        quantization at key precision; its own descendants then see a bounded
        perturbation equal to that re-quantization error).
    Chains are at most K-1 residuals long because t % K == 0 forces a keyframe."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.st: Optional[_DeltaState] = None
        self._B = 1
        self._d = None
        self._P = None
        self.n_converted = 0                   # statistics: orphan conversions

    @property
    def n_tokens(self) -> int:
        return 0 if self.st is None else self.st.T

    @property
    def fid(self):
        return self.st.fid

    @property
    def tid(self):
        return self.st.tid

    def decode(self):
        rec = self.st.decode(self._d, self._P, self.cfg['K'])
        B = self._B
        return rec[:B], rec[B:], self.st.pos.long().expand(B, -1, -1)

    def _stack(self, k, v):
        return torch.cat([k.float(), v.float()], 0)           # (2B, h, n, d)

    def update(self, k, v, pos, keep, dec):
        cfg = self.cfg
        B, h, N, d = k.shape
        self._B, self._d, self._P = B, d, N
        self.frame_shape = (B, h, 1, N, d)
        dev = k.device
        key_kind, res_kind, Kmax = cfg['key'], cfg['res'], cfg['K']
        t = self.n_frames
        cur = self._stack(k, v)
        fid_c = torch.full((N,), t, dtype=torch.int32, device=dev)
        tid_c = torch.arange(N, dtype=torch.int16, device=dev)
        pos_c = _pos_to_store(pos)

        if dec is None:
            self.n_first = N
            kst = RowStore.encode(cur, key_kind)
            dst = RowStore.empty(res_kind, 2 * B, h, d, dev)
            self.st = _DeltaState(fid_c, tid_c, torch.ones(N, dtype=torch.bool, device=dev),
                                  torch.arange(N, dtype=torch.int32, device=dev), kst, dst, pos_c)
            self.n_frames += 1
            return

        st = self.st
        T0 = st.T
        keep = torch.arange(T0, device=dev) if keep is None else keep
        # ---- 1. prune: gather kept tokens, keep their stored codes as they are
        fid, tid, is_key, row = st.fid[keep], st.tid[keep], st.is_key[keep], st.row[keep].long()
        pos_o = st.pos[:, keep]
        tmp = _DeltaState(fid, tid, is_key, row, st.kst, st.dst, pos_o)
        ref, ok = tmp.refs(N)
        orphan = (~is_key) & (~ok)
        # ---- 2. orphan conversion: residual whose reference was pruned -> keyframe
        kst_rows = st.kst.take(row[is_key]) if is_key.any() else RowStore.empty(key_kind, 2 * B, h, d, dev)
        dst_rows = st.dst.take(row[~is_key & ok]) if (~is_key & ok).any() else RowStore.empty(res_kind, 2 * B, h, d, dev)
        n_conv = int(orphan.sum())
        # new compact row numbering: [old key rows (in token order)] + [converted] , [delta rows]
        order_key = is_key.nonzero().squeeze(1)
        order_del = (~is_key & ok).nonzero().squeeze(1)
        order_conv = orphan.nonzero().squeeze(1)
        new_row = torch.zeros(keep.numel(), dtype=torch.long, device=dev)
        new_row[order_key] = torch.arange(order_key.numel(), device=dev)
        new_row[order_del] = torch.arange(order_del.numel(), device=dev)
        if n_conv:
            recon_all = dec_stack = torch.cat([dec[0], dec[1]], 0)[:, :, keep]
            conv_store = RowStore.encode(recon_all[:, :, order_conv], key_kind)
            new_row[order_conv] = order_key.numel() + torch.arange(n_conv, device=dev)
            kst_rows = RowStore.cat([kst_rows, conv_store])
            is_key = is_key.clone()
            is_key[order_conv] = True
            self.n_converted += n_conv
        st1 = _DeltaState(fid, tid, is_key, new_row.to(torch.int32), kst_rows, dst_rows, pos_o)
        # ---- 3. exact closed-loop reconstruction of the pruned state
        if n_conv:
            recon = st1.decode(d, N, Kmax)
        else:
            recon = torch.cat([dec[0], dec[1]], 0)[:, :, keep]
        # ---- 4. encode the current frame
        keys1 = st1.keys(N)
        tgt = (t - 1) * N + tid_c.long()
        idx = torch.searchsorted(keys1, tgt).clamp(max=keys1.numel() - 1)
        ref_ok = keys1[idx] == tgt
        use_delta = ref_ok & (t % Kmax != 0)
        d_idx = use_delta.nonzero().squeeze(1)
        k_idx = (~use_delta).nonzero().squeeze(1)
        n_k0, n_d0 = st1.kst.R, st1.dst.R
        new_k = RowStore.encode(cur[:, :, k_idx], key_kind)
        new_d = RowStore.encode(cur[:, :, d_idx] - recon[:, :, idx[d_idx]], res_kind)
        row_c = torch.zeros(N, dtype=torch.long, device=dev)
        row_c[k_idx] = n_k0 + torch.arange(k_idx.numel(), device=dev)
        row_c[d_idx] = n_d0 + torch.arange(d_idx.numel(), device=dev)
        self.st = _DeltaState(torch.cat([fid, fid_c]), torch.cat([tid, tid_c]),
                              torch.cat([is_key, use_delta.logical_not()]),
                              torch.cat([st1.row.long(), row_c]).to(torch.int32),
                              RowStore.cat([st1.kst, new_k]), RowStore.cat([st1.dst, new_d]),
                              torch.cat([pos_o, pos_c], 1))
        self.n_frames += 1

    def nbytes(self) -> int:
        if self.st is None:
            return 0
        s = self.st
        meta = (s.fid.numel() * 4 + s.tid.numel() * 2 + s.is_key.numel() + s.row.numel() * 4)
        return s.kst.nbytes() + s.dst.nbytes() + s.pos.numel() * 2 + meta

    def ideal_bits(self) -> float:
        if self.st is None:
            return 16.0
        s, d = self.st, self._d
        tot = s.kst.ideal_bits_total(d) + s.dst.ideal_bits_total(d)
        cnt = (s.kst.packed.shape[0] * s.kst.packed.shape[1]) * s.T * d
        return tot / max(cnt, 1)

    def key_fraction(self) -> float:
        return float(self.st.is_key.float().mean()) if self.st is not None else 1.0


def make_cache(cfg: dict) -> _PrunedKVBase:
    return XStreamKV(cfg) if cfg['mode'] == 'xstream' else XStreamDeltaKV(cfg)


# --------------------------------------------------------------------------
# Patched Attention.forward for the pruned caches
# --------------------------------------------------------------------------
def xstream_forward(self, x, pos, attn_mask, past_key_values, cfg):
    """StreamVGGT Attention.forward (use_cache=True) with a pruned cache.  Same
    maths as the original, except (a) the cached tokens may be a subset, with their
    own stored positions fed to RoPE, and (b) the cache is pruned/re-encoded after
    the attention of the current frame."""
    import torch.nn.functional as F
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                                        # (B, h, N, d)
    cache = past_key_values if isinstance(past_key_values, _PrunedKVBase) else make_cache(cfg)
    pos_store = pos if pos is not None else torch.zeros(B, N, 2, dtype=torch.long, device=x.device)
    if cache.n_tokens:
        dec = cache.decode()
        pk, pv, ppos = dec
        k_all = torch.cat([pk.to(k.dtype), k], 2)
        v_all = torch.cat([pv.to(v.dtype), v], 2)
        pos_k = torch.cat([ppos.to(pos.dtype), pos], 1) if pos is not None else None
    else:
        dec = None
        k_all, v_all, pos_k = k, v, pos
    qn, kn = self.q_norm(q), self.k_norm(k_all)
    if self.rope is not None:
        qn = self.rope(qn, pos)
        kn = self.rope(kn, pos_k)
    # ---- pruning decision (uses the same normed/roped q and k as the attention)
    keep = None
    if dec is not None:
        n_old = cache.n_tokens
        with torch.no_grad():
            if n_old + N > cfg['budget'] and n_old > cache.n_first:
                s = importance_scores(qn, kn[:, :, cache.n_first:n_old], cfg['pool'], cfg['n_special'])
            else:
                s = None
        keep = select_keep(s, cache.n_first, n_old, N, cfg['budget']) if s is not None else None
    # ---- attention over the full (unpruned) cache + current frame
    if self.fused_attn:
        out = F.scaled_dot_product_attention(qn, kn, v_all, attn_mask=attn_mask,
                                             dropout_p=self.attn_drop.p if self.training else 0.0)
    else:
        attn = (qn * self.scale) @ kn.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        out = self.attn_drop(attn.softmax(dim=-1)) @ v_all
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj_drop(self.proj(out))
    # ---- store the current frame's raw (pre-norm, pre-RoPE) k/v like the original
    cache.update(k.detach(), v.detach(), pos_store, keep, dec)
    return out, cache
