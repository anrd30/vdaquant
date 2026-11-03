"""
CPU tests for research/models/streamvggt_cache.py and xstream_baseline.py.

Builds StreamVGGT's real Attention module (and its 2-D RoPE) from source and
streams correlated random frames with use_cache=True through the original
forward and through each patched cache mode.

Run: pytest tests/test_streamvggt_cache.py -q
"""
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG_SRC = os.environ.get('STREAMVGGT_SRC', 'C:/Users/aniru/StreamVGGT/src')
if not os.path.isdir(SVG_SRC):
    pytest.skip('StreamVGGT source not available', allow_module_level=True)
sys.path.insert(0, ROOT)
sys.path.insert(0, SVG_SRC)

from streamvggt.layers.attention import Attention  # noqa: E402
from streamvggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D  # noqa: E402

from research.models import streamvggt_cache as kc  # noqa: E402
from research.models import xstream_baseline as xb  # noqa: E402

DIM, HEADS = 128, 2
GH, GW, NSPEC = 5, 7, 5
N = GH * GW + NSPEC                      # 40 tokens / frame
FRAMES = 20


def make_attn(rope: bool, seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    a = Attention(dim=DIM, num_heads=HEADS, qk_norm=True,
                  rope=RotaryPositionEmbedding2D() if rope else None).eval()
    a._orig_forward = types.MethodType(Attention.forward, a)
    return a


def make_pos(B=1):
    """Same construction as the aggregator: specials at 0, patches at grid+1."""
    p = PositionGetter()(B, GH, GW, torch.device('cpu')) + 1
    return torch.cat([torch.zeros(B, NSPEC, 2, dtype=p.dtype), p], 1)


def make_frames(n=FRAMES, B=1, seed=1, drift=0.15):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, DIM, generator=g)
    out = []
    for _ in range(n):
        x = x + drift * torch.randn(B, N, DIM, generator=g)
        out.append(x.clone())
    return out


@torch.no_grad()
def stream_original(attn, frames, pos):
    cache, outs = None, []
    for x in frames:
        o, cache = Attention.forward(attn, x, pos=pos, past_key_values=cache, use_cache=True)
        outs.append(o)
    return torch.stack(outs), cache


@torch.no_grad()
def stream_mode(attn, frames, pos, spec, track=None):
    attn._kv_cfg = kc.parse(spec)
    cache, outs = None, []
    for i, x in enumerate(frames):
        o, cache = kc._patched_forward(attn, x, pos=pos, past_key_values=cache, use_cache=True)
        outs.append(o)
        if track is not None:
            track.append((cache.shape, cache.nbytes(), cache.n_tokens if hasattr(cache, 'n_tokens') else None))
    return torch.stack(outs), cache


def raw_kv(attn, x):
    B, n, _ = x.shape
    qkv = attn.qkv(x).reshape(B, n, 3, HEADS, attn.head_dim).permute(2, 0, 3, 1, 4)
    return qkv[1], qkv[2]


def orig_bytes_fp16(cache_shape):
    B, h, S, n, d = cache_shape
    return 2 * B * h * S * n * d * 2


def relerr(a, b):
    return float((a - b).norm() / b.norm())


# ----------------------------------------------------------------------------
# existing modes
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('rope', [False, True])
def test_fp16_matches_original(rope):
    attn = make_attn(rope)
    pos = make_pos() if rope else None
    frames = make_frames()
    ref, rc = stream_original(attn, frames, pos)
    out, cache = stream_mode(attn, frames, pos, 'fp16')
    assert (out - ref).abs().max() < 1e-3
    assert cache.shape == tuple(rc[0].shape)


@pytest.mark.parametrize('rope', [False, True])
@pytest.mark.parametrize('spec', ['plain:int4', 'plain:int2', 'delta:int4:bin:8', 'delta:int3:bin:4'])
def test_plain_and_delta_run(spec, rope):
    attn = make_attn(rope)
    pos = make_pos() if rope else None
    frames = make_frames()
    ref, rc = stream_original(attn, frames, pos)
    out, cache = stream_mode(attn, frames, pos, spec)
    assert out.shape == ref.shape and torch.isfinite(out).all()
    fp16 = orig_bytes_fp16(cache.shape)
    assert 0 < cache.nbytes() < fp16 / 2            # real memory saving
    assert cache.ideal_bits() < 8.0
    assert len(cache.frames) == FRAMES
    if spec == 'plain:int4':
        assert relerr(out, ref) < 0.5


def test_delta_cheaper_than_plain_same_keyframes():
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames(drift=0.05)
    _, plain = stream_mode(attn, frames, pos, 'plain:int3')
    _, delta = stream_mode(attn, frames, pos, 'delta:int3:bin:8')
    assert delta.nbytes() < plain.nbytes()


# ----------------------------------------------------------------------------
# parse
# ----------------------------------------------------------------------------
def test_parse_xstream_specs():
    c = kc.parse('xstream:2048:4:4')
    assert c['mode'] == 'xstream' and c['budget'] == 2048 and c['kbits'] == 4 and c['vbits'] == 4
    assert c['group'] == 64 and c['pool'] == 16
    c = kc.parse('xstream+delta:4096:int3:bin:8')
    assert (c['mode'], c['budget'], c['key'], c['res'], c['K']) == ('xstream+delta', 4096, 'int3', 'bin', 8)
    assert kc.parse('xstream:inf:16:16')['budget'] >= 10 ** 9
    assert kc.parse('none') is None and kc.parse('fp16') == {'mode': 'fp16'}


# ----------------------------------------------------------------------------
# XStreamVGGT
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('rope', [False, True])
def test_xstream_no_prune_fp16_equals_original(rope):
    """budget=inf, 16-bit storage: positions/RoPE bookkeeping must reproduce the original."""
    attn = make_attn(rope)
    pos = make_pos() if rope else None
    frames = make_frames()
    ref, rc = stream_original(attn, frames, pos)
    out, cache = stream_mode(attn, frames, pos, 'xstream:inf:16:16')
    assert (out - ref).abs().max() < 1e-3
    assert cache.shape == tuple(rc[0].shape) and cache.n_tokens == FRAMES * N


def test_xstream_int8_no_prune_close():
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames()
    ref, _ = stream_original(attn, frames, pos)
    out, _ = stream_mode(attn, frames, pos, 'xstream:inf:8:8')
    assert relerr(out, ref) < 0.05


@pytest.mark.parametrize('spec', ['xstream:120:16:16', 'xstream:120:4:4', 'xstream+delta:120:int4:int2:6'])
def test_xstream_memory_bounded(spec):
    attn = make_attn(True)
    pos = make_pos()
    track = []
    _, cache = stream_mode(attn, make_frames(), pos, spec, track)
    toks = [t[2] for t in track]
    nb = [t[1] for t in track]
    # grows while below budget, then constant at exactly the budget
    assert toks[:3] == [N, 2 * N, 3 * N]
    assert all(t == 120 for t in toks[2:]), toks
    if spec.startswith('xstream:'):
        assert len(set(nb[2:])) == 1, nb              # constant bytes once the budget is hit
    else:
        assert max(nb[2:]) <= 1.35 * nb[2], nb        # delta: only the key/residual mix varies
    assert nb[-1] < nb[2] * 1.35
    # ... whereas the frame counter keeps counting
    assert cache.shape[2] == FRAMES == len(cache.frames)


def test_xstream_budget_scaling_vs_unbounded():
    attn = make_attn(True)
    pos = make_pos()
    _, c_pruned = stream_mode(attn, make_frames(), pos, 'xstream:200:4:4')
    _, c_full = stream_mode(attn, make_frames(), pos, 'xstream:inf:4:4')
    assert c_pruned.nbytes() * 3 < c_full.nbytes()


@pytest.mark.parametrize('spec', ['xstream:120:16:16', 'xstream+delta:120:int4:int4:4'])
def test_xstream_protected_tokens_and_positions(spec):
    """First and current frame are kept in full; every kept token carries the
    right 2-D position and (fp16 storage) the right raw K/V of its source frame."""
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames()
    attn._kv_cfg = kc.parse(spec)
    cache = None
    exact = spec == 'xstream:120:16:16'
    with torch.no_grad():
        for t, x in enumerate(frames):
            _, cache = kc._patched_forward(attn, x, pos=pos, past_key_values=cache, use_cache=True)
            fid, tid = cache.fid.long(), cache.tid.long()
            assert (fid == 0).sum() == N and (fid == t).sum() == N      # first + current frame complete
            assert bool((fid[1:] >= fid[:-1]).all())                    # chronological
    K, V, P = cache.decode()
    assert P.shape == (1, K.shape[2], 2)
    # positions travel with their tokens
    assert torch.equal(P[0], pos[0][tid])
    # kept tokens really come from several different frames (middle was sampled, not a window)
    assert len(set(fid.tolist())) >= 3
    # K of the first and current frame is the raw projection of that frame
    for f in (0, FRAMES - 1):
        rk, rv = raw_kv(attn, frames[f])
        m = (fid == f)
        order = tid[m]
        if exact:
            assert torch.allclose(K[0, :, m], rk[0][:, order], atol=2e-3)
            assert torch.allclose(V[0, :, m], rv[0][:, order], atol=2e-3)
        else:
            assert relerr(K[0, :, m], rk[0][:, order]) < 0.25


def test_xstream_rope_pos_repeat_would_be_wrong():
    """Design check: the original `pos.repeat(1, c, 1)` is only valid when every
    cached frame is complete.  After pruning, kept tokens must use their own stored
    positions, which differ from the naive repeated layout."""
    attn = make_attn(True)
    pos = make_pos()
    _, cache = stream_mode(attn, make_frames(), pos, 'xstream:120:16:16')
    stored = cache.decode()[2][0]
    naive = pos.repeat(1, cache.n_tokens // N, 1)[0]
    assert stored.shape == naive.shape
    assert not torch.equal(stored, naive)
    assert torch.equal(stored, pos[0][cache.tid.long()])


def test_importance_scores_select_aligned_tokens():
    torch.manual_seed(0)
    B, h, Nq, d = 1, 2, 40, 16
    q = torch.randn(B, h, Nq, d)
    mid = torch.randn(B, h, 30, d) * 0.1
    target = q.mean((1, 2), keepdim=True)                  # direction the pooled queries share
    for i in (3, 11, 25):
        mid[:, :, i] += 5 * target[:, :, 0]
    s = xb.importance_scores(q, mid, pool=16, n_special=5)
    assert s.shape == (1, 30)
    assert set(s[0].topk(3).indices.tolist()) == {3, 11, 25}


def test_select_keep_rules():
    s = torch.arange(10.0).unsqueeze(0)                    # 10 middle tokens, score = index
    # under budget -> no pruning
    assert xb.select_keep(s, 4, 14, 4, budget=100) is None
    # budget 4+3+4 = 11: keep first 4 + top-3 middle (indices 7,8,9 -> +4) sorted
    keep = xb.select_keep(s, 4, 14, 4, budget=11)
    assert keep.tolist() == [0, 1, 2, 3, 11, 12, 13]
    # budget smaller than first+current: only the protected first frame survives
    assert xb.select_keep(s, 4, 14, 4, budget=5).tolist() == [0, 1, 2, 3]


def test_xstream_quant_bytes_match_kivi_accounting():
    attn = make_attn(True)
    pos = make_pos()
    _, cache = stream_mode(attn, make_frames(8), pos, 'xstream:inf:4:4')
    T = cache.n_tokens                                     # 320 tokens = 5 groups of 64
    B, h, d = 1, HEADS, attn.head_dim
    codes = 2 * B * h * T * d // 2                         # 4 bits
    kparams = 2 * B * h * (T // 64) * d * 2
    vparams = 2 * B * h * T * (d // 64) * 2
    assert cache.nbytes() == codes + kparams + vparams + T * 2 * 2
    assert 4.4 < cache.ideal_bits() < 5.2


# ----------------------------------------------------------------------------
# xstream + delta
# ----------------------------------------------------------------------------
def test_xstream_delta_closed_loop_and_conversion():
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames(drift=0.05)
    out, cache = stream_mode(attn, frames, pos, 'xstream+delta:160:int4:int2:6')
    st = cache.st
    assert cache.n_tokens == 160 and torch.isfinite(out).all()
    # every residual token has its reference in the cache (decode asserts this too)
    ref, ok = st.refs(N)
    assert bool(ok[~st.is_key].all())
    # chains never longer than K-1 residuals
    assert 0 < cache.key_fraction() <= 1.0
    assert cache.n_converted > 0                           # pruning orphaned some residuals
    # reconstruction of every kept token is close to the true K of its frame
    K, V, _ = cache.decode()
    fid, tid = st.fid.long(), st.tid.long()
    errs = []
    for f in fid.unique().tolist():
        rk, _ = raw_kv(attn, frames[f])
        m = fid == f
        errs.append(relerr(K[0, :, m], rk[0][:, tid[m]]))
    assert max(errs) < 0.6, errs
    ref_out, _ = stream_original(attn, frames, pos)
    assert relerr(out, ref_out) < 1.0


def test_xstream_delta_uses_residuals_and_saves_bytes():
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames(drift=0.03)
    _, c_d = stream_mode(attn, frames, pos, 'xstream+delta:200:int3:bin:8')
    _, c_x = stream_mode(attn, frames, pos, 'xstream:200:4:4')
    assert c_d.n_tokens == c_x.n_tokens == 200
    assert c_d.key_fraction() < 1.0                        # some tokens are delta-coded
    assert c_d.ideal_bits() < c_x.ideal_bits()
    assert c_d.nbytes() < c_x.nbytes()


def test_xstream_delta_big_budget_equals_plain_delta_quality():
    """With no pruning, the xstream+delta cache keeps every token and (apart from the
    per-row fp16 step) behaves like the plain delta code: errors stay bounded."""
    attn = make_attn(True)
    pos = make_pos()
    frames = make_frames(drift=0.05)
    ref, _ = stream_original(attn, frames, pos)
    out_x, c = stream_mode(attn, frames, pos, 'xstream+delta:inf:int4:int2:8')
    out_d, _ = stream_mode(attn, frames, pos, 'delta:int4:int2:8')
    assert c.n_tokens == FRAMES * N and c.n_converted == 0
    assert abs(relerr(out_x, ref) - relerr(out_d, ref)) < 0.1


def test_batch2_shapes():
    attn = make_attn(True)
    pos = make_pos(B=2)
    frames = make_frames(8, B=2)
    for spec in ['xstream:100:4:4', 'xstream+delta:100:int3:bin:4']:
        out, cache = stream_mode(attn, frames, pos, spec)
        assert out.shape == (8, 2, N, DIM)
        assert cache.shape == (2, HEADS, 8, N, attn.head_dim)
        assert cache[0].shape == cache.shape
