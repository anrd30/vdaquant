"""
Tests for research/models/stream_cache.py additions:

1. Bit-packed scalar storage decodes EXACTLY like the previous one-byte-per-code
   implementation (a verbatim copy of the old functions is kept below), and
   ``nbytes()`` equals the allocated bytes of the stored tensors.
2. Delta-coding predictors ('prev', 'none', 'extrap2', 'linfit'): the encoder's
   closed-loop reconstruction and ``DeltaLazyLayer`` decoding agree bit-exactly.

CPU only.  Run: python -X utf8 -m pytest tests/test_stream_cache_extra.py -q
"""
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from research.models import stream_cache as sc
from research.models.stream_cache import (
    CompressedEntry, DeltaLazyLayer, _DeltaEncoder, _signs, fit_linear_predictor,
    irht, parse_scalar_spec, pack_codes, rht, scalar_compress, scalar_decompress,
    unpack_codes,
)

RHT_BLOCK = sc.RHT_BLOCK
MODES = ['int2-g16', 'int3-g16', 'int4-g16', 'bin-g16', 'int4-g64-asym',
         'int2-g16-rht', 'int3-gT']
SHAPES = [(1369, 1, 192), (361, 1, 384), (5476, 1, 64)]


# --------------------------------------------------------------------------
# Verbatim copy of the previous (unpacked) scalar implementation.
# --------------------------------------------------------------------------
def _old_compress(t, mode, signs):
    bits, g, rot, asym = parse_scalar_spec(mode)
    shape, C = tuple(t.shape), t.shape[-1]
    x = t.float().reshape(-1, C)
    if rot:
        x = rht(x.reshape(-1, C // RHT_BLOCK, RHT_BLOCK), signs).reshape(-1, C)
    xg = x.reshape(1, -1) if g == 0 else x.reshape(-1, g)
    G = xg.shape[0]
    if mode.startswith('bin'):
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
        return dict(q=q, scale=scale, zero=None, shape=shape, dtype=t.dtype, rot=rot,
                    stored=stored)
    if asym:
        mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
        scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8).half().float()
        zero = mn.half().float()
        q = ((xg - zero) / scale).round().clamp(0, 2 ** bits - 1).to(torch.uint8)
        meta = G * 4
    else:
        h = 2 ** (bits - 1)
        scale = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / max(h - 1, 1)
        if g == 16:
            step = scale.max().clamp(min=1e-8) / 255.0
            scale = (scale / step).round().clamp(0, 255) * step
            meta = G * 1
        else:
            scale = scale.half().float()
            meta = G * 2
        zero = None
        q = (xg / scale.clamp(min=1e-12)).round().clamp(-h, h - 1).to(torch.int8)
    stored = (xg.numel() * bits + 7) // 8 + meta
    return dict(q=q, scale=scale, zero=zero, shape=shape, dtype=t.dtype, rot=rot,
                stored=stored)


def _old_decompress(p, signs):
    x = p['q'].float() * p['scale'] + (p['zero'] if p['zero'] is not None else 0.0)
    C = p['shape'][-1]
    x = x.reshape(-1, C)
    if p['rot']:
        x = irht(x.reshape(-1, C // RHT_BLOCK, RHT_BLOCK), signs).reshape(-1, C)
    return x.reshape(p['shape']).to(p['dtype'])


# --------------------------------------------------------------------------
# Task 1: packed scalar storage
# --------------------------------------------------------------------------
@pytest.mark.parametrize('nb', [1, 2, 4, 8])
@pytest.mark.parametrize('n', [1, 7, 8, 9, 63, 1000])
def test_pack_unpack_roundtrip(nb, n):
    g = torch.Generator().manual_seed(n * 10 + nb)
    codes = torch.randint(0, 2 ** nb, (n,), generator=g, dtype=torch.int64).to(torch.uint8)
    packed = pack_codes(codes, nb)
    assert packed.dtype == torch.uint8
    assert packed.numel() == (n * nb + 7) // 8
    assert torch.equal(unpack_codes(packed, nb, n), codes)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape', SHAPES)
def test_packed_decode_matches_old(mode, shape):
    g = torch.Generator().manual_seed(zlib.crc32(repr((mode, shape)).encode()))
    t = torch.randn(shape, generator=g) * torch.rand(shape[-1], generator=g) * 3
    t[0, 0, 0] = 40.0                                           # an outlier
    signs = _signs(0, 'cpu')
    old = _old_decompress(_old_compress(t, mode, signs), signs)
    new = scalar_decompress(scalar_compress(t, mode, signs), signs)
    assert new.shape == t.shape and new.dtype == old.dtype
    assert torch.equal(old, new)


@pytest.mark.parametrize('mode', MODES)
def test_nbytes_is_allocated_bytes(mode):
    t = torch.randn(361, 1, 384)
    p = scalar_compress(t, mode, _signs(0, 'cpu'))
    tensors = [p.q, p.scale] + ([p.zero] if p.zero is not None else [])
    real = sum(x.numel() * x.element_size() for x in tensors)
    assert p.q.dtype == torch.uint8 and p.nbytes() == real + (4 if p.step is not None else 0)
    assert p.stored_bytes == p.nbytes()
    bits, g, _, asym = parse_scalar_spec(mode)
    n = t.numel()
    code_bits = 1 if mode.startswith('bin') else (2 if bits == 2 else 4)
    assert p.q.numel() == n * code_bits // 8
    G = 1 if g == 0 else n // g
    if g == 16:
        assert p.scale.dtype == torch.uint8 and p.scale.numel() == G and p.step is not None
    else:
        assert p.scale.dtype == torch.float16 and p.step is None
    if asym:
        assert p.zero.dtype == torch.float16 and p.zero.numel() == G


def test_compressed_entry_nbytes_sums():
    tensors = [torch.randn(100, 1, 64), torch.randn(100, 1, 192)]
    e = CompressedEntry(tensors, 'int4-g16', 4, _signs(0, 'cpu'))
    assert e.nbytes() == sum(p.nbytes() for p in e.items)
    # int4-g16: 0.5 B/code + 1 B per 16 codes + 4
    assert e.items[0].nbytes() == 100 * 64 // 2 + 100 * 64 // 16 + 4


# --------------------------------------------------------------------------
# Task 2: predictors
# --------------------------------------------------------------------------
def _sequence(T=40, N=48, Cs=(64, 192), seed=0, noise=0.004, vel=0.05, acc=0.004):
    """Smooth correlated frames: base + constant velocity + per-channel
    acceleration + a slow sinusoid + small noise.  Returns [T][layers] (N,1,C)."""
    g = torch.Generator().manual_seed(seed)
    layers = []
    for C in Cs:
        layers.append(dict(
            base=torch.randn(N, 1, C, generator=g),
            v=torch.randn(N, 1, C, generator=g) * vel,
            a=torch.randn(1, 1, C, generator=g) * acc,
            ph=torch.rand(N, 1, C, generator=g) * 6.28))
    seq = []
    for t in range(T):
        frame = []
        for L in layers:
            x = L['base'] + L['v'] * t + 0.5 * L['a'] * t * t + 0.05 * torch.sin(0.2 * t + L['ph'])
            x = x + torch.randn(x.shape, generator=g) * noise
            frame.append(x)
        seq.append(frame)
    return seq


def _run(seq, predictor, params=None, delta=10, mode='int4-g16', residual_mode='int3-g16'):
    """Encode through _DeltaEncoder; returns (entries, encoder recs per frame)."""
    signs = _signs(0, 'cpu')
    enc = _DeltaEncoder(delta, mode, residual_mode, 4, predictor, params)
    entries, recs = [], []
    with torch.no_grad():
        for frame in seq:
            entries.append(enc.encode(frame, signs))
            recs.append([r.clone() for r in enc.h1])
    return entries, recs


def _decode_layer(entries, i, k0=0, predictor='prev', params=None):
    """Decode layer i of entries[k0:] (entries[k0] must be a keyframe)."""
    chain = entries[k0:]
    n = len(chain)
    lazy = DeltaLazyLayer(chain[0], chain, 0, n, i, predictor,
                          None if params is None else params[i])
    out = lazy.materialize()                       # (N, 1 + 1 + n, C)
    return out[:, 2:]


PRED_CASES = ['prev', 'none', 'extrap2', 'linfit']


def _params_for(predictor):
    if predictor != 'linfit':
        return None
    return fit_linear_predictor(_sequence(seed=123))


@pytest.mark.parametrize('predictor', PRED_CASES)
def test_encoder_decoder_closed_loop_match(predictor):
    seq = _sequence()
    params = _params_for(predictor)
    entries, recs = _run(seq, predictor, params)
    intra_idx = [k for k, e in enumerate(entries) if e.intra]
    assert intra_idx == [0, 10, 20, 30]
    for i in range(2):
        for k0 in (0, 10, 30):                   # chains start at a keyframe
            dec = _decode_layer(entries, i, k0, predictor, params)
            for j in range(dec.shape[1]):
                enc_rec = recs[k0 + j][i][:, 0]
                err = (dec[:, j] - enc_rec).abs().max().item()
                assert err <= 1e-6, (predictor, i, k0, j, err)
                assert torch.equal(dec[:, j], enc_rec)      # in fact bit-identical


def test_prev_matches_legacy_recursion():
    """'prev' must reproduce the pre-change closed loop: rec = prev_rec + dec(cur - prev_rec)."""
    seq = _sequence(T=25)
    signs = _signs(0, 'cpu')
    entries, recs = _run(seq, 'prev', delta=8)
    prev = None
    for k, frame in enumerate(seq):
        cur = [t.float() for t in frame]
        intra = entries[k].intra
        assert intra == (prev is None or k % 8 == 0)
        pay = cur if intra else [c - p for c, p in zip(cur, prev)]
        mode = 'int4-g16' if intra else 'int3-g16'
        dec = [scalar_decompress(scalar_compress(x, mode, signs), signs) for x in pay]
        prev = dec if intra else [p + d for p, d in zip(prev, dec)]
        for i in range(2):
            assert torch.equal(recs[k][i], prev[i])


def test_none_equals_independent_coding():
    seq = _sequence(T=25)
    signs = _signs(0, 'cpu')
    entries, recs = _run(seq, 'none', delta=8)
    for k, frame in enumerate(seq):
        mode = 'int4-g16' if entries[k].intra else 'int3-g16'
        for i in range(2):
            ref = scalar_decompress(scalar_compress(frame[i].float(), mode, signs), signs)
            assert torch.equal(recs[k][i], ref)


def test_extrap2_uses_two_previous_within_segment():
    """Check the prediction rule directly: after a keyframe falls back to rec[t-1]."""
    seq = _sequence(T=14)
    entries, recs = _run(seq, 'extrap2', delta=5)   # keyframes at 0, 5, 10
    signs = _signs(0, 'cpu')
    for k in range(1, 14):
        if entries[k].intra:
            continue
        r1 = recs[k - 1][0]
        r2 = recs[k - 2][0] if (k >= 2 and not entries[k - 1].intra) else None
        pred = r1 if r2 is None else 2.0 * r1 - r2
        dec = scalar_decompress(scalar_compress(seq[k][0].float() - pred, 'int3-g16', signs), signs)
        assert torch.equal(recs[k][0], pred + dec)


def test_fit_linear_predictor_recovers_quadratic_drift():
    N, C, T = 64, 8, 30
    g = torch.Generator().manual_seed(1)
    base = torch.randn(N, 1, C, generator=g)
    v = torch.randn(N, 1, C, generator=g) * 0.1
    a = torch.randn(1, 1, C, generator=g) * 0.01
    seq = [[base + v * t + 0.5 * a * t * t]
           for t in range(T)]
    (A, B, Cc), = fit_linear_predictor(seq, ridge=1e-9)
    assert A.shape == B.shape == Cc.shape == (C,)
    # x[t] = 2x[t-1] - x[t-2] + a exactly
    assert torch.allclose(A, torch.full((C,), 2.0), atol=1e-2)
    assert torch.allclose(B, torch.full((C,), -1.0), atol=1e-2)
    assert torch.allclose(Cc, a.reshape(C), atol=1e-2)


def _mean_err(seq, recs, skip_intra_entries):
    errs = []
    for k, frame in enumerate(seq):
        if skip_intra_entries[k]:
            continue
        for i in range(len(frame)):
            errs.append(((recs[k][i] - frame[i]) ** 2).mean().item())
    return sum(errs) / len(errs)


def test_error_ordering_on_smooth_data():
    seq = _sequence(T=40)
    out = {}
    for pr in PRED_CASES:
        entries, recs = _run(seq, pr, _params_for(pr))
        out[pr] = _mean_err(seq, recs, [e.intra for e in entries])
    print({k: f"{v:.3e}" for k, v in out.items()})
    # coding the (large) raw frames at 3 bit is much worse than any temporal prediction
    assert out['none'] > 5 * out['prev']
    # velocity/acceleration dominate noise here: second-order predictors win
    assert out['extrap2'] < out['prev']
    assert out['linfit'] < out['prev']
