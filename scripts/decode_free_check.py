"""
Exactness check for decode-free temporal attention on a TCC-coded cache.

VDA's TemporalAttention maps a cached frame to K/V by  x_s -> W(x_s + pe_s) + b,
with nothing nonlinear in between.  With a TCC cache (keyframe + 1-bit residuals,
x_s = x_{s-1} + Q(r_s) inside a group) attention can be computed from the codes:

  scores:  q'_h = W_k,h^T q_h ;  q'_h . x_s = q'_h . x_{s-1} + sum_g scale_{s,g} sum_{i in g} sign_{s,i} q'_{h,i}
  values:  sum_s P_s x_s = sum_groups [ (sum_s P_s) key + sum_j T_j Q(r_j) ],  T_j = sum_{s>=j in group} P_s
           out_h = W_v,h (sum_s P_s x_s + sum_s P_s pe_s) + b_v,h

No cached frame is ever decoded.  This script compares that against the module's
own forward (decode first, then attend) in float64 on CPU.
usage: python -X utf8 scripts/decode_free_check.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Video-Depth-Anything'))
from video_depth_anything.motion_module.motion_module import TemporalAttention  # noqa: E402

G = 16          # residual group size
K = 8           # keyframe interval
N_CACHE = 31


def make_cache(B, C, gen):
    """Closed-loop TCC cache over 31 slots.  Slots 0, 1 are stand-alone anchors
    (coded as keyframes); slots 2..30 are consecutive frames in groups of K."""
    true = torch.cumsum(0.1 * torch.randn(B, N_CACHE, C, generator=gen, dtype=torch.float64), 1)
    true += torch.randn(B, 1, C, generator=gen, dtype=torch.float64)
    is_key = [True, True] + [((s - 2) % K == 0) for s in range(2, N_CACHE)]
    rec = torch.zeros_like(true)
    signs = torch.zeros_like(true)
    scales = torch.zeros(B, N_CACHE, C // G, dtype=torch.float64)
    for s in range(N_CACHE):
        if is_key[s]:
            rec[:, s] = true[:, s]                       # keyframe value (its own codec is irrelevant here)
            continue
        r = (true[:, s] - rec[:, s - 1]).view(B, C // G, G)
        sc = r.abs().mean(-1)                            # scale = mean |r| per group
        sg = torch.where(r >= 0, 1.0, -1.0).to(r)
        scales[:, s] = sc
        signs[:, s] = sg.view(B, C)
        rec[:, s] = rec[:, s - 1] + (sc.unsqueeze(-1) * sg).view(B, C)
    return rec, is_key, signs, scales


def groups_of(is_key):
    out, cur = [], []
    for s, k in enumerate(is_key):
        if k and cur:
            out.append(cur)
            cur = []
        cur.append(s)
    out.append(cur)
    return out


@torch.no_grad()
def decode_free(attn, x_cur, keys, is_key, signs, scales):
    """x_cur (B,1,C); keys holds only keyframe slots (others unused); returns (B,1,C)."""
    B, _, C = x_cur.shape
    H = attn.heads
    d = C // H
    pe = attn.pos_encoder.pe[0, :N_CACHE + 1].to(x_cur)              # (32, C)
    Wq, Wk, Wv = attn.to_q.weight, attn.to_k.weight, attn.to_v.weight  # (H*d, C)
    bk = attn.to_k.bias if attn.to_k.bias is not None else torch.zeros(H * d, dtype=x_cur.dtype)
    bv = attn.to_v.bias if attn.to_v.bias is not None else torch.zeros(H * d, dtype=x_cur.dtype)

    xc = x_cur[:, 0] + pe[N_CACHE]                                   # current frame, with position
    q = attn.to_q(xc).view(B, H, d)
    qp = torch.einsum('bhd,hdc->bhc', q, Wk.view(H, d, C))           # absorbed query q' (B,H,C)

    # scores for cached slots: running sums over residual codes
    dot = torch.zeros(B, H, N_CACHE, dtype=x_cur.dtype)
    qp_g = qp.view(B, H, C // G, G)
    for grp in groups_of(is_key):
        k0 = grp[0]
        run = torch.einsum('bhc,bc->bh', qp, keys[:, k0])
        dot[:, :, k0] = run
        for s in grp[1:]:
            sg = signs[:, s].view(B, 1, C // G, G)
            run = run + ((qp_g * sg).sum(-1) * scales[:, s].unsqueeze(1)).sum(-1)
            dot[:, :, s] = run
    pos_k = torch.einsum('hdc,sc->hsd', Wk.view(H, d, C), pe) + bk.view(H, 1, d)   # (H,32,d)
    score_c = dot + torch.einsum('bhd,hsd->bhs', q, pos_k[:, :N_CACHE])
    k_cur = attn.to_k(xc).view(B, H, d)
    score_n = (q * k_cur).sum(-1, keepdim=True)
    P = (torch.cat([score_c, score_n], -1) * attn.scale).softmax(-1)  # (B,H,32)
    Pc = P[..., :N_CACHE]

    # values: weighted sum over keyframes and residual codes (tail sums of P)
    agg = torch.zeros(B, H, C, dtype=x_cur.dtype)
    for grp in groups_of(is_key):
        Pg = Pc[..., grp]
        tail = Pg.flip(-1).cumsum(-1).flip(-1)                       # T_j
        agg += tail[..., :1] * keys[:, grp[0]].unsqueeze(1)
        for i, s in enumerate(grp[1:], start=1):
            res = (scales[:, s].unsqueeze(-1) * signs[:, s].view(B, C // G, G)).view(B, 1, C)
            agg += tail[..., i:i + 1] * res
    agg += Pc @ pe[:N_CACHE]                                          # cached position terms
    out_c = torch.einsum('hdc,bhc->bhd', Wv.view(H, d, C), agg) + Pc.sum(-1, keepdim=True) * bv.view(1, H, d)
    out = out_c + P[..., N_CACHE:] * attn.to_v(xc).view(B, H, d)
    out = attn.to_out[1](attn.to_out[0](out.reshape(B, 1, H * d)))
    return out


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    for C in (64, 128, 256, 384):
        attn = TemporalAttention(query_dim=C, heads=8, dim_head=C // 8, temporal_max_len=32).double().eval()
        attn._use_memory_efficient_attention_xformers = False
        for p in attn.parameters():
            torch.nn.init.normal_(p, std=C ** -0.5)
        B = 6
        rec, is_key, signs, scales = make_cache(B, C, gen)
        x_cur = torch.randn(B, 1, C, generator=gen, dtype=torch.float64)
        ref, _ = attn(x_cur, video_length=1, cached_hidden_states=rec)   # decode first, then attend
        keys = torch.where(torch.tensor(is_key).view(1, -1, 1), rec, torch.zeros_like(rec))
        got = decode_free(attn, x_cur, keys, is_key, signs, scales)
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
        print(f'C={C:4d}  max rel err {err:.2e}')
    print('PASS' if worst < 1e-10 else 'FAIL')


if __name__ == '__main__':
    main()
