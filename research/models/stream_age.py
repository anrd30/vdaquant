"""
Age-dependent treatment of Video-Depth-Anything's streaming temporal cache,
for testing "fading memory" (recent frames precise, old frames coarse).

In VDA's streaming step the temporal attention of every layer receives a
cache of 31 past frames, ordered
    slot 0      : frame 0 (permanent first keyframe)
    slot 1      : the oldest frame of the sliding window (~40 frames old)
    slots 2..30 : frames t-29 .. t-1   (age = 31 - slot)
This module hooks each TemporalAttention so that, before attention runs,
chosen slots are quantized (quantize + dequantize, deterministic per frame,
hence equivalent to storing them quantized) or zeroed, with a per-slot
scheme.  It can also record the attention mass each slot receives.

Spec strings (used by scripts/stream_eval.py with an 'age:' prefix):
    age:<bucket>=<kind>[+<bucket>=<kind>...]
buckets: anchor (slot 0), oldest (slot 1), old (ages 20-29), mid (10-19),
         recent (1-9), all (every slot)
kinds:   fp32 | zero | any scalar spec from stream_cache (int4-g16, int2-g16, bin-g16, ...)
Example: age:recent=int4-g16+mid=int2-g16+old=bin-g16+oldest=bin-g16+anchor=int4-g16
"""
from __future__ import annotations

from typing import Dict, List

import torch

from research.models.stream_cache import scalar_compress, scalar_decompress

N_SLOTS = 31


def bucket_slots(name: str) -> List[int]:
    if name.startswith('s') and name[1:].isdigit():      # single slot, e.g. s17
        return [int(name[1:])]
    if name == 'anchor':
        return [0]
    if name == 'oldest':
        return [1]
    ages = {'old': range(20, 30), 'mid': range(10, 20), 'recent': range(1, 10),
            'all': None}[name]
    if ages is None:
        return list(range(N_SLOTS))
    return [31 - a for a in ages]          # age = 31 - slot


def parse_age_spec(spec: str) -> Dict[int, str]:
    body = spec.split(':', 1)[1]
    plan = {}
    for part in body.split('+'):
        b, kind = part.split('=')
        for s in bucket_slots(b):
            plan[s] = kind
    return plan


def bits_of(kind: str) -> float:
    """Packed bits/scalar of a scheme with 8-bit per-16 scales (for memory accounting)."""
    if kind == 'fp32':
        return 32.0
    if kind == 'zero':
        return 0.0
    if kind.startswith('bin'):
        return 1.5
    b = int(kind.split('-')[0][3:])
    return b + 0.5


def plan_bits(plan: Dict[int, str]) -> float:
    return sum(bits_of(plan.get(s, 'fp32')) for s in range(N_SLOTS)) / N_SLOTS


class AttentionLog:
    """Accumulates attention mass per cache slot, per layer."""

    def __init__(self):
        self.mass = {}
        self.count = {}

    def add(self, layer: int, probs: torch.Tensor):
        # probs: (BD*heads, 1, 32) over [31 cached slots, current frame]
        m = probs.float().mean(dim=(0, 1)).cpu()
        self.mass[layer] = self.mass.get(layer, 0) + m
        self.count[layer] = self.count.get(layer, 0) + 1

    def summary(self):
        return {l: (self.mass[l] / self.count[l]).tolist() for l in self.mass}


def enable_age_quant(model, plan: Dict[int, str], log: AttentionLog = None):
    """Hook every TemporalAttention: transform cache slots per `plan`, and
    optionally log attention mass per slot."""
    attns = [m for m in model.head.modules() if type(m).__name__ == 'TemporalAttention']
    for li, attn in enumerate(attns):
        original = attn.forward

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,
                    video_length=None, cached_hidden_states=None, _orig=original, _li=li, _attn=attn):
            if cached_hidden_states is not None and cached_hidden_states.shape[1] == N_SLOTS:
                c = cached_hidden_states
                if plan:
                    c = c.clone()
                    for s, kind in plan.items():
                        if kind == 'fp32':
                            continue
                        if kind == 'zero':
                            c[:, s] = 0
                        else:
                            q = scalar_compress(c[:, s:s + 1].contiguous(), kind, None)
                            c[:, s:s + 1] = scalar_decompress(q, None).to(c.dtype)
                if log is not None:
                    _log_attention(_attn, _li, hidden_states, c, log)
                cached_hidden_states = c
            return _orig(hidden_states, encoder_hidden_states=encoder_hidden_states,
                         attention_mask=attention_mask, video_length=video_length,
                         cached_hidden_states=cached_hidden_states)

        attn.forward = forward
    return model


@torch.no_grad()
def _log_attention(attn, li, hidden_states, cached, log: AttentionLog):
    """Recompute the softmax VDA's TemporalAttention uses for the current frame's query."""
    from einops import rearrange
    h = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=1)
    x = torch.cat([cached.to(h.dtype), h], dim=1)
    if attn.pos_encoder is not None:
        x = attn.pos_encoder(x)
    if attn.group_norm is not None:
        x = attn.group_norm(x.transpose(1, 2)).transpose(1, 2)
    q = attn.reshape_heads_to_batch_dim(attn.to_q(x[:, -1:]))
    k = attn.reshape_heads_to_batch_dim(attn.to_k(x))
    probs = (torch.bmm(q.float(), k.float().transpose(1, 2)) * attn.scale).softmax(-1)
    log.add(li, probs)


def fade_controls(base_spec: str, seed: int = 0):
    """Equal-bit controls for an age plan: same multiset of per-slot schemes,
    (a) with ages reversed over slots 1..30 and (b) randomly shuffled over 1..30.
    Slot 0 (permanent first frame) is kept as in the base plan."""
    import random
    plan = parse_age_spec(base_spec)
    kinds = {s: plan.get(s, 'fp32') for s in range(N_SLOTS)}
    rev = {0: kinds[0]}
    for i in range(1, N_SLOTS):
        rev[i] = kinds[N_SLOTS - i]
    perm = list(range(1, N_SLOTS))
    random.Random(seed).shuffle(perm)
    shuf = {0: kinds[0]}
    for i, j in zip(range(1, N_SLOTS), perm):
        shuf[i] = kinds[j]
    to_spec = lambda d: 'age:' + '+'.join(f's{k}={v}' for k, v in sorted(d.items()))
    return to_spec(rev), to_spec(shuf)
