"""
Verifies scripts/report_kv_memory.py (S7): the analytic KV-cache memory
table. Pure-arithmetic tests — no GPU, no model (per S7 spec).

Run: pytest tests/test_kv_memory.py -q
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from report_kv_memory import (
    kv_cache_bytes,
    motion_module_specs,
    build_table,
    _down2,
)


def test_kv_cache_bytes_hand_computed():
    """One module, tokens=4, dim=8, W=2, eff_bits=4, 1 attn layer:
    bits = tokens*W*dim*2(K,V)*eff = 4*2*8*2*4 = 512 -> 64 bytes."""
    specs = [{"name": "m", "tokens": 4, "dim": 8}]
    b = kv_cache_bytes(specs, window=2, eff_bits=4, n_attn_per_module=1)
    assert b == 512 / 8, b
    assert b == 64.0


def test_kv_cache_bytes_scales_linearly():
    specs = [{"name": "m", "tokens": 10, "dim": 16}]
    base = kv_cache_bytes(specs, window=4, eff_bits=8, n_attn_per_module=2)
    # Double window -> double bytes; half eff_bits -> half bytes.
    assert kv_cache_bytes(specs, window=8, eff_bits=8, n_attn_per_module=2) == 2 * base
    assert kv_cache_bytes(specs, window=4, eff_bits=4, n_attn_per_module=2) == base / 2


def test_compression_ratio_is_exactly_16_over_eff_bits():
    specs = motion_module_specs("vitl")
    fp16 = kv_cache_bytes(specs, 32, 16.0)
    for eff in (9.0, 5.0, 4.0, 3.0):
        ratio = fp16 / kv_cache_bytes(specs, 32, eff)
        assert abs(ratio - 16.0 / eff) < 1e-9, (eff, ratio)


def test_down2_matches_conv_stride2_k3_p1():
    """dpt.py resize_layers[3]: Conv2d k=3 s=2 p=1. For P=37 -> 19."""
    assert _down2(37) == 19
    assert _down2(74) == 37


def test_motion_module_specs_dims_and_tokens_518px():
    """Token counts and channel dims traced from the VDA source (see
    report_kv_memory docstring citations). P = 518//14 = 37."""
    P = 37
    vits = motion_module_specs("vits", input_size=518)
    # vits out_channels=[48,96,192,384], features=64
    assert vits[0] == {"name": "mm0_layer3", "tokens": P * P, "dim": 192}
    assert vits[1] == {"name": "mm1_layer4", "tokens": 19 * 19, "dim": 384}
    assert vits[2] == {"name": "mm2_path4", "tokens": P * P, "dim": 64}
    assert vits[3] == {"name": "mm3_path3", "tokens": 74 * 74, "dim": 64}

    vitl = motion_module_specs("vitl", input_size=518)
    # vitl out_channels=[256,512,1024,1024], features=256
    assert vitl[0]["dim"] == 1024 and vitl[0]["tokens"] == P * P
    assert vitl[1]["dim"] == 1024 and vitl[1]["tokens"] == 361
    assert vitl[2]["dim"] == 256 and vitl[3]["dim"] == 256


def test_build_table_runs_cpu_only_and_has_both_encoders():
    md, rows = build_table()
    assert "vits" in md and "vitl" in md
    assert "KV-cache memory" in md
    # rows cover 2 encoders x 2 windows x 5 eff-bit columns = 20 entries.
    assert len(rows) == 2 * 2 * 5
    # fp16 vitl W=32 should be a sane sub-2GB number, 4-bit a quarter of it.
    vitl32 = {r["eff_bits"]: r for r in rows if r["encoder"] == "vitl" and r["window"] == 32}
    assert 0.1 < vitl32[16.0]["gb"] < 3.0, vitl32[16.0]
    assert abs(vitl32[4.0]["gb"] - vitl32[16.0]["gb"] / 4.0) < 1e-6


if __name__ == "__main__":
    test_kv_cache_bytes_hand_computed()
    test_kv_cache_bytes_scales_linearly()
    test_compression_ratio_is_exactly_16_over_eff_bits()
    test_down2_matches_conv_stride2_k3_p1()
    test_motion_module_specs_dims_and_tokens_518px()
    test_build_table_runs_cpu_only_and_has_both_encoders()
    print("All KV-memory tests passed.")
