"""
Verifies scripts/dump_activation_stats.py (S3, docs/optimization_ledger.md):
the pre/post-RHT activation statistics dump that provides evidence for
decision gate DG-2 (does rotation matter on video-ViT KV activations?).

Run: pytest tests/test_activation_stats.py -q
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dump_activation_stats import (
    excess_kurtosis,
    outlier_ratio,
    channel_spread,
    compute_activation_stats,
    build_synthetic_model,
    synthetic_frames,
    register_capture_hooks,
    run_capture_pass,
    _synthetic_forward,
)
from research.transforms.hadamard import HadamardRotation


# ----------------------------- statistic functions -----------------------------

def test_excess_kurtosis_zero_on_gaussian_large_n():
    """A large Gaussian sample should have excess kurtosis near 0."""
    torch.manual_seed(0)
    x = torch.randn(200_000, 8)
    k = excess_kurtosis(x)
    assert abs(k) < 0.1, k


def test_excess_kurtosis_positive_on_heavy_tailed():
    """A mixture with rare huge outliers must show strongly positive
    (leptokurtic) excess kurtosis."""
    torch.manual_seed(0)
    x = torch.randn(5000, 8) * 0.2
    x[::200] += 50.0  # sparse huge outliers
    k = excess_kurtosis(x)
    assert k > 10.0, k


def test_excess_kurtosis_constant_input_no_nan():
    x = torch.zeros(50, 8)
    k = excess_kurtosis(x)
    assert math.isfinite(k)
    assert k == 0.0, k


def test_outlier_ratio_and_channel_spread_basic():
    x = torch.zeros(10, 4)
    x[0, 0] = 100.0  # single huge outlier in channel 0
    stats = compute_activation_stats(x)
    assert stats["outlier_ratio"] > 1.0
    assert stats["chan_spread"] > 1.0
    assert stats["n_tokens"] == 10


# ----------------------------- RHT gaussianization (the DG-2 evidence) ---------

def test_hadamard_rotation_gaussianizes_heavy_tailed_input():
    """
    Ailon-Chazelle: RHT's one job is to map a heavy-tailed input to a
    near-Gaussian one. Feed a deliberately heavy-tailed (small-Gaussian bulk
    + rare huge per-token outlier channel) tensor through JUST the
    HadamardRotation module (unit-level, no surgery/model involved) and
    assert rotated kurtosis is substantially lower than raw. Fixed seed,
    deterministic, large n (4000) for a stable estimate -- confirmed by hand
    to show raw~188 -> rotated~0.07 at this scale before writing the assert.
    """
    torch.manual_seed(0)
    d = 64
    x = torch.randn(4000, d) * 0.3
    outlier_idx = torch.randint(0, d, (x.shape[0],))
    x[torch.arange(x.shape[0]), outlier_idx] += torch.randn(x.shape[0]) * 20.0

    raw_kurt = excess_kurtosis(x)

    rot = HadamardRotation(dim=d, seed=1)
    x_rot = rot(x)
    rotated_kurt = excess_kurtosis(x_rot)

    print(f"  raw kurtosis={raw_kurt:.2f}, rotated kurtosis={rotated_kurt:.2f}")
    assert raw_kurt > 5.0, f"test input should be heavy-tailed to begin with, got {raw_kurt}"
    assert rotated_kurt < raw_kurt / 2.0, (
        f"RHT should substantially reduce kurtosis (raw={raw_kurt:.2f}, rotated={rotated_kurt:.2f})"
    )


# ----------------------------- capture machinery (synthetic, CPU) --------------

def test_synthetic_model_capture_hits_both_layers_and_tensors():
    """register_capture_hooks + run_capture_pass on the tiny mock model must
    find K and V on both mock layers (surgery actually replaced them)."""
    model = build_synthetic_model(use_rotation=True, bits=8, dim=64)
    frames = synthetic_frames(num_images=3, max_tokens=64, dim=64, seed=0)
    stats, tensors = run_capture_pass(model, frames, _synthetic_forward, max_tokens=None)

    keys = set(stats.keys())
    assert keys == {"layer1::K", "layer1::V", "layer2::K", "layer2::V"}, keys
    for key, s in stats.items():
        assert math.isfinite(s["kurtosis"]), (key, s)
        assert math.isfinite(s["outlier_ratio"]), (key, s)
        assert s["n_tokens"] == 3 * 64  # num_images * tokens per frame


def test_synthetic_frames_deterministic_with_seed():
    a = synthetic_frames(num_images=2, max_tokens=32, dim=64, seed=5)
    b = synthetic_frames(num_images=2, max_tokens=32, dim=64, seed=5)
    for fa, fb in zip(a, b):
        assert torch.equal(fa, fb)


# ----------------------------- CLI end-to-end (S3 acceptance test) -------------

def test_synthetic_cli_end_to_end(tmp_path):
    """
    S3 acceptance test: run the script with --synthetic --num-images 1;
    JSON must exist with entries for both rotated and raw, both K and V,
    and all kurtosis values finite. CPU-only, no GPU, no dataset.
    """
    out_dir = tmp_path / "activation_stats"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dump_activation_stats.py"),
         "--synthetic", "--num-images", "1", "--max-tokens", "64",
         "--output-dir", str(out_dir), "--seed", "0"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    json_path = out_dir / "activation_stats.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    records = data["records"]
    assert len(records) > 0
    assert data["mode"] == "synthetic"

    tensor_types = {r["tensor"] for r in records}
    assert tensor_types == {"K", "V"}, tensor_types

    for r in records:
        assert "raw_kurtosis" in r and "rotated_kurtosis" in r, r
        assert math.isfinite(r["raw_kurtosis"]), r
        assert math.isfinite(r["rotated_kurtosis"]), r
        assert math.isfinite(r["raw_outlier_ratio"]) and math.isfinite(r["rotated_outlier_ratio"]), r


if __name__ == "__main__":
    test_excess_kurtosis_zero_on_gaussian_large_n()
    test_excess_kurtosis_positive_on_heavy_tailed()
    test_excess_kurtosis_constant_input_no_nan()
    test_outlier_ratio_and_channel_spread_basic()
    test_hadamard_rotation_gaussianizes_heavy_tailed_input()
    test_synthetic_model_capture_hits_both_layers_and_tensors()
    test_synthetic_frames_deterministic_with_seed()
    print("Run via pytest for the CLI subprocess test.")
