"""
Verifies scripts/compute_stats.py (S2, docs/optimization_ledger.md): the
paired BCa bootstrap CI script that turns "0.9128 vs 0.9036" into a
defensible claim with a confidence interval. Feeds ledger decision gate DG-1.

Run: pytest tests/test_stats.py -q
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_stats import (
    bca_bootstrap_ci,
    bootstrap_mean_std,
    paired_arrays,
    _norm_cdf,
    _norm_ppf,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "stats_fixture.json"


def test_norm_ppf_roundtrips_through_norm_cdf():
    """Sanity check on the scipy-free probit implementation: known quantiles
    and a CDF(PPF(p)) == p round trip."""
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-5
    assert abs(_norm_ppf(0.5) - 0.0) < 1e-9
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert abs(_norm_cdf(_norm_ppf(p)) - p) < 1e-8, p


def test_bca_ci_detects_known_shift():
    """Two arrays differing by a constant 0.01 shift plus small noise
    (n=200, sd=0.001, ~10-sigma effect): the CI must contain 0.01 and
    exclude 0. Fully seeded/deterministic."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.90, scale=0.001, size=200)
    b = a + 0.01 + rng.normal(loc=0.0, scale=0.001, size=200)
    diffs = b - a

    theta_hat, lo, hi, _ = bca_bootstrap_ci(diffs, n_boot=5000, seed=0, confidence=0.95)
    print(f"  mean_diff={theta_hat:.5f}, CI=[{lo:.5f}, {hi:.5f}]")
    assert lo < 0.01 < hi, (lo, hi)
    assert lo > 0.0, f"CI should exclude 0 (lo={lo})"


def test_bca_ci_null_case_identical_arrays():
    """Identical arrays -> mean diff is exactly 0 and the CI contains 0."""
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.9, scale=0.01, size=100)
    diffs = a - a  # exactly zero, elementwise

    theta_hat, lo, hi, _ = bca_bootstrap_ci(diffs, n_boot=2000, seed=0, confidence=0.95)
    assert theta_hat == 0.0, theta_hat
    assert lo <= 0.0 <= hi, (lo, hi)


def test_paired_arrays_misalignment_hard_fails():
    """Misaligned idx lists must raise, never silently pair the wrong images."""
    per_a = [{"idx": 0, "delta1": 0.9}, {"idx": 1, "delta1": 0.8}, {"idx": 2, "delta1": 0.7}]
    per_b = [{"idx": 0, "delta1": 0.9}, {"idx": 5, "delta1": 0.6}, {"idx": 2, "delta1": 0.5}]
    with pytest.raises(ValueError, match="misaligned"):
        paired_arrays(per_a, per_b, "delta1")


def test_paired_arrays_aligned_case_ok():
    per_a = [{"idx": 0, "delta1": 0.9}, {"idx": 1, "delta1": 0.8}]
    per_b = [{"idx": 0, "delta1": 0.85}, {"idx": 1, "delta1": 0.75}]
    a, b = paired_arrays(per_a, per_b, "delta1")
    assert np.allclose(a, [0.9, 0.8])
    assert np.allclose(b, [0.85, 0.75])


def test_bootstrap_mean_std_finite_and_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(0.9, 0.02, size=50)
    m1, s1 = bootstrap_mean_std(x, n_boot=1000, seed=7)
    m2, s2 = bootstrap_mean_std(x, n_boot=1000, seed=7)
    assert m1 == m2 and s1 == s2
    assert np.isfinite(m1) and np.isfinite(s1) and s1 >= 0.0


def test_cli_runs_against_fixture_and_is_deterministic(tmp_path):
    """Acceptance test (S2 item 4): compute_stats.py runs end-to-end against
    a hand-written 5-image, 2-config fixture, and --all is byte-identical
    across two runs with the same seed."""
    work_dir = tmp_path / "run1"
    work_dir.mkdir()
    fixture_copy = work_dir / "pareto_benchmark_results.json"
    shutil.copy(FIXTURE, fixture_copy)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compute_stats.py"),
         str(fixture_copy), "--config-a", "FP32_Baseline", "--config-b", "3bit",
         "--metric", "delta1", "--n-boot", "500", "--seed", "0"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CI excludes 0" in result.stdout, result.stdout

    # --all determinism: run twice into separate copies, same seed -> byte-identical stats.json
    run_a_dir = tmp_path / "all_a"
    run_b_dir = tmp_path / "all_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()
    fixture_a = run_a_dir / "pareto_benchmark_results.json"
    fixture_b = run_b_dir / "pareto_benchmark_results.json"
    shutil.copy(FIXTURE, fixture_a)
    shutil.copy(FIXTURE, fixture_b)

    for fixture_path in (fixture_a, fixture_b):
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "compute_stats.py"),
             str(fixture_path), "--all", "--config-a", "FP32_Baseline",
             "--metric", "delta1", "--n-boot", "500", "--seed", "0"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    stats_a = (run_a_dir / "stats.json").read_bytes()
    stats_b = (run_b_dir / "stats.json").read_bytes()
    assert stats_a == stats_b, "same seed must produce byte-identical stats.json"

    parsed = json.loads(stats_a)
    assert "3bit" in parsed["comparisons"]
    assert (run_a_dir / "stats.md").exists()


if __name__ == "__main__":
    test_norm_ppf_roundtrips_through_norm_cdf()
    test_bca_ci_detects_known_shift()
    test_bca_ci_null_case_identical_arrays()
    test_paired_arrays_misalignment_hard_fails()
    test_paired_arrays_aligned_case_ok()
    test_bootstrap_mean_std_finite_and_deterministic()
    print("Run via pytest for the CLI subprocess test.")
