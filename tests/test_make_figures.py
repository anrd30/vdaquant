"""
Verifies scripts/make_paper_figures.py (S8): the deterministic figure factory.
No GPU, no model — builds tiny fixture result JSONs and checks the factory
renders PDFs and degrades gracefully (SKIP, never traceback) on missing inputs.

Run: pytest tests/test_make_figures.py -q
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_paper_figures as mpf

requires_mpl = pytest.mark.skipif(not mpf.HAS_MPL, reason="matplotlib not installed")


def _write_exp(root: Path, name: str, dataset: str, configs: dict, extra_top=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {"eval_mode": "groundtruth", "results": {dataset: configs}}
    if extra_top:
        payload.update(extra_top)
    (d / "pareto_benchmark_results.json").write_text(json.dumps(payload))


def _pareto_cfgs():
    return {
        "FP32_Baseline": {"delta1": 0.91, "effective_bits_per_scalar": 32.0},
        "8bit": {"delta1": 0.909, "effective_bits_per_scalar": 9.0},
        "4bit": {"delta1": 0.907, "effective_bits_per_scalar": 5.0},
        "3bit": {"delta1": 0.908, "effective_bits_per_scalar": 4.0},
        "2bit": {"delta1": 0.64, "effective_bits_per_scalar": 3.0},
    }


@requires_mpl
def test_pareto_renders_pdf(tmp_path):
    root = tmp_path / "phase3"
    _write_exp(root, "e5a_nyu_full", "nyuv2", _pareto_cfgs())
    _write_exp(root, "e5b_kitti_full", "kitti", _pareto_cfgs())
    out = tmp_path / "figs"
    idx = mpf.index_experiments([str(root)])
    mpf.fig_pareto(idx, out)
    pdf = out / "F1_pareto.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1024, pdf


@requires_mpl
def test_tae_gameability_renders_pdf(tmp_path):
    root = tmp_path / "phase3"
    cfgs = {
        "FP32_Baseline": {"delta1": 0.71, "effective_bits_per_scalar": 32.0,
                          "tae_percent": 55.4, "tae_median_percent": 7.8, "tae_covis_percent": 6.0},
        "3bit": {"delta1": 0.69, "effective_bits_per_scalar": 4.0,
                 "tae_percent": 54.9, "tae_median_percent": 6.9, "tae_covis_percent": 6.1},
        "2bit": {"delta1": 0.50, "effective_bits_per_scalar": 3.0,
                 "tae_percent": 7.4, "tae_median_percent": 4.5, "tae_covis_percent": 30.0},
    }
    _write_exp(root, "e5c_sintel_temporal_full", "sintel", cfgs)
    out = tmp_path / "figs"
    mpf.fig_tae(mpf.index_experiments([str(root)]), out)
    pdf = out / "F3_tae_gameability.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1024


@requires_mpl
def test_scalebits_and_seeds_render(tmp_path):
    root = tmp_path / "phase4"
    for sb in (8, 16):
        for seed in (0, 1, 2):
            _write_exp(root, f"g2b_sb{sb}_seed{seed}", "nyuv2", {
                "FP32_Baseline": {"delta1": 0.9099, "effective_bits_per_scalar": 32.0},
                "3bit": {"delta1": 0.908 + 0.001 * seed, "effective_bits_per_scalar": 4.0 if sb == 8 else 5.0},
            })
    out = tmp_path / "figs"
    idx = mpf.index_experiments([str(root)])
    mpf.fig_scalebits(idx, out)
    mpf.fig_seeds(idx, out)
    assert (out / "F4_scalebits.pdf").exists()
    assert (out / "F6_seeds.pdf").exists()


@requires_mpl
def test_activations_renders(tmp_path):
    root = tmp_path / "phase4"
    d = root / "g3d_activation_stats_nyu"
    d.mkdir(parents=True)
    (d / "activation_stats.json").write_text(json.dumps({
        "mode": "real", "records": [
            {"layer": "l0", "tensor": "K", "raw_kurtosis": 4.1, "rotated_kurtosis": 2.1},
            {"layer": "l0", "tensor": "V", "raw_kurtosis": 20.0, "rotated_kurtosis": 10.1},
        ]}))
    out = tmp_path / "figs"
    mpf.fig_activations(None, out, [str(root)])
    assert (out / "F2_activations.pdf").exists()


@requires_mpl
def test_missing_inputs_skip_not_crash(tmp_path, capsys):
    """Every figure with no inputs must print a SKIP line and produce no PDF,
    never raise."""
    out = tmp_path / "figs"
    empty_idx = {}
    mpf.fig_pareto(empty_idx, out)
    mpf.fig_tae(empty_idx, out)
    mpf.fig_scalebits(empty_idx, out)
    mpf.fig_seeds(empty_idx, out)
    mpf.fig_activations(None, out, [str(tmp_path / "nope")])
    mpf.fig_qualitative(out, [str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert captured.out.count("[SKIP]") == 6, captured.out
    assert not out.exists() or not list(out.glob("*.pdf"))


def test_cli_empty_roots_exit_zero_all_skip(tmp_path):
    """CLI against an empty root: exit 0, all SKIP, no traceback."""
    empty = tmp_path / "empty"
    empty.mkdir()
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_paper_figures.py"),
         "--roots", str(empty), "--output-dir", str(tmp_path / "figs")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    if mpf.HAS_MPL:
        assert "[SKIP]" in r.stdout, r.stdout


if __name__ == "__main__":
    print("Run via pytest (needs matplotlib + tmp fixtures).")
