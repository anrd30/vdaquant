#!/usr/bin/env python
"""
Paired BCa bootstrap confidence intervals between two configs in a
pareto_benchmark_results.json (S2, docs/optimization_ledger.md). Turns
"0.9128 vs 0.9036" into a defensible claim with an interval and a
statement of whether it excludes zero -- this is what ledger decision
gate DG-1 and reviewer attack A6 require.

Usage:
    python scripts/compute_stats.py outputs/phase3/e5a_nyu_full/pareto_benchmark_results.json \
        --config-a FP32_Baseline --config-b 3bit --metric delta1

    python scripts/compute_stats.py results.json --all
        # compares every OTHER config against --config-a (default FP32_Baseline),
        # writes stats.json + stats.md next to the input file.

No scipy dependency: the normal CDF uses math.erf (stdlib) and the inverse
normal CDF (probit) uses Acklam's rational approximation, accurate to
~1.15e-9 -- see _norm_ppf below.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the stdlib error function -- no scipy needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (probit), Peter Acklam's rational
    approximation. Accurate to about 1.15e-9 over (0, 1). Pure stdlib math,
    no scipy dependency (S2 spec: "numpy only").
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def bca_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, confidence: float = 0.95):
    """
    BCa (bias-corrected and accelerated) bootstrap CI for the mean of
    `diffs` (paired per-image differences B - A). Returns
    (theta_hat, ci_lo, ci_hi, boot_means).

    Bias correction z0 comes from the fraction of bootstrap replicate means
    below the observed mean; acceleration `a` comes from the jackknife
    (Efron & Tibshirani, "An Introduction to the Bootstrap", Ch. 14).
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    theta_hat = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)

    # Bias correction. Clip away from exactly 0/1 so a degenerate all-equal
    # bootstrap distribution (e.g. all-zero diffs) can't produce +/-inf.
    eps = 1.0 / (n_boot * 10.0)
    prop_less = float(np.mean(boot_means < theta_hat))
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = _norm_ppf(prop_less)

    # Acceleration via jackknife leave-one-out means.
    jack_means = np.array([np.mean(np.delete(diffs, i)) for i in range(n)])
    jack_bar = jack_means.mean()
    num = np.sum((jack_bar - jack_means) ** 3)
    den = 6.0 * (np.sum((jack_bar - jack_means) ** 2) ** 1.5)
    a = float(num / den) if den != 0.0 else 0.0

    alpha = 1.0 - confidence
    z_lo = _norm_ppf(alpha / 2.0)
    z_hi = _norm_ppf(1.0 - alpha / 2.0)

    def _adjusted_percentile(z):
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0:
            denom = 1e-12
        return _norm_cdf(z0 + (z0 + z) / denom)

    alpha1 = min(max(_adjusted_percentile(z_lo), 0.0), 1.0)
    alpha2 = min(max(_adjusted_percentile(z_hi), 0.0), 1.0)

    ci_lo = float(np.percentile(boot_means, 100.0 * alpha1))
    ci_hi = float(np.percentile(boot_means, 100.0 * alpha2))
    return theta_hat, ci_lo, ci_hi, boot_means


def bootstrap_mean_std(x: np.ndarray, n_boot: int, seed: int):
    """Simple bootstrap of a single array's own mean; returns (mean, bootstrap std)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    return float(x.mean()), float(boot_means.std(ddof=1))


def _resolve_dataset(results: dict, dataset: str = None) -> str:
    if dataset is not None:
        if dataset not in results:
            raise ValueError(f"Dataset '{dataset}' not found; available: {list(results.keys())}")
        return dataset
    if len(results) != 1:
        raise ValueError(
            f"Results JSON has multiple datasets ({list(results.keys())}); "
            f"pass --dataset explicitly."
        )
    return next(iter(results))


def _get_per_image(cfgs: dict, config: str):
    if config not in cfgs:
        raise ValueError(f"Config '{config}' not found; available: {list(cfgs.keys())}")
    cfg = cfgs[config]
    if "per_image" not in cfg:
        raise ValueError(
            f"Config '{config}' has no 'per_image' field -- this results JSON predates the "
            f"S2 per-image dump, or was produced in a mode that doesn't emit one (fidelity/"
            f"temporal). Re-run with a groundtruth-mode benchmark to get per-image stats."
        )
    return cfg["per_image"]


def paired_arrays(per_image_a, per_image_b, metric: str):
    """
    Extracts matched-index arrays for `metric` from two per_image lists.
    Hard-fails on ANY idx misalignment -- a silent mismatch would pair the
    wrong images together and invalidate every downstream stat.
    """
    idx_a = [r["idx"] for r in per_image_a]
    idx_b = [r["idx"] for r in per_image_b]
    if idx_a != idx_b:
        n = min(len(idx_a), len(idx_b))
        first_diff = next((i for i in range(n) if idx_a[i] != idx_b[i]), n)
        val_a = idx_a[first_diff] if first_diff < len(idx_a) else "<end>"
        val_b = idx_b[first_diff] if first_diff < len(idx_b) else "<end>"
        raise ValueError(
            f"Per-image idx lists are misaligned (len A={len(idx_a)}, len B={len(idx_b)}, "
            f"first divergence at position {first_diff}: idx_a={val_a} vs idx_b={val_b}). "
            f"Pairing would silently compare the wrong images -- refusing to compute stats."
        )
    a = np.array([r[metric] for r in per_image_a], dtype=np.float64)
    b = np.array([r[metric] for r in per_image_b], dtype=np.float64)
    return a, b


def compare_pair(cfgs: dict, config_a: str, config_b: str, metric: str,
                  n_boot: int, seed: int, confidence: float) -> dict:
    per_a = _get_per_image(cfgs, config_a)
    per_b = _get_per_image(cfgs, config_b)
    a_arr, b_arr = paired_arrays(per_a, per_b, metric)
    diffs = b_arr - a_arr

    theta_hat, ci_lo, ci_hi, _ = bca_bootstrap_ci(diffs, n_boot, seed, confidence)
    mean_a, std_a = bootstrap_mean_std(a_arr, n_boot, seed + 1)
    mean_b, std_b = bootstrap_mean_std(b_arr, n_boot, seed + 2)
    excludes_zero = bool(ci_lo > 0.0 or ci_hi < 0.0)

    return {
        "metric": metric,
        "n": int(len(diffs)),
        "config_a": config_a, "mean_a": round(mean_a, 6), "bootstrap_std_a": round(std_a, 6),
        "config_b": config_b, "mean_b": round(mean_b, 6), "bootstrap_std_b": round(std_b, 6),
        "mean_diff_b_minus_a": round(theta_hat, 6),
        "ci_confidence": confidence,
        "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
        "ci_excludes_zero": excludes_zero,
        "n_boot": n_boot, "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", type=str, help="Path to a pareto_benchmark_results.json")
    ap.add_argument("--dataset", type=str, default=None,
                     help="Dataset key inside 'results' (auto-resolved if only one is present)")
    ap.add_argument("--config-a", type=str, default="FP32_Baseline")
    ap.add_argument("--config-b", type=str, default=None,
                     help="Required unless --all is set")
    ap.add_argument("--metric", type=str, default="delta1", choices=["delta1", "abs_rel", "rmse"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--all", action="store_true",
                     help="Compare every OTHER config against --config-a; writes "
                          "stats.json + stats.md next to the input file.")
    args = ap.parse_args()

    results_path = Path(args.results_json)
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]
    dataset = _resolve_dataset(results, args.dataset)
    cfgs = results[dataset]

    if args.all:
        if args.config_a not in cfgs:
            raise ValueError(f"--config-a '{args.config_a}' not in dataset '{dataset}'.")
        rows = []
        all_stats = {"dataset": dataset, "config_a": args.config_a, "metric": args.metric,
                      "n_boot": args.n_boot, "seed": args.seed, "confidence": args.confidence,
                      "comparisons": {}}
        for name, cfg in cfgs.items():
            if name == args.config_a or "per_image" not in cfg:
                continue
            row = compare_pair(cfgs, args.config_a, name, args.metric, args.n_boot, args.seed, args.confidence)
            rows.append(row)
            all_stats["comparisons"][name] = row

        md = [f"# Paired BCa bootstrap vs {args.config_a} (metric={args.metric}, dataset={dataset})", "",
              f"| Config | mean({args.config_a}) | mean(config) | diff | {args.confidence * 100:.0f}% CI | excludes 0 |",
              "|---|---|---|---|---|---|"]
        for row in rows:
            md.append(
                f"| {row['config_b']} | {row['mean_a']:.4f} | {row['mean_b']:.4f} | "
                f"{row['mean_diff_b_minus_a']:+.4f} | [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] | "
                f"{'YES' if row['ci_excludes_zero'] else 'no'} |"
            )
        md_text = "\n".join(md) + "\n"

        stats_path = results_path.parent / "stats.json"
        md_path = results_path.parent / "stats.md"
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2, sort_keys=True)
        with open(md_path, "w") as f:
            f.write(md_text)
        print(md_text)
        print(f"Wrote {stats_path}")
        print(f"Wrote {md_path}")
        return

    if args.config_b is None:
        raise ValueError("--config-b is required unless --all is set.")

    row = compare_pair(cfgs, args.config_a, args.config_b, args.metric, args.n_boot, args.seed, args.confidence)
    print(f"Dataset: {dataset}   metric: {args.metric}   n={row['n']}")
    print(f"  {row['config_a']}: mean={row['mean_a']:.4f} (bootstrap std={row['bootstrap_std_a']:.4f})")
    print(f"  {row['config_b']}: mean={row['mean_b']:.4f} (bootstrap std={row['bootstrap_std_b']:.4f})")
    print(f"  mean diff (B - A): {row['mean_diff_b_minus_a']:+.4f}")
    print(f"  {args.confidence * 100:.0f}% BCa CI: [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]")
    print(f"  CI excludes 0: {'YES' if row['ci_excludes_zero'] else 'no'}")


if __name__ == "__main__":
    main()
