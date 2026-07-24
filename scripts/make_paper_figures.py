#!/usr/bin/env python
"""
S8: deterministic paper-figure factory. Reads result JSONs (no GPU, no model)
and emits vector PDFs into outputs/figures/. Every figure is independently
callable (--only <name>); a figure whose inputs are missing prints a SKIP line
naming the missing file and moves on — never a traceback, never an empty PDF.

Figures (docs/phase4_a_star_plan.md §4 S8):
  F1 pareto        delta1 vs effective bits/scalar, per dataset, per quantizer
                   family, with FP32 as a dashed asymptote  [the money figure]
  F2 activations   per-layer K/V kurtosis raw vs rotated (from dump_activation_stats)
  F3 tae           Sintel TAE vs bit-width: mean / median / covis-mean, with the
                   accuracy (delta1) overlaid — shows the gameability (F16)
  F4 scalebits     delta1 vs scale-bits {8,16} at E8@3b, per seed (G2b)
  F6 seeds         delta1 across RHT seeds at E8@3b (robustness strip)
  F5 qual          qualitative montage from dump_depth_samples PNG dirs (skipped
                   if the PNGs aren't present)

Usage:
  python scripts/make_paper_figures.py
  python scripts/make_paper_figures.py --only pareto
  python scripts/make_paper_figures.py --roots outputs/phase3 outputs/phase4 outputs/finals
"""
import argparse
import glob
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Okabe-Ito colourblind-safe palette.
PALETTE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
QUANT_COLOR = {
    "lattice_e8": PALETTE["blue"], "lattice_d4": PALETTE["green"],
    "scalar_g8": PALETTE["orange"], "scalar": PALETTE["vermillion"],
}
FIGSIZE = (7.0, 4.2)


# ------------------------------------------------------------------ loading

def _load_results(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def index_experiments(roots):
    """Map {experiment_dir_name: results_dict} across all roots (later roots
    win on name collision). Tolerates missing roots."""
    idx = {}
    for root in roots:
        for p in sorted(glob.glob(str(Path(root) / "*" / "pareto_benchmark_results.json"))):
            name = Path(p).parent.name
            data = _load_results(p)
            if data is not None:
                idx[name] = data
    return idx


def _configs(data):
    """Returns the {config: metrics} dict from a results JSON (first dataset)."""
    if not data or "results" not in data or not data["results"]:
        return {}
    return next(iter(data["results"].values()))


def _skip(name, msg):
    print(f"[SKIP] {name}: {msg}")


def _save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {stem} -> {path}")
    return path


# ------------------------------------------------------------------ figures

def fig_pareto(idx, out_dir):
    """F1: delta1 vs eff bits per dataset, one line per quantizer family."""
    # (dataset label, {quantizer: experiment_name})
    panels = [
        ("NYUv2", {"lattice_e8": "e5a_nyu_full", "scalar_g8": "fv2_scalarg8_nyu",
                   "lattice_d4": "e1b_d4_nyu"}),
        ("KITTI", {"lattice_e8": "e5b_kitti_full", "scalar_g8": "fv2_scalarg8_kitti",
                   "lattice_d4": "e1b_d4_kitti"}),
    ]
    have_any = any(exp in idx for _, m in panels for exp in m.values())
    if not have_any:
        _skip("pareto", "no pareto result dirs found (e5a_nyu_full / e5b_kitti_full / ...)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (label, mapping) in zip(axes, panels):
        fp32 = None
        for quant, exp in mapping.items():
            if exp not in idx:
                continue
            cfgs = _configs(idx[exp])
            xs, ys = [], []
            for cname, m in cfgs.items():
                if cname == "FP32_Baseline":
                    fp32 = m.get("delta1")
                    continue
                eff = m.get("effective_bits_per_scalar")
                d1 = m.get("delta1")
                if eff is not None and d1 is not None:
                    xs.append(eff); ys.append(d1)
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
                ax.plot(xs, ys, marker="o", color=QUANT_COLOR.get(quant, "gray"), label=quant)
        if fp32 is not None:
            ax.axhline(fp32, ls="--", color=PALETTE["black"], lw=1, label="FP32")
        ax.set_title(label)
        ax.set_xlabel("effective bits / scalar")
        ax.set_ylabel(r"$\delta_1\uparrow$")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Rate–distortion: KV-cache quantization of Video-Depth-Anything")
    fig.tight_layout()
    _save(fig, out_dir, "F1_pareto")


def fig_tae(idx, out_dir):
    """F3: Sintel TAE (mean / median / covis) vs bit-width with delta1 overlaid."""
    exp = next((e for e in ("e5c_sintel_temporal_full", "e7_sintel_covis") if e in idx), None)
    if exp is None:
        _skip("tae", "no Sintel temporal result dir (e5c_sintel_temporal_full / e7_sintel_covis)")
        return
    cfgs = _configs(idx[exp])
    rows = []
    for cname, m in cfgs.items():
        eff = m.get("effective_bits_per_scalar")
        if eff is None:
            continue
        rows.append((eff, cname, m))
    rows.sort(key=lambda r: r[0])
    xs = [r[0] for r in rows]
    tae_mean = [r[2].get("tae_percent") for r in rows]
    tae_med = [r[2].get("tae_median_percent") for r in rows]
    tae_cov = [r[2].get("tae_covis_percent") for r in rows]
    d1 = [r[2].get("delta1") for r in rows]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.plot(xs, tae_mean, marker="o", color=PALETTE["vermillion"], label="TAE mean %")
    ax1.plot(xs, tae_med, marker="s", color=PALETTE["orange"], label="TAE median %")
    if any(v is not None for v in tae_cov):
        ax1.plot(xs, tae_cov, marker="^", color=PALETTE["blue"], label="TAE covis-mean %")
    ax1.set_xlabel("effective bits / scalar")
    ax1.set_ylabel("TAE % (lower = smoother)")
    ax1.grid(True, ls="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(xs, d1, marker="D", color=PALETTE["green"], ls=":", label=r"$\delta_1$ (accuracy)")
    ax2.set_ylabel(r"$\delta_1\uparrow$")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax1.set_title("TAE is gameable: the collapsed low-rate model 'wins' on TAE while δ1 falls")
    fig.tight_layout()
    _save(fig, out_dir, "F3_tae_gameability")


def fig_activations(idx_unused, out_dir, roots):
    """F2: per-layer K/V kurtosis raw vs rotated from dump_activation_stats."""
    cand = []
    for root in roots:
        cand += glob.glob(str(Path(root) / "*" / "activation_stats.json"))
    path = next((c for c in cand if Path(c).exists()), None)
    if path is None:
        _skip("activations", "no activation_stats.json (run scripts/dump_activation_stats.py / G3d)")
        return
    data = _load_results(path)
    recs = (data or {}).get("records", [])
    if not recs:
        _skip("activations", f"{path} has no records")
        return
    raw = [r.get("raw_kurtosis") for r in recs if r.get("raw_kurtosis") is not None]
    rot = [r.get("rotated_kurtosis") for r in recs if r.get("rotated_kurtosis") is not None]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(range(len(raw)), raw, marker="o", color=PALETTE["vermillion"], label="raw K/V kurtosis")
    ax.plot(range(len(rot)), rot, marker="s", color=PALETTE["blue"], label="post-RHT kurtosis")
    ax.axhline(0.0, ls="--", color=PALETTE["black"], lw=0.8)
    ax.set_xlabel("temporal K/V tensor (per layer)")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("RHT gaussianizes heavy-tailed activations (mechanism for the rotation ablation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "F2_activations")


def fig_scalebits(idx, out_dir):
    """F4: delta1 vs scale-bits {8,16} at E8@3b, per seed (G2b)."""
    pts = {}  # scale_bits -> list of delta1 across seeds
    for sb in (8, 16):
        for seed in (0, 1, 2):
            exp = f"g2b_sb{sb}_seed{seed}"
            if exp in idx:
                m = _configs(idx[exp]).get("3bit", {})
                if m.get("delta1") is not None:
                    pts.setdefault(sb, []).append(m["delta1"])
    if not pts:
        _skip("scalebits", "no g2b_sb{8,16}_seed{0,1,2} result dirs")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for sb, vals in sorted(pts.items()):
        ax.scatter([sb] * len(vals), vals, color=PALETTE["blue"], zorder=3)
        ax.scatter([sb], [sum(vals) / len(vals)], color=PALETTE["vermillion"],
                   marker="_", s=400, zorder=4)
    # FP32 reference if available.
    any_exp = next(iter([idx[e] for e in idx if e.startswith("g2b_")]), None)
    if any_exp:
        fp = _configs(any_exp).get("FP32_Baseline", {}).get("delta1")
        if fp is not None:
            ax.axhline(fp, ls="--", color=PALETTE["black"], lw=1, label="FP32")
            ax.legend(fontsize=8)
    ax.set_xticks(sorted(pts))
    ax.set_xlabel("scale bits")
    ax.set_ylabel(r"$\delta_1$ (E8 @ 3-bit, NYU-654)")
    ax.set_title("Scale precision 8 ≡ 16: 4.0 eff bits is free")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "F4_scalebits")


def fig_seeds(idx, out_dir):
    """F6: delta1 across RHT seeds at E8@3b (sb8) — robustness strip."""
    vals = []
    for seed in (0, 1, 2):
        exp = f"g2b_sb8_seed{seed}"
        if exp in idx:
            m = _configs(idx[exp]).get("3bit", {})
            if m.get("delta1") is not None:
                vals.append((seed, m["delta1"]))
    if not vals:
        _skip("seeds", "no g2b_sb8_seed{0,1,2} result dirs")
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter([v[0] for v in vals], [v[1] for v in vals], color=PALETTE["blue"], zorder=3)
    ax.set_xticks([v[0] for v in vals])
    ax.set_xlabel("RHT seed")
    ax.set_ylabel(r"$\delta_1$ (E8 @ 3-bit, NYU-654)")
    ax.set_title("Seed robustness")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "F6_seeds")


def fig_qualitative(out_dir, roots):
    """F5: montage from dump_depth_samples PNG dirs (skipped if absent)."""
    pngs = []
    for root in roots:
        pngs += glob.glob(str(Path(root) / "**" / "*.png"), recursive=True)
    pngs = [p for p in pngs if "e6_figures" in p or "depth_samples" in p]
    if not pngs:
        _skip("qual", "no qualitative depth-sample PNGs (run scripts/dump_depth_samples.py / E6)")
        return
    import matplotlib.image as mpimg
    pngs = sorted(pngs)[:6]
    n = len(pngs)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 2.2 * n), squeeze=False)
    for ax, p in zip(axes[:, 0], pngs):
        ax.imshow(mpimg.imread(p))
        ax.set_title(Path(p).name, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, out_dir, "F5_qualitative")


ALL_FIGS = ("pareto", "tae", "activations", "scalebits", "seeds", "qual")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+",
                     default=["outputs/phase3", "outputs/phase4", "outputs/finals"])
    ap.add_argument("--output-dir", default="outputs/figures")
    ap.add_argument("--only", choices=ALL_FIGS, default=None)
    args = ap.parse_args()

    if not HAS_MPL:
        print("[error] matplotlib not installed; cannot render figures.")
        sys.exit(0)  # not a crash -- graceful, per S8 spec

    roots = [str(Path(r)) for r in args.roots]
    out_dir = Path(args.output_dir)
    idx = index_experiments(roots)
    print(f"  Indexed {len(idx)} experiment dir(s) across roots: {roots}")

    want = [args.only] if args.only else list(ALL_FIGS)
    for name in want:
        if name == "pareto":
            fig_pareto(idx, out_dir)
        elif name == "tae":
            fig_tae(idx, out_dir)
        elif name == "activations":
            fig_activations(idx, out_dir, roots)
        elif name == "scalebits":
            fig_scalebits(idx, out_dir)
        elif name == "seeds":
            fig_seeds(idx, out_dir)
        elif name == "qual":
            fig_qualitative(out_dir, roots)


if __name__ == "__main__":
    main()
