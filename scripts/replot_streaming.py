"""
Rebuild the streaming accuracy figure from the JSON produced by
streaming_accuracy_figure.py. Cheap - no inference reruns.

Drops the broken TAE panel (scale mismatch between GT disparity and
VDA's affine-unaligned disparity produced garbage 240-255% values).
The delta1 panel is the real story anyway: 'BW16-3b tracks FP32 to
within delta1 <= 0.005 across every frame in the clip' - directly
answers 'does aggregate delta1 hide per-frame drift?'.

Usage:
  python scripts/replot_streaming.py \
      --json outputs/streaming_figure/streaming_bamboo_1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--output-dir", default=None,
                    help="Where to save the figure (default: alongside the JSON).")
    args = ap.parse_args()

    j = json.loads(Path(args.json).read_text())
    scene = j["scene"]
    N = j["n_frames"]

    fp32 = j["configs"]["FP32_Baseline"]["per_frame"]
    bw16 = j["configs"]["BW16_3b_sim"]["per_frame"]
    d1_fp32 = np.array([m["delta1"] for m in fp32])
    d1_bw16 = np.array([m["delta1"] for m in bw16])
    ar_fp32 = np.array([m["abs_rel"] for m in fp32])
    ar_bw16 = np.array([m["abs_rel"] for m in bw16])

    max_gap = float(np.abs(d1_fp32 - d1_bw16).max())
    mean_gap = float(np.abs(d1_fp32 - d1_bw16).mean())
    print(f"[{scene}] n_frames={N}")
    print(f"  delta1 mean: FP32={d1_fp32.mean():.4f}  BW16={d1_bw16.mean():.4f}"
          f"  gap={d1_bw16.mean() - d1_fp32.mean():+.4f}")
    print(f"  delta1 per-frame  max |gap|={max_gap:.4f}  mean |gap|={mean_gap:.4f}")
    print(f"  AbsRel mean: FP32={ar_fp32.mean():.4f}  BW16={ar_bw16.mean():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    xs = np.arange(N)
    ax1.plot(xs, d1_fp32, color="#2E7D32", label="FP32 baseline", linewidth=1.6)
    ax1.plot(xs, d1_bw16, color="#C62828", label="BW16 3-bit KV",
             linewidth=1.6, linestyle="--")
    lo1 = float(min(d1_fp32.min(), d1_bw16.min())) - 0.02
    hi1 = float(max(d1_fp32.max(), d1_bw16.max())) + 0.02
    ax1.set_ylim(lo1, hi1)
    ax1.set_ylabel(r"$\delta_1$")
    ax1.set_title(rf"Per-frame $\delta_1$ on Sintel {scene} ({N} frames)."
                  f"  Max per-frame |gap| = {max_gap:.3f}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", frameon=False)

    ax2.plot(xs, ar_fp32, color="#2E7D32", label="FP32 baseline", linewidth=1.6)
    ax2.plot(xs, ar_bw16, color="#C62828", label="BW16 3-bit KV",
             linewidth=1.6, linestyle="--")
    lo2 = float(min(ar_fp32.min(), ar_bw16.min())) - 0.005
    hi2 = float(max(ar_fp32.max(), ar_bw16.max())) + 0.005
    ax2.set_ylim(lo2, hi2)
    ax2.set_ylabel("AbsRel")
    ax2.set_xlabel("Frame index")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.json).parent
    stem = out_dir / f"streaming_{scene}_v2"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    print(f"Saved {stem}.pdf / .png")


if __name__ == "__main__":
    main()
