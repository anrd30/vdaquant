#!/usr/bin/env python3
"""
Compare QJL bias correction modes across three benchmark result JSONs.

Usage:
    python scripts/compare_qjl_modes.py <off_json> <full_json> <shrinkage_json>
    python scripts/compare_qjl_modes.py --off off.json --full full.json --shrinkage shrinkage.json
"""
import argparse
import json
import sys
from pathlib import Path


def load_json(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"[Error] File not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r") as f:
        data = json.load(f)
    return data.get("results", data)


def main():
    parser = argparse.ArgumentParser(description="Compare QJL modes from benchmark JSON reports")
    parser.add_argument("pos_jsons", nargs="*", help="Positional paths: <off_json> <full_json> <shrinkage_json>")
    parser.add_argument("--off", type=str, help="Path to qjl_mode=off JSON")
    parser.add_argument("--full", type=str, help="Path to qjl_mode=full JSON")
    parser.add_argument("--shrinkage", type=str, help="Path to qjl_mode=shrinkage JSON")
    args = parser.parse_args()

    off_path = args.off
    full_path = args.full
    shrinkage_path = args.shrinkage

    if args.pos_jsons:
        if len(args.pos_jsons) == 3:
            off_path = off_path or args.pos_jsons[0]
            full_path = full_path or args.pos_jsons[1]
            shrinkage_path = shrinkage_path or args.pos_jsons[2]
        else:
            print("[Error] If using positional arguments, please provide exactly 3: <off_json> <full_json> <shrinkage_json>", file=sys.stderr)
            sys.exit(1)

    if not (off_path and full_path and shrinkage_path):
        parser.print_help()
        sys.exit(1)

    res_off = load_json(off_path)
    res_full = load_json(full_path)
    res_shrinkage = load_json(shrinkage_path)

    # Find common datasets
    datasets = sorted(list(set(res_off.keys()) | set(res_full.keys()) | set(res_shrinkage.keys())))

    for dataset in datasets:
        print(f"\n{'=' * 98}")
        print(f"  QJL Mode Comparison — Dataset: {dataset.upper()}")
        print(f"{'=' * 98}")
        print(f"  {'Bit':<8} | {'δ1 (off)':<11} | {'δ1 (full)':<11} | {'δ1 (shrink)':<11} | {'Pearson (off)':<13} | {'Pearson (full)':<14} | {'Pearson (shrink)':<16}")
        print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*13}-+-{'-'*14}-+-{'-'*16}")

        d_off = res_off.get(dataset, {})
        d_full = res_full.get(dataset, {})
        d_shrink = res_shrinkage.get(dataset, {})

        # Collect bit widths
        bits = sorted(
            list(set(d_off.keys()) | set(d_full.keys()) | set(d_shrink.keys())),
            key=lambda x: int(x.replace("bit", "")) if "bit" in x and x.replace("bit", "").isdigit() else 0,
            reverse=True,
        )

        for b in bits:
            m_off = d_off.get(b, {})
            m_full = d_full.get(b, {})
            m_shrink = d_shrink.get(b, {})

            d1_off = f"{m_off.get('delta1', 'N/A'):.4f}" if isinstance(m_off.get('delta1'), (int, float)) else str(m_off.get('delta1', 'N/A'))
            d1_full = f"{m_full.get('delta1', 'N/A'):.4f}" if isinstance(m_full.get('delta1'), (int, float)) else str(m_full.get('delta1', 'N/A'))
            d1_shrink = f"{m_shrink.get('delta1', 'N/A'):.4f}" if isinstance(m_shrink.get('delta1'), (int, float)) else str(m_shrink.get('delta1', 'N/A'))

            p_off = f"{m_off.get('pearson', 'N/A'):.4f}" if isinstance(m_off.get('pearson'), (int, float)) else str(m_off.get('pearson', 'N/A'))
            p_full = f"{m_full.get('pearson', 'N/A'):.4f}" if isinstance(m_full.get('pearson'), (int, float)) else str(m_full.get('pearson', 'N/A'))
            p_shrink = f"{m_shrink.get('pearson', 'N/A'):.4f}" if isinstance(m_shrink.get('pearson'), (int, float)) else str(m_shrink.get('pearson', 'N/A'))

            print(f"  {b:<8} | {d1_off:<11} | {d1_full:<11} | {d1_shrink:<11} | {p_off:<13} | {p_full:<14} | {p_shrink:<16}")
        print(f"{'=' * 98}")


if __name__ == "__main__":
    main()
