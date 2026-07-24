#!/usr/bin/env python
"""
S7: analytic KV-cache memory table for Video-Depth-Anything's temporal
attention (docs/optimization_ledger.md; reviewer attack A8 "what does this
buy me"). Pure arithmetic — no GPU, no model needed for the table itself.
Only the optional --measure sanity flag touches a GPU.

ARCHITECTURE (all values traced from the VDA source in this repo, cited so
the numbers are verifiable, not guessed):

  * 4 temporal "motion modules", one per DPT feature level, attached at
    layer_3, layer_4, path_4, path_3
        -> Video-Depth-Anything/video_depth_anything/dpt_temporal.py:42-51, :79-93
  * each motion module: num_transformer_block=1, num_attention_blocks=2
        -> dpt_temporal.py:35-40  (kwargs)
        -> motion_module/motion_module.py:149-159  (num_attention_blocks
           TemporalAttention layers, each caching its own K,V)
    => 4 modules x 1 block x 2 attention layers = 8 KV-caching layers.
  * each TemporalAttention: heads=8, inner_dim = heads*dim_head = in_channels,
    so cached K and V each have width = the module's in_channels
        -> motion_module/attention.py:50-59
  * per-module in_channels = [out_channels[2], out_channels[3], features,
    features]  -> dpt_temporal.py:43-50
  * DPT reassemble/resize factors on the P x P patch grid (P = input/14):
        level0 ->x4, level1 ->x2, level2 ->x1(Identity), level3 ->/2
        -> dpt.py:70-90 (ConvTranspose stride4 / stride2 / Identity / Conv stride2)
    Motion-module spatial token counts follow from where each attaches:
        mm0 (layer_3, x1 level):  P*P
        mm1 (layer_4, /2 level):  down2(P)^2
        mm2 (path_4, sized to layer_3): P*P
        mm3 (path_3, sized to layer_2, x2 level): (2P)^2
  * temporal attention runs over the FRAME axis, each spatial token an
    independent sequence, so a window of W frames caches W frames of K,V per
    spatial token per layer.

At 518 px, P = 518//14 = 37.

Usage:
    python scripts/report_kv_memory.py
    python scripts/report_kv_memory.py --measure   # + one real fp16 peak (GPU)

IMPORTANT (stated so nobody quotes the wrong number): our quantizers are
SIMULATED (dequantized on the fly), so a measured CUDA peak does NOT show the
savings. The ANALYTIC table below is the deployable number; --measure exists
only for transparency about the simulation.
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_pareto_benchmark_suite import MODEL_CONFIGS

N_ATTN_PER_MODULE = 2   # num_attention_blocks (dpt_temporal.py:37)
HEADS = 8               # num_attention_heads (dpt_temporal.py:35)


def _down2(p: int) -> int:
    """Conv2d kernel=3 stride=2 padding=1 output size (dpt.py resize_layers[3])."""
    return (p + 2 * 1 - 3) // 2 + 1


def motion_module_specs(encoder: str, input_size: int = 518, patch: int = 14):
    """
    Per-motion-module (spatial_tokens, channel_dim) at `input_size`, derived
    from the cited architecture. Returns a list of 4 dicts.
    """
    cfg = MODEL_CONFIGS[encoder]
    oc = cfg["out_channels"]
    feat = cfg["features"]
    P = input_size // patch
    return [
        {"name": "mm0_layer3", "tokens": P * P,          "dim": oc[2]},
        {"name": "mm1_layer4", "tokens": _down2(P) ** 2, "dim": oc[3]},
        {"name": "mm2_path4",  "tokens": P * P,          "dim": feat},
        {"name": "mm3_path3",  "tokens": (2 * P) ** 2,   "dim": feat},
    ]


def kv_cache_bytes(specs, window: int, eff_bits: float,
                   n_attn_per_module: int = N_ATTN_PER_MODULE) -> float:
    """
    Total KV-cache bytes for `window` frames at `eff_bits` effective bits per
    scalar. Pure arithmetic:
        per attention layer = tokens * W * dim * 2(K,V) * eff_bits
        module = n_attn_per_module * that
        total  = sum over modules, / 8 (bits -> bytes).
    """
    total_bits = 0.0
    for s in specs:
        per_layer_bits = s["tokens"] * window * s["dim"] * 2 * eff_bits
        total_bits += n_attn_per_module * per_layer_bits
    return total_bits / 8.0


def _fmt_gb(nbytes: float) -> str:
    return f"{nbytes / (1024 ** 3):.3f}"


def build_table(encoders=("vits", "vitl"), windows=(8, 32),
                eff_bits_list=(16.0, 9.0, 5.0, 4.0, 3.0), input_size=518):
    """Returns (markdown_str, rows) for the analytic KV-memory table."""
    header = ("| Encoder | Window (frames) | " +
              " | ".join(f"{b:g}-bit (GB)" for b in eff_bits_list) +
              " | vs fp16 @ 4b |")
    sep = "|" + "---|" * (2 + len(eff_bits_list) + 1)
    lines = [f"# KV-cache memory (analytic, {input_size}px) — VDA temporal attention",
             "",
             "All-inclusive effective bits per scalar on the x-axis. fp16 = 16b is "
             "the realistic deployment baseline. Compression is exactly 16/eff_bits "
             "(only the rate varies); the GB columns give the absolute footprint.",
             "", header, sep]
    rows = []
    for enc in encoders:
        specs = motion_module_specs(enc, input_size)
        for w in windows:
            cells = []
            fp16_b = kv_cache_bytes(specs, w, 16.0)
            eff4_b = kv_cache_bytes(specs, w, 4.0)
            for b in eff_bits_list:
                nb = kv_cache_bytes(specs, w, b)
                cells.append(_fmt_gb(nb))
                rows.append({"encoder": enc, "window": w, "eff_bits": b,
                             "bytes": nb, "gb": nb / (1024 ** 3)})
            ratio = fp16_b / eff4_b
            lines.append(f"| {enc} | {w} | " + " | ".join(cells) + f" | {ratio:.1f}x |")
    return "\n".join(lines) + "\n", rows


def measure_peak(encoder: str, window: int, bits: int, quantizer: str, scale_bits: int):
    """
    Optional GPU sanity measurement of a real fp16 vs quantized-sim forward
    peak. Prints both and the honest caveat that the sim does NOT reduce peak.
    """
    import torch
    if not torch.cuda.is_available():
        print("  [--measure] no CUDA device; skipping the measured peak (analytic table stands).")
        return
    from video_depth_anything.video_depth import VideoDepthAnything
    from run_pareto_benchmark_suite import get_model_config, checkpoint_candidates
    from research.models.rotated_attention import apply_rotated_quantization_to_vda

    cfg = get_model_config(encoder)
    model = VideoDepthAnything(**cfg).eval()
    for ck in checkpoint_candidates(encoder):
        if ck.exists() and ck.stat().st_size >= 10_000_000:
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            break
    model = model.cuda()
    clip = torch.randn(1, window, 3, 518, 518, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(clip)
    fp16_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model_q = apply_rotated_quantization_to_vda(
        model, bits=bits, quantizer=quantizer, use_qjl=False, scale_bits=scale_bits, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model_q(clip)
    q_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  [--measure] fp16 peak={fp16_peak:.0f} MB, quantized-SIM peak={q_peak:.0f} MB")
    print("  [--measure] NOTE: quantizers are simulated (dequantized on the fly), so the "
          "measured peak does NOT reflect the analytic savings above. The analytic table "
          "is the deployable number; this is here only for transparency.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--output-dir", type=str, default="outputs")
    ap.add_argument("--measure", action="store_true",
                     help="Also run one real fp16-vs-quantized-sim GPU peak (needs CUDA).")
    ap.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()

    md, rows = build_table(input_size=args.input_size)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kv_memory_table.md").write_text(md)
    print(md)
    print(f"  Wrote {out_dir / 'kv_memory_table.md'}")

    # Per-encoder module breakdown (for the appendix / verification).
    for enc in ("vits", "vitl"):
        specs = motion_module_specs(enc, args.input_size)
        print(f"\n  {enc} motion-module breakdown @ {args.input_size}px "
              f"(x{N_ATTN_PER_MODULE} attn layers each):")
        for s in specs:
            print(f"    {s['name']:12s} tokens={s['tokens']:6d} dim={s['dim']:5d}")

    if args.measure:
        measure_peak(args.encoder, args.window, bits=3, quantizer="lattice_e8", scale_bits=8)


if __name__ == "__main__":
    main()
