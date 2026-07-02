#!/usr/bin/env python3
"""
Evaluate VDA-HyperQuant on the full local MPI-Sintel dataset.

Walks sinteldataset/{test,training}/{clean,final}/<sequence>/frame_*.png,
runs FP32 VDA and quantized variants per sequence, and aggregates metrics
(quantized depth vs FP32 baseline — Sintel has no depth ground truth).
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VDA_DIR = REPO_ROOT / "Video-Depth-Anything"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
if VDA_DIR.exists():
    sys.path.insert(0, str(VDA_DIR))

from research.models.rotated_attention import apply_rotated_quantization_to_vda
from run_pareto_benchmark_suite import compute_academic_metrics, generate_pareto_charts

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CKPT_URL = (
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/"
    "resolve/main/video_depth_anything_vits.pth"
)
MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
# VDA temporal positional encoding is hard-coded to 32 frames max
MAX_FRAMES_PER_CHUNK = 32


def discover_sequences(sintel_root: Path, splits, passes):
    sequences = []
    for split in splits:
        for pass_name in passes:
            pass_dir = sintel_root / split / pass_name
            if not pass_dir.is_dir():
                print(f"  [Skip] Missing directory: {pass_dir}")
                continue
            for seq_dir in sorted(pass_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("frame_*.png"))
                if frames:
                    sequences.append(
                        {
                            "split": split,
                            "pass": pass_name,
                            "name": seq_dir.name,
                            "path": seq_dir,
                            "num_frames": len(frames),
                        }
                    )
    return sequences


def load_sequence_frames(seq_dir: Path, input_size: int, max_frames: int = None):
    frames = []
    frame_paths = sorted(seq_dir.glob("frame_*.png"))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((input_size, input_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        frames.append(tensor)
    return frames


def load_sequence_rgb(seq_dir: Path, input_size: int, max_frames: int = None):
    """Load RGB frames for video panels (no ImageNet normalization)."""
    rgbs = []
    frame_paths = sorted(seq_dir.glob("frame_*.png"))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((input_size, input_size), Image.BILINEAR)
        rgbs.append(np.array(img))
    return rgbs


def ensure_checkpoint(ckpt_path: Path) -> Path:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists() and ckpt_path.stat().st_size >= 10_000_000:
        return ckpt_path

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  [Model] Downloading checkpoint to {ckpt_path} (~111 MB)...")
    req = urllib.request.Request(CKPT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(ckpt_path, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    return ckpt_path


def setup_vda():
    import video_depth_anything.dinov2_layers.attention as dino_attn
    import video_depth_anything.motion_module.attention as motion_attn

    if not torch.cuda.is_available():
        dino_attn.memory_efficient_attention = lambda q, k, v, attn_bias=None, **kwargs: (
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        )
        motion_attn.XFORMERS_AVAILABLE = False

    from video_depth_anything.video_depth import VideoDepthAnything

    return VideoDepthAnything


def _load_checkpoint(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def load_fp32_model(ckpt_path: Path, device: torch.device):
    VideoDepthAnything = setup_vda()
    model = VideoDepthAnything(**MODEL_CONFIG).eval()
    model.load_state_dict(_load_checkpoint(ckpt_path))
    return model.to(device)


def build_quant_model(ckpt_path: Path, bits: int, device: torch.device, use_qjl: bool = True):
    VideoDepthAnything = setup_vda()
    model = VideoDepthAnything(**MODEL_CONFIG).eval()
    model.load_state_dict(_load_checkpoint(ckpt_path))
    model = apply_rotated_quantization_to_vda(
        model,
        bits=bits,
        quantizer="lattice_d4",
        use_qjl=use_qjl,
        verbose=False,
        replace_temporal=False,
    )
    return model.to(device).eval()


@torch.no_grad()
def run_inference(model, video_input, device, max_frames=MAX_FRAMES_PER_CHUNK):
    """Run VDA inference, chunking long sequences to respect the 32-frame temporal limit."""
    _, num_frames, _, _, _ = video_input.shape
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    if num_frames <= max_frames:
        output = model(video_input)
    else:
        chunks = []
        for start in range(0, num_frames, max_frames):
            end = min(start + max_frames, num_frames)
            chunk = video_input[:, start:end]
            chunks.append(model(chunk))
        output = torch.cat(chunks, dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-6)
    return output, elapsed


def aggregate_metrics(per_sequence_metrics):
    if not per_sequence_metrics:
        return {}
    keys = per_sequence_metrics[0].keys()
    aggregated = {}
    for key in keys:
        if key in ("fps", "mem_savings_x", "measured_mem_mb"):
            aggregated[key] = round(
                sum(m[key] for m in per_sequence_metrics) / len(per_sequence_metrics), 4
            )
        else:
            aggregated[key] = round(
                sum(m[key] for m in per_sequence_metrics) / len(per_sequence_metrics), 4
            )
    return aggregated


def pick_sample_sequences(sequences, count: int):
    """Evenly sample sequences across the dataset for representative videos."""
    if not sequences:
        return []
    if len(sequences) <= count:
        return sequences
    if count == 1:
        return [sequences[0]]
    indices = [int(round(i * (len(sequences) - 1) / (count - 1))) for i in range(count)]
    return [sequences[i] for i in indices]


def create_labeled_panel(img: np.ndarray, label: str, is_depth: bool = True) -> np.ndarray:
    """Apply inferno colormap to depth and overlay a title banner."""
    if is_depth:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        panel = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    else:
        panel = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def save_sequence_comparison_video(
    seq: dict,
    seq_key: str,
    bits: list,
    ckpt_path: Path,
    device: torch.device,
    output_path: Path,
    input_size: int,
    max_frames: int,
    fps: int,
    use_qjl: bool = True,
):
    """Render RGB | FP32 | quantized depth panels side-by-side as an MP4."""
    rgb_frames = load_sequence_rgb(seq["path"], input_size, max_frames=max_frames)
    frame_tensors = load_sequence_frames(seq["path"], input_size, max_frames=max_frames)
    if not frame_tensors:
        print(f"  [Video] Skip empty sequence: {seq_key}")
        return

    video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)
    num_frames = len(frame_tensors)

    model_fp32 = load_fp32_model(ckpt_path, device)
    fp32_out, _ = run_inference(model_fp32, video_input, device)
    fp32_np = fp32_out[0].detach().cpu().numpy()  # [T, H, W]

    depth_maps = {"FP32": fp32_np}
    for bit in bits:
        model_q = build_quant_model(ckpt_path, bit, device, use_qjl=use_qjl)
        q_out, _ = run_inference(model_q, video_input, device)
        depth_maps[f"{bit}-bit"] = q_out[0].detach().cpu().numpy()
        del model_q
    del model_fp32

    panel_w, panel_h = input_size, input_size
    labels = ["RGB Input", "FP32 Baseline"] + [f"{b}-bit D4" for b in bits]
    grid_width = panel_w * len(labels)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_width, panel_h))

    for t in range(num_frames):
        fp32_frame = depth_maps["FP32"][t]
        d_min, d_max = fp32_frame.min(), fp32_frame.max()
        d_range = max(d_max - d_min, 1e-8)

        panels = [create_labeled_panel(rgb_frames[t], labels[0], is_depth=False)]
        panels.append(
            create_labeled_panel((depth_maps["FP32"][t] - d_min) / d_range, labels[1])
        )
        for i, bit in enumerate(bits, start=2):
            key = f"{bit}-bit"
            panels.append(
                create_labeled_panel((depth_maps[key][t] - d_min) / d_range, labels[i])
            )
        writer.write(np.hstack(panels))

    writer.release()
    print(f"  [Video] Saved {output_path} ({num_frames} frames @ {fps} FPS)")


def generate_sample_videos(sequences, args, ckpt_path: Path, device: torch.device, video_dir: Path):
    """Generate a few side-by-side comparison MP4s from evenly sampled Sintel sequences."""
    video_dir.mkdir(parents=True, exist_ok=True)
    sample_seqs = pick_sample_sequences(sequences, args.video_count)
    video_bits = args.video_bits if args.video_bits else args.bits

    print(f"\n{'=' * 70}")
    print(f"  Generating {len(sample_seqs)} sample comparison videos")
    print(f"  Output      : {video_dir}")
    print(f"  Bit-widths  : {video_bits}")
    print(f"  Max frames  : {args.video_max_frames} per video")
    print(f"{'=' * 70}\n")

    for seq in sample_seqs:
        seq_key = f"{seq['split']}_{seq['pass']}_{seq['name']}"
        safe_name = seq_key.replace("/", "_")
        out_path = video_dir / f"{safe_name}_comparison.mp4"
        print(f"  -> {seq_key}")
        save_sequence_comparison_video(
            seq=seq,
            seq_key=seq_key,
            bits=video_bits,
            ckpt_path=ckpt_path,
            device=device,
            output_path=out_path,
            input_size=args.input_size,
            max_frames=args.video_max_frames,
            fps=args.video_fps,
            use_qjl=not args.no_qjl,
        )

    print(f"\n  Videos saved to {video_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Full MPI-Sintel evaluation for VDA-HyperQuant")
    parser.add_argument(
        "--sintel-root",
        type=str,
        default=str(REPO_ROOT / "sinteldataset"),
        help="Path to local sinteldataset/ directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "training"],
        help="Dataset splits to evaluate (default: test training)",
    )
    parser.add_argument(
        "--passes",
        nargs="+",
        default=["clean", "final"],
        help="Rendering passes to evaluate (default: clean final)",
    )
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2])
    parser.add_argument("--input-size", type=int, default=266)
    parser.add_argument("--output-dir", type=str, default="outputs/sintel_full")
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit sequences (smoke test)")
    parser.add_argument("--resume", action="store_true", help="Skip sequences already in results JSON")
    parser.add_argument("--save-videos", action="store_true", help="Also save sample comparison MP4s after eval")
    parser.add_argument("--videos-only", action="store_true", help="Only generate sample comparison MP4s (skip metrics)")
    parser.add_argument("--video-count", type=int, default=3, help="Number of sample videos to generate")
    parser.add_argument("--video-fps", type=int, default=10, help="FPS for output MP4s")
    parser.add_argument("--video-max-frames", type=int, default=32, help="Max frames per sample video")
    parser.add_argument(
        "--video-bits",
        nargs="+",
        type=int,
        default=None,
        help="Bit-widths shown in videos (default: same as --bits)",
    )
    parser.add_argument(
        "--no-qjl",
        action="store_true",
        help="Disable QJL bias correction (ablation: compare AbsRel vs default QJL-on run)",
    )
    args = parser.parse_args()
    use_qjl = not args.no_qjl

    sintel_root = Path(args.sintel_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sintel_full_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  VDA-HyperQuant — Full MPI-Sintel Evaluation")
    print("=" * 70)
    print(f"  Sintel root : {sintel_root}")
    print(f"  Splits      : {args.splits}")
    print(f"  Passes      : {args.passes}")
    print(f"  Bit-widths  : {args.bits}")
    print(f"  QJL bias    : {'enabled' if use_qjl else 'disabled'}")
    print(f"  Device      : {device}")
    print(f"  Chunk size  : {MAX_FRAMES_PER_CHUNK} frames (VDA temporal limit)")
    print("=" * 70)

    if not sintel_root.is_dir():
        raise FileNotFoundError(f"Sintel dataset not found at {sintel_root}")

    sequences = discover_sequences(sintel_root, args.splits, args.passes)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    total_frames = sum(s["num_frames"] for s in sequences)
    print(f"\n  Found {len(sequences)} sequences, {total_frames} frames total.\n")
    if not sequences:
        raise RuntimeError("No Sintel sequences found. Check --sintel-root, --splits, --passes.")

    ckpt_path = ensure_checkpoint(REPO_ROOT / "checkpoints" / "video_depth_anything_vits.pth")

    if args.videos_only:
        generate_sample_videos(sequences, args, ckpt_path, device, output_dir / "videos")
        return

    model_fp32 = load_fp32_model(ckpt_path, device)

    existing = {}
    if args.resume and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f).get("per_sequence", {})

    per_sequence = dict(existing)
    fp32_runs = []
    bit_runs = {bit: [] for bit in args.bits}

    for idx, seq in enumerate(sequences, start=1):
        seq_key = f"{seq['split']}/{seq['pass']}/{seq['name']}"
        if args.resume and seq_key in per_sequence and "FP32_Baseline" in per_sequence[seq_key]:
            print(f"[{idx}/{len(sequences)}] Skipping (already done): {seq_key}")
            fp32_runs.append(per_sequence[seq_key]["FP32_Baseline"])
            for bit in args.bits:
                if f"{bit}bit" in per_sequence[seq_key]:
                    bit_runs[bit].append(per_sequence[seq_key][f"{bit}bit"])
            continue

        print(f"[{idx}/{len(sequences)}] {seq_key} ({seq['num_frames']} frames)")
        frame_tensors = load_sequence_frames(seq["path"], args.input_size)
        video_input = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)

        fp32_out, fp32_time = run_inference(model_fp32, video_input, device)
        fp32_metrics = compute_academic_metrics(fp32_out, fp32_out)
        fp32_metrics["fps"] = round(seq["num_frames"] / fp32_time, 2)
        fp32_metrics["mem_savings_x"] = 1.0
        fp32_metrics["measured_mem_mb"] = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        )
        fp32_runs.append(fp32_metrics)

        seq_result = {"FP32_Baseline": fp32_metrics}

        for bit in args.bits:
            model_q = build_quant_model(ckpt_path, bit, device, use_qjl=use_qjl)
            q_out, q_time = run_inference(model_q, video_input, device)
            q_metrics = compute_academic_metrics(q_out, fp32_out)
            q_metrics["fps"] = round(seq["num_frames"] / q_time, 2)
            q_metrics["mem_savings_x"] = round(32.0 / bit, 1)
            q_metrics["measured_mem_mb"] = (
                round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                if torch.cuda.is_available()
                else 0.0
            )
            bit_runs[bit].append(q_metrics)
            seq_result[f"{bit}bit"] = q_metrics
            print(
                f"    {bit}-bit -> delta1={q_metrics['delta1']:.4f} "
                f"abs_rel={q_metrics['abs_rel']:.4f} pearson={q_metrics['pearson']:.4f} "
                f"fps={q_metrics['fps']}"
            )
            del model_q

        per_sequence[seq_key] = seq_result

        with open(results_path, "w") as f:
            json.dump(
                {
                    "per_sequence": per_sequence,
                    "aggregate": {},
                    "config": {**vars(args), "use_qjl": use_qjl},
                },
                f,
                indent=2,
            )

    aggregate = {"FP32_Baseline": aggregate_metrics(fp32_runs)}
    for bit in args.bits:
        aggregate[f"{bit}bit"] = aggregate_metrics(bit_runs[bit])

    pareto_payload = {"sintel_full": aggregate}
    chart_dir = output_dir / "charts"
    generate_pareto_charts(pareto_payload, chart_dir)

    final_report = {
        "per_sequence": per_sequence,
        "aggregate": aggregate,
        "config": {**vars(args), "use_qjl": use_qjl},
        "summary": {
            "num_sequences": len(sequences),
            "total_frames": total_frames,
            "device": str(device),
        },
        "note": (
            "Metrics compare quantized depth to FP32 VDA baseline. "
            "MPI-Sintel provides optical-flow GT, not depth GT."
        ),
    }
    with open(results_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 90)
    print("  AGGREGATE RESULTS (mean over sequences)")
    print("=" * 90)
    print(f"{'Bit-Width':<12} | {'delta1':<8} | {'abs_rel':<8} | {'rmse':<8} | {'pearson':<8} | {'fps':<8}")
    print("-" * 90)
    for label, metrics in aggregate.items():
        print(
            f"{label:<12} | {metrics['delta1']:<8} | {metrics['abs_rel']:<8} | "
            f"{metrics['rmse']:<8} | {metrics['pearson']:<8} | {metrics['fps']:<8}"
        )
    print("=" * 90)
    print(f"\nResults saved to {results_path}")
    print(f"Charts saved to {chart_dir}")

    if args.save_videos:
        generate_sample_videos(sequences, args, ckpt_path, device, output_dir / "videos")


if __name__ == "__main__":
    main()
