#!/usr/bin/env python3
"""Reproduce LightX2V Wan2.1 autoencoder speed/memory comparisons.

This script benchmarks the VAE/TAE path used by LightX2V's Autoencoders
collection. It does not download checkpoints. Install LightX2V and pass local
checkpoint paths explicitly.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import traceback
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class Case:
    name: str
    model_type: str
    checkpoint: Path
    use_lightvae: bool = False


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def cuda_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "free_gb": free / 1024**3,
        "total_gb": total / 1024**3,
    }


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def import_lightx2v_class(model_type: str):
    if model_type in ("taew2_1", "taew2_2"):
        try:
            from lightx2v_standalone_loader import load_wan_vae_tiny_class

            return load_wan_vae_tiny_class(model_type)
        except Exception:
            print("[lightx2v-ae] standalone LightTAE loader failed; falling back to package import", flush=True)

    targets = {
        "taew2_1": ("lightx2v.models.video_encoders.hf.wan.vae_tiny", "WanVAE_tiny"),
        "taew2_2": ("lightx2v.models.video_encoders.hf.wan.vae_tiny", "Wan2_2_VAE_tiny"),
        "vaew2_1": ("lightx2v.models.video_encoders.hf.wan.vae", "WanVAE"),
        "vaew2_2": ("lightx2v.models.video_encoders.hf.wan.vae_2_2", "Wan2_2_VAE"),
    }
    module_name, class_name = targets[model_type]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:
        detail = traceback.format_exc()
        raise SystemExit(
            f"Could not import LightX2V class {class_name} from {module_name}.\n"
            "LightX2V itself may be installed, but one optional dependency for this specific backend failed.\n"
            f"Original error:\n{detail}"
        ) from exc


def load_model(case: Case, *, dtype: torch.dtype, device: torch.device):
    cls = import_lightx2v_class(case.model_type)
    kwargs: dict[str, Any] = {"vae_path": str(case.checkpoint), "dtype": dtype, "device": device}
    if case.use_lightvae:
        kwargs["use_lightvae"] = True
    model = cls(**kwargs)
    if case.model_type.startswith("tae"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def make_video(*, frames: int, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    video = torch.rand((1, frames, 3, height, width), generator=generator, dtype=torch.float32)
    return video.to(device=device, dtype=dtype)


def run_timed(fn, *, repeats: int) -> tuple[list[float], dict[str, float]]:
    times = []
    peak: dict[str, float] = {}
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        start = time.perf_counter()
        fn()
        sync_cuda()
        times.append(time.perf_counter() - start)
        stats = cuda_stats()
        for key, value in stats.items():
            peak[key] = max(peak.get(key, 0.0), value) if key.startswith("peak_") else value
    return times, peak


def encode_then_decode(model, video):
    latent = model.encode_video(video)
    if isinstance(latent, tuple):
        latent = latent[0]
    return model.decode_video(latent)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_case(value: str) -> Case:
    parts = value.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "Case must be name:model_type:checkpoint[:lightvae], e.g. "
            "lighttae:taew2_1:/path/lighttaew2_1.pth"
        )
    name, model_type, checkpoint = parts[:3]
    use_lightvae = len(parts) == 4 and parts[3].lower() in ("lightvae", "1", "true")
    if model_type not in ("taew2_1", "taew2_2", "vaew2_1", "vaew2_2"):
        raise argparse.ArgumentTypeError(f"Unsupported model_type: {model_type}")
    return Case(name=name, model_type=model_type, checkpoint=Path(checkpoint), use_lightvae=use_lightvae)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        help="name:model_type:checkpoint[:lightvae]. Repeat this flag for multiple models.",
    )
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lightx2v_autoencoder_bench"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.output_dir / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in args.case:
        if not case.checkpoint.exists():
            raise SystemExit(f"Missing checkpoint for {case.name}: {case.checkpoint}")
        print(f"[lightx2v-ae] loading {case.name} ({case.model_type})")
        model = load_model(case, dtype=dtype, device=device)
        video = make_video(frames=args.frames, height=args.height, width=args.width, dtype=dtype, device=device)

        with torch.inference_mode():
            for _ in range(args.warmup):
                latent = model.encode_video(video)
                if isinstance(latent, tuple):
                    latent = latent[0]
                _ = model.decode_video(latent)
            sync_cuda()

            latent = model.encode_video(video)
            if isinstance(latent, tuple):
                latent = latent[0]
            sync_cuda()

            encode_times, encode_mem = run_timed(
                lambda: model.encode_video(video),
                repeats=args.repeats,
            )
            decode_times, decode_mem = run_timed(
                lambda: model.decode_video(latent),
                repeats=args.repeats,
            )
            e2e_times, e2e_mem = run_timed(
                lambda: encode_then_decode(model, video),
                repeats=args.repeats,
            )

        row = {
            "case": case.name,
            "model_type": case.model_type,
            "checkpoint": str(case.checkpoint),
            "use_lightvae": str(case.use_lightvae),
            "dtype": args.dtype,
            "frames": str(args.frames),
            "height": str(args.height),
            "width": str(args.width),
            "encode_mean_s": f"{mean(encode_times):.4f}",
            "decode_mean_s": f"{mean(decode_times):.4f}",
            "end_to_end_mean_s": f"{mean(e2e_times):.4f}",
            "encode_peak_allocated_gb": f"{encode_mem.get('peak_allocated_gb', 0.0):.4f}",
            "decode_peak_allocated_gb": f"{decode_mem.get('peak_allocated_gb', 0.0):.4f}",
            "end_to_end_peak_allocated_gb": f"{e2e_mem.get('peak_allocated_gb', 0.0):.4f}",
            "encode_peak_reserved_gb": f"{encode_mem.get('peak_reserved_gb', 0.0):.4f}",
            "decode_peak_reserved_gb": f"{decode_mem.get('peak_reserved_gb', 0.0):.4f}",
            "end_to_end_peak_reserved_gb": f"{e2e_mem.get('peak_reserved_gb', 0.0):.4f}",
        }
        rows.append(row)
        print(
            f"[lightx2v-ae] {case.name}: encode={row['encode_mean_s']}s "
            f"decode={row['decode_mean_s']}s decode_peak={row['decode_peak_allocated_gb']}GB"
        )

        del model, video, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = result_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = result_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# LightX2V Autoencoder Benchmark\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- result_csv: `{csv_path}`\n\n")
        f.write("| case | model_type | encode_s | decode_s | decode_peak_gb |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['case']} | {row['model_type']} | {row['encode_mean_s']} | "
                f"{row['decode_mean_s']} | {row['decode_peak_allocated_gb']} |\n"
            )
        f.write("\n```json\n")
        f.write(json.dumps(rows, indent=2, ensure_ascii=False))
        f.write("\n```\n")

    print(f"[lightx2v-ae] summary: {summary_path}")
    print(f"[lightx2v-ae] csv:     {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
