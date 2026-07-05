#!/usr/bin/env python3
"""Benchmark baseline, VAE-chunk optimized, and TorchAO-quantized optimized runs."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from bench_minwm_common import BenchCase, hms, run_case, write_stage_profile


def write_results(result_dir: Path, run_id: str, rows: list[dict[str, str]]) -> None:
    csv_file = result_dir / "results.csv"
    fieldnames = [
        "run_id",
        "case",
        "status",
        "elapsed_hms",
        "elapsed_seconds",
        "output_dir",
        "log_file",
        "torchao_quant",
        "compile",
        "cleanup_each_sample",
        "async_video_writer",
        "llv2_cache_quant",
        "offload_generator_before_vae",
        "vae_temporal_chunk",
        "vae_chunk_overlap",
        "profile_files",
        "profile_stage_seconds",
        "profile_peak_allocated_gb",
        "profile_peak_reserved_gb",
        "profile_min_free_gb",
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, **{key: row.get(key, "") for key in fieldnames if key != "run_id"}})

    stage_file = write_stage_profile(result_dir, run_id, rows)
    summary = result_dir / "summary.md"
    baseline = next((r for r in rows if r["case"] == "baseline"), None)
    optimized = next((r for r in rows if r["case"] == "optimized"), None)
    torchao = next((r for r in rows if r["case"] == "torchao"), None)

    def speedup(a: dict[str, str] | None, b: dict[str, str] | None) -> str:
        if not a or not b or a["status"] != "success" or b["status"] != "success":
            return "not computed"
        seconds = float(b["elapsed_seconds"])
        return "not computed" if seconds <= 0 else f"{float(a['elapsed_seconds']) / seconds:.4f}x"

    with summary.open("w", encoding="utf-8") as f:
        f.write("# minWM DMD TorchAO Benchmark\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- started_at: `{datetime.now().astimezone().isoformat(timespec='seconds')}`\n")
        f.write(f"- cwd: `{Path.cwd()}`\n")
        f.write(f"- stage_csv: `{stage_file}`\n\n")
        f.write("| case | status | elapsed | seconds | torchao | peak_allocated_gb | min_free_gb |\n")
        f.write("| --- | --- | ---: | ---: | --- | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['case']} | {row['status']} | {row['elapsed_hms']} | {row['elapsed_seconds']} | "
                f"{row.get('torchao_quant', '')} | {row.get('profile_peak_allocated_gb', '')} | "
                f"{row.get('profile_min_free_gb', '')} |\n"
            )
        f.write("\n## Speedup\n\n")
        f.write(f"- baseline_to_optimized: `{speedup(baseline, optimized)}`\n")
        f.write(f"- baseline_to_torchao: `{speedup(baseline, torchao)}`\n")
        f.write(f"- optimized_to_torchao: `{speedup(optimized, torchao)}`\n")
        f.write("\n## Stage Seconds\n\n")
        for row in rows:
            f.write(f"- {row['case']}: `{row.get('profile_stage_seconds', '{}')}`\n")

    print(f"[bench-torchao] summary: {summary}")
    print(f"[bench-torchao] csv:     {csv_file}")
    print(f"[bench-torchao] stages:  {stage_file}")


def main() -> int:
    root = Path.cwd()
    if not (root / "Wan21" / "wan_inference.py").exists():
        print("Run this from the root of a cloned shengshu-ai/minWM repo.", file=sys.stderr)
        return 1

    script_dir = Path(__file__).resolve().parent
    run_script = Path(os.environ.get("RUN_SCRIPT", script_dir / "run_minwm_dmd_4090.py"))
    result_root = Path(os.environ.get("RESULT_ROOT", "outputs/benchmark_results"))
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S") + "_torchao")
    result_dir = result_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    prompts = os.environ.get("DATA_PATH", "Wan21/prompts/demos.txt")
    frames = os.environ.get("NUM_OUTPUT_FRAMES", "20")
    bench_prompts = os.environ.get("BENCH_PROMPTS", "3")
    checkpoint = os.environ.get("CHECKPOINT_PATH", "./ckpts/Wan21/Action2V/dmd/model.pt")
    vae_chunk = os.environ.get("MINWM_VAE_TEMPORAL_CHUNK", "2")
    vae_overlap = os.environ.get("MINWM_VAE_CHUNK_OVERLAP", "0")
    vae_backend = os.environ.get("MINWM_VAE_BACKEND", "wan")
    torchao_mode = os.environ.get("MINWM_TORCHAO_QUANT", "int8wo")
    cleanup = os.environ.get("MINWM_CLEANUP_EACH_SAMPLE", "1")

    common = {
        "DATA_PATH": prompts,
        "NUM_OUTPUT_FRAMES": frames,
        "MAX_PROMPTS": bench_prompts,
        "CHECKPOINT_PATH": checkpoint,
        "TORCHINDUCTOR_USE_CUDAGRAPHS": "0",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    optimized_common = {
        **common,
        "MINWM_COMPILE": os.environ.get("MINWM_COMPILE", "0"),
        "MINWM_CLEANUP_EACH_SAMPLE": cleanup,
        "MINWM_ASYNC_VIDEO_WRITER": os.environ.get("MINWM_ASYNC_VIDEO_WRITER", "1"),
        "MINWM_LLV2_CACHE_QUANT": os.environ.get("MINWM_LLV2_CACHE_QUANT", "0"),
        "MINWM_OFFLOAD_GENERATOR_BEFORE_VAE": "0",
        "MINWM_VAE_BACKEND": vae_backend,
        "MINWM_VAE_TEMPORAL_CHUNK": vae_chunk,
        "MINWM_VAE_CHUNK_OVERLAP": vae_overlap,
    }

    cases = [
        BenchCase(
            name="baseline",
            output_dir=os.environ.get("BASE_OUT", "outputs/bench_dmd_baseline"),
            env={
                **common,
                "OUTPUT_FOLDER": os.environ.get("BASE_OUT", "outputs/bench_dmd_baseline"),
                "MINWM_COMPILE": "0",
                "MINWM_CLEANUP_EACH_SAMPLE": "0",
                "MINWM_ASYNC_VIDEO_WRITER": "0",
                "MINWM_LLV2_CACHE_QUANT": "0",
                "MINWM_OFFLOAD_GENERATOR_BEFORE_VAE": os.environ.get("BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE", "1"),
                "MINWM_VAE_BACKEND": "wan",
                "MINWM_VAE_TEMPORAL_CHUNK": "0",
                "MINWM_VAE_CHUNK_OVERLAP": "0",
                "MINWM_TORCHAO_QUANT": "none",
                "MINWM_PROFILE_JSONL": str(result_dir / "baseline_profile_rank0.jsonl"),
            },
        ),
        BenchCase(
            name="optimized",
            output_dir=os.environ.get("FAST_OUT", "outputs/bench_dmd_4090_fast"),
            env={
                **optimized_common,
                "OUTPUT_FOLDER": os.environ.get("FAST_OUT", "outputs/bench_dmd_4090_fast"),
                "MINWM_TORCHAO_QUANT": "none",
                "MINWM_PROFILE_JSONL": str(result_dir / "optimized_profile_rank0.jsonl"),
            },
        ),
        BenchCase(
            name="torchao",
            output_dir=os.environ.get("TORCHAO_OUT", "outputs/bench_dmd_4090_torchao"),
            env={
                **optimized_common,
                "OUTPUT_FOLDER": os.environ.get("TORCHAO_OUT", "outputs/bench_dmd_4090_torchao"),
                "MINWM_TORCHAO_QUANT": torchao_mode,
                "MINWM_TORCHAO_STRICT": os.environ.get("MINWM_TORCHAO_STRICT", "0"),
                "MINWM_PROFILE_JSONL": str(result_dir / "torchao_profile_rank0.jsonl"),
            },
        ),
    ]

    rows = []
    for case in cases:
        print(f"[bench-torchao] running {case.name}")
        row = run_case(case, run_script, result_dir)
        row["torchao_quant"] = case.env.get("MINWM_TORCHAO_QUANT", "none")
        rows.append(row)

    write_results(result_dir, run_id, rows)
    return 0 if all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
