#!/usr/bin/env python3
"""Run one apples-to-apples minWM DMD optimization comparison.

Run this from the official minWM repo root:

    python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_all.py

Prefer the split entry points for normal use:

    bench_minwm_dmd_4090_lighttae.py
    bench_minwm_dmd_4090_wan_experimental.py

This module keeps the shared case construction and result-writing code.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from bench_minwm_common import BenchCase, run_case, write_stage_profile


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def case_enabled(name: str, default: bool = True) -> bool:
    return env_bool(f"RUN_{name.upper()}", "1" if default else "0")


def output_name(prefix: str, case: str) -> str:
    return os.environ.get(f"{case.upper()}_OUT", f"outputs/{prefix}_{case}")


def speedup(baseline: dict[str, str] | None, row: dict[str, str]) -> str:
    if not baseline or baseline.get("status") != "success" or row.get("status") != "success":
        return ""
    seconds = float(row.get("elapsed_seconds", "0") or 0.0)
    return "" if seconds <= 0 else f"{float(baseline['elapsed_seconds']) / seconds:.4f}"


def stage_speedup(
    baseline: dict[str, str] | None,
    row: dict[str, str],
    stage: str,
) -> str:
    import json

    if not baseline or baseline.get("status") != "success" or row.get("status") != "success":
        return ""
    try:
        base_stages = json.loads(baseline.get("profile_stage_seconds", "{}"))
        row_stages = json.loads(row.get("profile_stage_seconds", "{}"))
    except json.JSONDecodeError:
        return ""
    base_seconds = float(base_stages.get(stage, 0.0) or 0.0)
    row_seconds = float(row_stages.get(stage, 0.0) or 0.0)
    return "" if row_seconds <= 0 else f"{base_seconds / row_seconds:.4f}"


def generation_stage_seconds(row: dict[str, str]) -> float:
    import json

    try:
        stages = json.loads(row.get("profile_stage_seconds", "{}"))
    except json.JSONDecodeError:
        return 0.0
    # `pipeline_inference` wraps `pipeline.inference(...)`, and the VAE decode
    # profiler is nested inside that call. Do not add `vae_decode` again.
    return float(stages.get("pipeline_inference", 0.0) or 0.0) + float(
        stages.get("write_video_submit", 0.0) or 0.0
    )


def diffusion_excluding_vae_seconds(row: dict[str, str]) -> float:
    import json

    try:
        stages = json.loads(row.get("profile_stage_seconds", "{}"))
    except json.JSONDecodeError:
        return 0.0
    pipeline = float(stages.get("pipeline_inference", 0.0) or 0.0)
    vae = float(stages.get("vae_decode", 0.0) or 0.0)
    return max(0.0, pipeline - vae)


def generation_speedup(baseline: dict[str, str] | None, row: dict[str, str]) -> str:
    if not baseline or baseline.get("status") != "success" or row.get("status") != "success":
        return ""
    base_seconds = generation_stage_seconds(baseline)
    row_seconds = generation_stage_seconds(row)
    return "" if row_seconds <= 0 else f"{base_seconds / row_seconds:.4f}"


def write_results(
    result_dir: Path,
    run_id: str,
    rows: list[dict[str, str]],
    prompt_path: str,
    frames: str,
    bench_prompts: str,
    checkpoint: str,
    lighttae_path: str,
) -> None:
    baseline = next((row for row in rows if row["case"] == "baseline_offload"), None)
    stage_file = write_stage_profile(result_dir, run_id, rows)
    csv_file = result_dir / "results.csv"

    fieldnames = [
        "run_id",
        "case",
        "status",
        "elapsed_hms",
        "elapsed_seconds",
        "speedup_vs_baseline",
        "generation_stage_seconds",
        "generation_speedup_vs_baseline",
        "diffusion_excluding_vae_seconds",
        "vae_speedup_vs_baseline",
        "pipeline_speedup_vs_baseline",
        "output_dir",
        "log_file",
        "vae_backend",
        "torchao_quant",
        "lightx2v_parallel",
        "lightx2v_output_device",
        "compile",
        "cleanup_each_sample",
        "async_video_writer",
        "llv2_cache_quant",
        "offload_generator_before_vae",
        "vae_temporal_chunk",
        "vae_chunk_overlap",
        "cudagraphs",
        "profile_files",
        "profile_stage_seconds",
        "profile_peak_allocated_gb",
        "profile_peak_reserved_gb",
        "profile_min_free_gb",
    ]

    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        enriched["speedup_vs_baseline"] = speedup(baseline, row)
        enriched["generation_stage_seconds"] = f"{generation_stage_seconds(row):.3f}"
        enriched["generation_speedup_vs_baseline"] = generation_speedup(baseline, row)
        enriched["diffusion_excluding_vae_seconds"] = f"{diffusion_excluding_vae_seconds(row):.3f}"
        enriched["vae_speedup_vs_baseline"] = stage_speedup(baseline, row, "vae_decode")
        enriched["pipeline_speedup_vs_baseline"] = stage_speedup(baseline, row, "pipeline_inference")
        enriched_rows.append(enriched)

    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched_rows:
            writer.writerow({"run_id": run_id, **{key: row.get(key, "") for key in fieldnames if key != "run_id"}})

    summary = result_dir / "summary.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# minWM Action2V DMD Full Optimization Benchmark\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- started_at: `{datetime.now().astimezone().isoformat(timespec='seconds')}`\n")
        f.write(f"- cwd: `{Path.cwd()}`\n")
        f.write(f"- prompts: `{prompt_path}`\n")
        f.write(f"- frames: `{frames}`\n")
        f.write(f"- bench_prompts: `{bench_prompts}`\n")
        f.write(f"- checkpoint: `{checkpoint}`\n")
        f.write(f"- lighttae_checkpoint: `{lighttae_path}`\n")
        f.write(f"- results_csv: `{csv_file}`\n")
        f.write(f"- stage_csv: `{stage_file}`\n\n")
        f.write("## Case Meaning\n\n")
        f.write("- `baseline_offload`: original Wan VAE decode, generator offloaded before VAE to avoid 24GB OOM.\n")
        f.write("- `wan_chunk`: original Wan VAE with temporal chunk decode, no generator offload.\n")
        f.write("- `wan_chunk_torchao`: `wan_chunk` plus TorchAO generator weight-only quantization.\n")
        f.write("- `paper_lighttae`: LightX2V/LightTAE Wan2.1 autoencoder decode inside minWM, no generator offload.\n")
        f.write("- `paper_lighttae_torchao`: LightTAE decode plus TorchAO generator quantization.\n\n")
        f.write("## Results\n\n")
        f.write("| case | status | seconds | speedup | gen_seconds | gen_speedup | diffusion_excl_vae | vae_speedup | pipeline_speedup | backend | lightx2v_parallel | torchao | peak_alloc_gb | min_free_gb |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |\n")
        for row in enriched_rows:
            f.write(
                f"| {row['case']} | {row['status']} | {row['elapsed_seconds']} | "
                f"{row.get('speedup_vs_baseline', '')} | {row.get('generation_stage_seconds', '')} | "
                f"{row.get('generation_speedup_vs_baseline', '')} | {row.get('diffusion_excluding_vae_seconds', '')} | "
                f"{row.get('vae_speedup_vs_baseline', '')} | "
                f"{row.get('pipeline_speedup_vs_baseline', '')} | {row.get('vae_backend', '')} | "
                f"{row.get('lightx2v_parallel', '')} | {row.get('torchao_quant', '')} | {row.get('profile_peak_allocated_gb', '')} | "
                f"{row.get('profile_min_free_gb', '')} |\n"
            )
        f.write("\n## Stage Seconds\n\n")
        for row in enriched_rows:
            f.write(f"- {row['case']}: `{row.get('profile_stage_seconds', '{}')}`\n")
        f.write("\n## Notes\n\n")
        f.write(
            "This script reproduces the LightTAE-style optimization inside minWM by replacing the final Wan VAE decode. "
            "It is not the full LightX2V pipeline and does not claim true FP8 kernels unless your TorchAO/LightX2V stack enables them.\n"
        )

    print(f"[bench-all] summary: {summary}")
    print(f"[bench-all] csv:     {csv_file}")
    print(f"[bench-all] stages:  {stage_file}")
    print(f"[bench-all] logs:    {result_dir}")


def add_row_metadata(row: dict[str, str], case: BenchCase) -> dict[str, str]:
    row = dict(row)
    row["vae_backend"] = case.env.get("MINWM_VAE_BACKEND", "")
    row["torchao_quant"] = case.env.get("MINWM_TORCHAO_QUANT", "none")
    row["lightx2v_parallel"] = case.env.get("MINWM_LIGHTX2V_PARALLEL", "")
    row["lightx2v_output_device"] = case.env.get("MINWM_LIGHTX2V_OUTPUT_DEVICE", "")
    return row


def main() -> int:
    root = Path.cwd()
    if not (root / "Wan21" / "wan_inference.py").exists():
        print("Run this from the root of a cloned shengshu-ai/minWM repo.", file=sys.stderr)
        return 1

    script_dir = Path(__file__).resolve().parent
    run_script = Path(os.environ.get("RUN_SCRIPT", script_dir / "run_minwm_dmd_4090.py"))

    prompts = os.environ.get("DATA_PATH", "Wan21/prompts/demos.txt")
    frames = os.environ.get("NUM_OUTPUT_FRAMES", "20")
    bench_prompts = os.environ.get("BENCH_PROMPTS", "3")
    checkpoint = os.environ.get("CHECKPOINT_PATH", "./ckpts/Wan21/Action2V/dmd/model.pt")
    result_root = Path(os.environ.get("RESULT_ROOT", "outputs/benchmark_results"))
    default_suffix = os.environ.get("RUN_ID_SUFFIX", "all")
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{default_suffix}")
    result_dir = result_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = os.environ.get("OUTPUT_PREFIX", "bench_dmd_4090_all")
    vae_chunk = os.environ.get("MINWM_VAE_TEMPORAL_CHUNK", "2")
    vae_overlap = os.environ.get("MINWM_VAE_CHUNK_OVERLAP", "0")
    torchao_mode = os.environ.get("MINWM_TORCHAO_QUANT", "int8wo")
    lighttae_path = os.environ.get(
        "MINWM_LIGHTX2V_VAE_PATH",
        "/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth",
    )
    lightx2v_repo = os.environ.get("LIGHTX2V_REPO", "/root/autodl-tmp/workspace/LightX2V")
    lighttae_parallel = os.environ.get("MINWM_LIGHTX2V_PARALLEL", "1")
    lighttae_output_device = os.environ.get("MINWM_LIGHTX2V_OUTPUT_DEVICE", "cpu")

    common = {
        "DATA_PATH": prompts,
        "NUM_OUTPUT_FRAMES": frames,
        "MAX_PROMPTS": bench_prompts,
        "CHECKPOINT_PATH": checkpoint,
        "TORCHINDUCTOR_USE_CUDAGRAPHS": os.environ.get("TORCHINDUCTOR_USE_CUDAGRAPHS", "0"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    optimized_common = {
        **common,
        "MINWM_COMPILE": os.environ.get("MINWM_COMPILE", "0"),
        "MINWM_CLEANUP_EACH_SAMPLE": os.environ.get("MINWM_CLEANUP_EACH_SAMPLE", "1"),
        "MINWM_ASYNC_VIDEO_WRITER": os.environ.get("MINWM_ASYNC_VIDEO_WRITER", "1"),
        "MINWM_LLV2_CACHE_QUANT": os.environ.get("MINWM_LLV2_CACHE_QUANT", "0"),
        "MINWM_OFFLOAD_GENERATOR_BEFORE_VAE": "0",
        "MINWM_VAE_TEMPORAL_CHUNK": vae_chunk,
        "MINWM_VAE_CHUNK_OVERLAP": vae_overlap,
    }

    cases: list[BenchCase] = []
    if case_enabled("baseline", True):
        cases.append(
            BenchCase(
                name="baseline_offload",
                output_dir=output_name(output_prefix, "baseline_offload"),
                env={
                    **common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "baseline_offload"),
                    "MINWM_COMPILE": "0",
                    "MINWM_CLEANUP_EACH_SAMPLE": "0",
                    "MINWM_ASYNC_VIDEO_WRITER": "0",
                    "MINWM_LLV2_CACHE_QUANT": "0",
                    "MINWM_OFFLOAD_GENERATOR_BEFORE_VAE": os.environ.get("BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE", "1"),
                    "MINWM_VAE_BACKEND": "wan",
                    "MINWM_VAE_TEMPORAL_CHUNK": "0",
                    "MINWM_VAE_CHUNK_OVERLAP": "0",
                    "MINWM_TORCHAO_QUANT": "none",
                    "MINWM_PROFILE_JSONL": str(result_dir / "baseline_offload_profile_rank0.jsonl"),
                },
            )
        )
    if case_enabled("wan_chunk", True):
        cases.append(
            BenchCase(
                name="wan_chunk",
                output_dir=output_name(output_prefix, "wan_chunk"),
                env={
                    **optimized_common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "wan_chunk"),
                    "MINWM_VAE_BACKEND": "wan",
                    "MINWM_TORCHAO_QUANT": "none",
                    "MINWM_PROFILE_JSONL": str(result_dir / "wan_chunk_profile_rank0.jsonl"),
                },
            )
        )
    if case_enabled("wan_chunk_torchao", True):
        cases.append(
            BenchCase(
                name="wan_chunk_torchao",
                output_dir=output_name(output_prefix, "wan_chunk_torchao"),
                env={
                    **optimized_common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "wan_chunk_torchao"),
                    "MINWM_VAE_BACKEND": "wan",
                    "MINWM_TORCHAO_QUANT": torchao_mode,
                    "MINWM_TORCHAO_STRICT": os.environ.get("MINWM_TORCHAO_STRICT", "0"),
                    "MINWM_PROFILE_JSONL": str(result_dir / "wan_chunk_torchao_profile_rank0.jsonl"),
                },
            )
        )
    if case_enabled("paper_lighttae", True):
        cases.append(
            BenchCase(
                name="paper_lighttae",
                output_dir=output_name(output_prefix, "paper_lighttae"),
                env={
                    **optimized_common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "paper_lighttae"),
                    "MINWM_VAE_BACKEND": "lightx2v_tae",
                    "MINWM_LIGHTX2V_VAE_PATH": lighttae_path,
                    "MINWM_LIGHTX2V_DTYPE": os.environ.get("MINWM_LIGHTX2V_DTYPE", "bfloat16"),
                    "MINWM_LIGHTX2V_NEED_SCALED": os.environ.get("MINWM_LIGHTX2V_NEED_SCALED", "1"),
                    "MINWM_LIGHTX2V_PARALLEL": lighttae_parallel,
                    "MINWM_LIGHTX2V_OUTPUT_DEVICE": lighttae_output_device,
                    "LIGHTX2V_REPO": lightx2v_repo,
                    "MINWM_TORCHAO_QUANT": "none",
                    "MINWM_PROFILE_JSONL": str(result_dir / "paper_lighttae_profile_rank0.jsonl"),
                },
            )
        )
    if case_enabled("paper_lighttae_fast3", False):
        cases.append(
            BenchCase(
                name="paper_lighttae_fast3",
                output_dir=output_name(output_prefix, "paper_lighttae_fast3"),
                env={
                    **optimized_common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "paper_lighttae_fast3"),
                    "MINWM_VAE_BACKEND": "lightx2v_tae",
                    "MINWM_LIGHTX2V_VAE_PATH": lighttae_path,
                    "MINWM_LIGHTX2V_DTYPE": os.environ.get("MINWM_LIGHTX2V_DTYPE", "bfloat16"),
                    "MINWM_LIGHTX2V_NEED_SCALED": os.environ.get("MINWM_LIGHTX2V_NEED_SCALED", "1"),
                    "MINWM_LIGHTX2V_PARALLEL": lighttae_parallel,
                    "MINWM_LIGHTX2V_OUTPUT_DEVICE": lighttae_output_device,
                    "LIGHTX2V_REPO": lightx2v_repo,
                    "MINWM_DENOISING_STEP_LIST": os.environ.get("MINWM_FAST3_DENOISING_STEP_LIST", "1000,500,250"),
                    "MINWM_TORCHAO_QUANT": "none",
                    "MINWM_PROFILE_JSONL": str(result_dir / "paper_lighttae_fast3_profile_rank0.jsonl"),
                },
            )
        )
    if case_enabled("paper_lighttae_torchao", True):
        cases.append(
            BenchCase(
                name="paper_lighttae_torchao",
                output_dir=output_name(output_prefix, "paper_lighttae_torchao"),
                env={
                    **optimized_common,
                    "OUTPUT_FOLDER": output_name(output_prefix, "paper_lighttae_torchao"),
                    "MINWM_VAE_BACKEND": "lightx2v_tae",
                    "MINWM_LIGHTX2V_VAE_PATH": lighttae_path,
                    "MINWM_LIGHTX2V_DTYPE": os.environ.get("MINWM_LIGHTX2V_DTYPE", "bfloat16"),
                    "MINWM_LIGHTX2V_NEED_SCALED": os.environ.get("MINWM_LIGHTX2V_NEED_SCALED", "1"),
                    "MINWM_LIGHTX2V_PARALLEL": lighttae_parallel,
                    "MINWM_LIGHTX2V_OUTPUT_DEVICE": lighttae_output_device,
                    "LIGHTX2V_REPO": lightx2v_repo,
                    "MINWM_TORCHAO_QUANT": torchao_mode,
                    "MINWM_TORCHAO_STRICT": os.environ.get("MINWM_TORCHAO_STRICT", "0"),
                    "MINWM_PROFILE_JSONL": str(result_dir / "paper_lighttae_torchao_profile_rank0.jsonl"),
                },
            )
        )

    if not cases:
        print("No cases enabled. Set RUN_BASELINE=1 or another RUN_* variable.", file=sys.stderr)
        return 1

    print(f"[bench-all] prompts={prompts} frames={frames} bench_prompts={bench_prompts}")
    print(f"[bench-all] checkpoint={checkpoint}")
    print(f"[bench-all] lighttae_checkpoint={lighttae_path}")
    print(f"[bench-all] results will be written to {result_dir}")

    rows = []
    for case in cases:
        print(f"[bench-all] running {case.name}")
        row = run_case(case, run_script, result_dir)
        rows.append(add_row_metadata(row, case))

    write_results(result_dir, run_id, rows, prompts, frames, bench_prompts, checkpoint, lighttae_path)
    return 0 if all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
