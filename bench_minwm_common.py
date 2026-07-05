#!/usr/bin/env python3
"""Shared benchmark helpers for minWM overlay entry points."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchCase:
    name: str
    output_dir: str
    env: dict[str, str]


def hms(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    return path if path.is_absolute() else Path.cwd() / path


def load_profile_records(output_dir: str, profile_hint: str = "") -> tuple[list[dict[str, object]], list[str]]:
    if profile_hint:
        files = [Path(profile_hint)]
    else:
        root = _resolve_output_dir(output_dir)
        files = sorted(root.glob("minwm_profile_rank*.jsonl"))
    records: list[dict[str, object]] = []
    for file in files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_profile_file"] = str(file)
                records.append(record)
    return records, [str(file) for file in files]


def dedupe_profile_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    sorted_records = sorted(
        records,
        key=lambda item: (
            str(item.get("stage", "")),
            str(item.get("pid", "")),
            float(item.get("time_unix", 0.0) or 0.0),
        ),
    )
    for record in sorted_records:
        stage = record.get("stage")
        pid = record.get("pid")
        timestamp = float(record.get("time_unix", 0.0) or 0.0)
        elapsed = float(record.get("elapsed_seconds", 0.0) or 0.0)
        is_duplicate = False
        for previous in reversed(deduped):
            if previous.get("stage") != stage or previous.get("pid") != pid:
                continue
            previous_timestamp = float(previous.get("time_unix", 0.0) or 0.0)
            previous_elapsed = float(previous.get("elapsed_seconds", 0.0) or 0.0)
            if abs(timestamp - previous_timestamp) <= 0.25 and abs(elapsed - previous_elapsed) <= 0.25:
                is_duplicate = True
            break
        if is_duplicate:
            continue
        deduped.append(record)
    return deduped


def summarize_profile(records: list[dict[str, object]]) -> dict[str, str]:
    records = dedupe_profile_records(records)
    by_stage: dict[str, float] = {}
    peak_allocated = 0.0
    peak_reserved = 0.0
    min_free: float | None = None
    for record in records:
        stage = str(record.get("stage", "unknown"))
        elapsed = float(record.get("elapsed_seconds", 0.0) or 0.0)
        by_stage[stage] = by_stage.get(stage, 0.0) + elapsed
        peak_allocated = max(peak_allocated, float(record.get("after_max_allocated_gb", 0.0) or 0.0))
        peak_reserved = max(peak_reserved, float(record.get("after_max_reserved_gb", 0.0) or 0.0))
        for key in ("before_free_gb", "after_free_gb"):
            if key in record and record[key] is not None:
                value = float(record[key])
                min_free = value if min_free is None else min(min_free, value)
    return {
        "profile_stage_seconds": json.dumps(by_stage, sort_keys=True),
        "profile_peak_allocated_gb": f"{peak_allocated:.3f}",
        "profile_peak_reserved_gb": f"{peak_reserved:.3f}",
        "profile_min_free_gb": "" if min_free is None else f"{min_free:.3f}",
    }


def run_case(case: BenchCase, run_script: Path, result_dir: Path) -> dict[str, str]:
    log_file = result_dir / f"{case.name}.log"
    env = os.environ.copy()
    env.update(case.env)

    command = [sys.executable, str(run_script)] if run_script.suffix == ".py" else ["bash", str(run_script)]
    start = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(proc.stdout)
        print(proc.stdout, end="")
    elapsed = time.time() - start

    status = "success" if proc.returncode == 0 else f"failed:{proc.returncode}"
    print(f"[bench] {case.name} status={status} elapsed={hms(elapsed)} ({elapsed:.1f}s)")
    profile_records, profile_files = load_profile_records(case.output_dir, env.get("MINWM_PROFILE_JSONL", ""))
    profile_summary = summarize_profile(profile_records)

    return {
        "case": case.name,
        "status": status,
        "elapsed_hms": hms(elapsed),
        "elapsed_seconds": f"{elapsed:.3f}",
        "output_dir": case.output_dir,
        "log_file": str(log_file),
        "compile": case.env.get("MINWM_COMPILE", ""),
        "cleanup_each_sample": case.env.get("MINWM_CLEANUP_EACH_SAMPLE", ""),
        "async_video_writer": case.env.get("MINWM_ASYNC_VIDEO_WRITER", ""),
        "llv2_cache_quant": case.env.get("MINWM_LLV2_CACHE_QUANT", ""),
        "offload_generator_before_vae": case.env.get("MINWM_OFFLOAD_GENERATOR_BEFORE_VAE", ""),
        "vae_temporal_chunk": case.env.get("MINWM_VAE_TEMPORAL_CHUNK", ""),
        "vae_chunk_overlap": case.env.get("MINWM_VAE_CHUNK_OVERLAP", ""),
        "cudagraphs": case.env.get("TORCHINDUCTOR_USE_CUDAGRAPHS", ""),
        "profile_files": ";".join(profile_files),
        **profile_summary,
    }


def write_stage_profile(result_dir: Path, run_id: str, rows: list[dict[str, str]]) -> Path:
    stage_file = result_dir / "stage_profile.csv"
    fieldnames = [
        "run_id",
        "case",
        "stage",
        "calls",
        "total_seconds",
        "mean_seconds",
        "max_peak_allocated_gb",
        "max_peak_reserved_gb",
        "min_free_gb",
    ]
    with stage_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            records, _ = load_profile_records(row["output_dir"], row.get("profile_files", "").split(";")[0])
            records = dedupe_profile_records(records)
            grouped: dict[str, list[dict[str, object]]] = {}
            for record in records:
                grouped.setdefault(str(record.get("stage", "unknown")), []).append(record)
            for stage, items in sorted(grouped.items()):
                total = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in items)
                peak_allocated = max(float(item.get("after_max_allocated_gb", 0.0) or 0.0) for item in items)
                peak_reserved = max(float(item.get("after_max_reserved_gb", 0.0) or 0.0) for item in items)
                free_values = [
                    float(item[key])
                    for item in items
                    for key in ("before_free_gb", "after_free_gb")
                    if key in item and item[key] is not None
                ]
                writer.writerow(
                    {
                        "run_id": run_id,
                        "case": row["case"],
                        "stage": stage,
                        "calls": len(items),
                        "total_seconds": f"{total:.3f}",
                        "mean_seconds": f"{(total / len(items)) if items else 0.0:.3f}",
                        "max_peak_allocated_gb": f"{peak_allocated:.3f}",
                        "max_peak_reserved_gb": f"{peak_reserved:.3f}",
                        "min_free_gb": "" if not free_values else f"{min(free_values):.3f}",
                    }
                )
    return stage_file
