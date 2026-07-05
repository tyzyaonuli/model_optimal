#!/usr/bin/env python3
"""Compatibility entry point for the original Wan-VAE optimization benchmark.

Historically this script had its own `baseline` case. It now delegates to the
same shared benchmark implementation used by the LightTAE/combined scripts, so
all comparisons use the same `baseline_offload` construction.
"""

from __future__ import annotations

import os

from bench_minwm_dmd_4090_all import main


os.environ.setdefault("RUN_BASELINE", "1")
os.environ.setdefault("RUN_WAN_CHUNK", "1")
os.environ.setdefault("RUN_WAN_CHUNK_TORCHAO", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE_TORCHAO", "0")
os.environ.setdefault("OUTPUT_PREFIX", "bench_dmd_4090_wan")
os.environ.setdefault("RUN_ID_SUFFIX", "wan")
os.environ.setdefault("MINWM_VAE_BACKEND", "wan")
os.environ.setdefault("MINWM_VAE_TEMPORAL_CHUNK", "2")
os.environ.setdefault("MINWM_VAE_CHUNK_OVERLAP", "0")
os.environ.setdefault("MINWM_ASYNC_VIDEO_WRITER", "1")
os.environ.setdefault("MINWM_CLEANUP_EACH_SAMPLE", "1")
os.environ.setdefault("MINWM_TORCHAO_QUANT", "none")


if __name__ == "__main__":
    raise SystemExit(main())
