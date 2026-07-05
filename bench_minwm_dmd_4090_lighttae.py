#!/usr/bin/env python3
"""Compare the safe minWM baseline against the paper-style LightTAE path.

Run this from the official minWM repo root:

    python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_lighttae.py

This entry point intentionally excludes the experimental Wan VAE chunk cases so
their OOM/layout behavior cannot interrupt the LightTAE reproduction.
"""

from __future__ import annotations

import os

from bench_minwm_dmd_4090_all import main


os.environ.setdefault("RUN_BASELINE", "1")
os.environ.setdefault("RUN_WAN_CHUNK", "0")
os.environ.setdefault("RUN_WAN_CHUNK_TORCHAO", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE", "1")
os.environ.setdefault("RUN_PAPER_LIGHTTAE_TORCHAO", "0")
os.environ.setdefault("OUTPUT_PREFIX", "bench_dmd_4090_lighttae")
os.environ.setdefault("RUN_ID_SUFFIX", "lighttae")
os.environ.setdefault("MINWM_ASYNC_VIDEO_WRITER", "1")
os.environ.setdefault("MINWM_CLEANUP_EACH_SAMPLE", "0")
os.environ.setdefault("MINWM_LIGHTX2V_OUTPUT_DEVICE", "cuda")
os.environ.setdefault("MINWM_LAZY_WAN_VAE", "1")
os.environ.setdefault("MINWM_TORCHAO_QUANT", "none")


if __name__ == "__main__":
    raise SystemExit(main())
