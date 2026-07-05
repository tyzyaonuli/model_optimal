#!/usr/bin/env python3
"""Compare the baseline against experimental original-Wan-VAE optimizations.

Run this from the official minWM repo root:

    python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_wan_experimental.py

This is separate from the LightTAE benchmark because the original Wan VAE chunk
path can still OOM on a single 4090 depending on frame count and free VRAM.
"""

from __future__ import annotations

import os

from bench_minwm_dmd_4090_all import main


os.environ.setdefault("RUN_BASELINE", "1")
os.environ.setdefault("RUN_WAN_CHUNK", "1")
os.environ.setdefault("RUN_WAN_CHUNK_TORCHAO", "1")
os.environ.setdefault("RUN_PAPER_LIGHTTAE", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE_TORCHAO", "0")
os.environ.setdefault("OUTPUT_PREFIX", "bench_dmd_4090_wan_experimental")
os.environ.setdefault("RUN_ID_SUFFIX", "wan_experimental")


if __name__ == "__main__":
    raise SystemExit(main())
