#!/usr/bin/env python3
"""Recommended combined RTX 4090 benchmark.

This compares the safe offload baseline against the combined optimized path:

- LightTAE / LightTAEW 2.1 final VAE decode
- no generator offload on the optimized row
- async video writer
- no per-sample CUDA cache purge
- decoded LightTAE video kept on CUDA for speed
- lazy original Wan VAE loading, so LightTAE runs do not load Wan2.1_VAE.pth

TorchAO and original-Wan-VAE chunk decode are intentionally disabled here. They
remain available in the experimental scripts, but were slower or less stable in
the measured single-RTX-4090 runs.
"""

from __future__ import annotations

import os

from bench_minwm_dmd_4090_all import main


os.environ.setdefault("RUN_BASELINE", "1")
os.environ.setdefault("RUN_WAN_CHUNK", "0")
os.environ.setdefault("RUN_WAN_CHUNK_TORCHAO", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE", "1")
os.environ.setdefault("RUN_PAPER_LIGHTTAE_TORCHAO", "0")
os.environ.setdefault("RUN_PAPER_LIGHTTAE_FAST3", "0")
os.environ.setdefault("OUTPUT_PREFIX", "bench_dmd_4090_combined")
os.environ.setdefault("RUN_ID_SUFFIX", "combined")

os.environ.setdefault("MINWM_COMPILE", "0")
os.environ.setdefault("MINWM_ASYNC_VIDEO_WRITER", "1")
os.environ.setdefault("MINWM_CLEANUP_EACH_SAMPLE", "0")
os.environ.setdefault("MINWM_LLV2_CACHE_QUANT", "0")
os.environ.setdefault("MINWM_LIGHTX2V_OUTPUT_DEVICE", "cuda")
os.environ.setdefault("MINWM_LAZY_WAN_VAE", "1")
os.environ.setdefault("MINWM_TORCHAO_QUANT", "none")


if __name__ == "__main__":
    raise SystemExit(main())
