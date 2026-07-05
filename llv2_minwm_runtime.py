#!/usr/bin/env python3
"""LongLive-2 style runtime helpers for minWM inference.

LongLive-2's headline W4A4/NVFP4 path targets Blackwell GPUs. RTX 4090 cannot
run that exact NVFP4 stack, so this module implements the portable parts:
asynchronous video writing and conservative INT8 cache/history compression.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class QuantizedCacheTensor:
    q: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        return (self.q.float() * self.scale).to(self.dtype)


def _cuda_mem_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "free_gb": free / 1024**3,
        "total_gb": total / 1024**3,
    }


def _append_profile_record(record: dict[str, Any]) -> None:
    path = os.environ.get("MINWM_PROFILE_JSONL")
    if not path:
        return
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


@contextmanager
def profile_stage(stage: str):
    """Record wall time and CUDA memory stats for one inference stage."""
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = _cuda_mem_gb()
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = _cuda_mem_gb()
        record = {
            "stage": stage,
            "status": status,
            "elapsed_seconds": elapsed,
            "rank": rank,
            "local_rank": local_rank,
            "pid": os.getpid(),
            "time_unix": time.time(),
        }
        for prefix, stats in (("before", before), ("after", after)):
            for key, value in stats.items():
                record[f"{prefix}_{key}"] = value
        _append_profile_record(record)


def _looks_like_cache_tensor(name: str, tensor: torch.Tensor, min_numel: int) -> bool:
    if tensor.numel() < min_numel:
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    cache_words = ("cache", "kv", "key", "value", "history", "past", "context", "latent")
    return tensor.ndim >= 3 and any(word in name.lower() for word in cache_words)


def quantize_cache_tensor(tensor: torch.Tensor) -> QuantizedCacheTensor:
    source = tensor.detach()
    max_abs = source.float().abs().amax().clamp_min(1e-6)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(source.float() / scale), -127, 127).to(torch.int8)
    return QuantizedCacheTensor(q=q, scale=scale, dtype=tensor.dtype)


def compress_cache_tree(obj: Any, *, name: str = "cache", min_numel: int = 16384) -> Any:
    if isinstance(obj, QuantizedCacheTensor):
        return obj
    if torch.is_tensor(obj):
        return quantize_cache_tensor(obj) if _looks_like_cache_tensor(name, obj, min_numel) else obj
    if isinstance(obj, dict):
        return {key: compress_cache_tree(value, name=f"{name}.{key}", min_numel=min_numel) for key, value in obj.items()}
    if isinstance(obj, list):
        return [compress_cache_tree(value, name=f"{name}.{index}", min_numel=min_numel) for index, value in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(compress_cache_tree(value, name=f"{name}.{index}", min_numel=min_numel) for index, value in enumerate(obj))
    return obj


def dequantize_cache_tree(obj: Any) -> Any:
    if isinstance(obj, QuantizedCacheTensor):
        return obj.dequantize()
    if isinstance(obj, dict):
        return {key: dequantize_cache_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [dequantize_cache_tree(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(dequantize_cache_tree(value) for value in obj)
    return obj


def _slice_dim(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(start, end)
    return tensor[tuple(index)]


def _infer_decoded_time_dim(decoded: torch.Tensor, ctx_len: int, fallback: int = 1) -> int:
    for dim, size in enumerate(decoded.shape):
        if size == ctx_len:
            return dim
    if decoded.ndim >= 5:
        # WanVAEWrapper.decode_to_pixel returns [B, T, C, H, W].
        # If no decoded dimension equals the latent ctx_len, keep concatenation
        # on T; concatenating on C turns RGB into a fake frame/channel axis.
        if decoded.shape[2] in (1, 3, 4):
            return 1
        if decoded.shape[1] in (1, 3, 4):
            return 2
    return min(fallback, decoded.ndim - 1)


def decode_to_pixel_memory_efficient(
    vae,
    latents: Any,
    *,
    use_cache: bool = False,
    chunk_size: int = 0,
    overlap: int = 0,
) -> Any:
    """Decode video latents in temporal chunks to avoid VAE conv3d OOM.

    minWM's `decode_to_pixel` can allocate a large 3D-conv activation. Chunking
    keeps the generator on GPU and reduces only the VAE decode peak, which is
    the OOM observed on RTX 4090.
    """
    if chunk_size <= 0 or not torch.is_tensor(latents) or latents.ndim < 4:
        return vae.decode_to_pixel(latents, use_cache=use_cache)

    added_batch_dim = False
    decode_latents = latents
    if latents.ndim == 4:
        decode_latents = latents.unsqueeze(0)
        added_batch_dim = True

    # WanVAEWrapper.decode_to_pixel expects [B, F, C, H, W], then internally
    # permutes to [B, C, F, H, W]. Slice only F, never latent channel C=16.
    time_dim = 1
    total_frames = int(decode_latents.shape[time_dim])
    if total_frames <= chunk_size:
        decoded = vae.decode_to_pixel(decode_latents, use_cache=use_cache)
        if added_batch_dim and decoded.ndim > 0 and decoded.shape[0] == 1:
            return decoded.squeeze(0)
        return decoded

    overlap = max(0, int(overlap))
    active_chunk_size = max(1, int(chunk_size))

    while active_chunk_size >= 1:
        outputs: list[torch.Tensor] = []
        concat_dim: int | None = None
        try:
            for start in range(0, total_frames, active_chunk_size):
                end = min(total_frames, start + active_chunk_size)
                ctx_start = max(0, start - overlap)
                ctx_end = min(total_frames, end + overlap)
                chunk = _slice_dim(decode_latents, time_dim, ctx_start, ctx_end)

                decoded = vae.decode_to_pixel(chunk, use_cache=use_cache).detach().cpu()
                decoded_time_dim = _infer_decoded_time_dim(decoded, ctx_end - ctx_start)
                if concat_dim is None:
                    concat_dim = decoded_time_dim

                crop_start = start - ctx_start
                crop_end = crop_start + (end - start)
                if decoded.shape[decoded_time_dim] >= crop_end:
                    decoded = _slice_dim(decoded, decoded_time_dim, crop_start, crop_end)

                outputs.append(decoded)
                del chunk, decoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            break
        except torch.OutOfMemoryError:
            for item in outputs:
                del item
            outputs.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if active_chunk_size == 1:
                raise
            next_chunk_size = max(1, active_chunk_size // 2)
            print(
                f"[llv2] VAE chunk decode OOM at chunk={active_chunk_size}; retrying with chunk={next_chunk_size}",
                flush=True,
            )
            active_chunk_size = next_chunk_size

    merged = torch.cat(outputs, dim=concat_dim if concat_dim is not None else 1)
    if added_batch_dim and merged.ndim > 0 and merged.shape[0] == 1:
        return merged.squeeze(0)
    return merged


class AsyncVideoWriter:
    def __init__(self, enabled: bool) -> None:
        self.pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.futures: list[Future] = []

    def submit(self, write_fn, output_path: str, video: torch.Tensor, fps: int) -> None:
        if not torch.is_tensor(video):
            raise TypeError(f"Expected video tensor for write_video, got {type(video)!r}")
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(
                "Expected video layout [T,H,W,3] before torchvision.write_video. "
                f"Got shape={tuple(video.shape)}. Check the pipeline VAE output layout and wan_inference video rearrange step."
            )
        if self.pool is None:
            write_fn(output_path, video, fps=fps)
            return
        self.futures.append(self.pool.submit(write_fn, output_path, video.detach().cpu(), fps=fps))

    def close(self) -> None:
        for future in self.futures:
            future.result()
        if self.pool is not None:
            self.pool.shutdown(wait=True)


class LongLive2Runtime:
    def __init__(self, args) -> None:
        self.cleanup_each_sample = bool(getattr(args, "cleanup_each_sample", False))
        self.cache_quant = bool(getattr(args, "llv2_cache_quant", False))
        self.cache_min_numel = int(getattr(args, "llv2_cache_min_numel", 16384))
        self.video_writer = AsyncVideoWriter(bool(getattr(args, "async_video_writer", False)))

    def maybe_compress_cache(self, obj: Any, name: str = "cache") -> Any:
        if not self.cache_quant:
            return obj
        return compress_cache_tree(obj, name=name, min_numel=self.cache_min_numel)

    def prepare_cache_for_use(self, obj: Any) -> Any:
        if not self.cache_quant:
            return obj
        return dequantize_cache_tree(obj)

    def stage(self, name: str):
        return profile_stage(name)

    def write_video(self, write_fn, output_path: str, video: torch.Tensor, fps: int = 16) -> None:
        self.video_writer.submit(write_fn, output_path, video, fps)

    def cleanup_cuda(self) -> None:
        if self.cleanup_each_sample and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.video_writer.close()
