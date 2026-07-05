#!/usr/bin/env python3
"""Run minWM Wan21/Action2V DMD on one GPU with optional 4090 optimizations.

Run from the root of a cloned shengshu-ai/minWM repository. Configuration is
kept environment-variable compatible with the old shell wrappers.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def require_minwm_root(root: Path) -> None:
    if not (root / "Wan21" / "wan_inference.py").exists():
        raise SystemExit("Run from the root of a cloned shengshu-ai/minWM repo.")


def check_flash_attn() -> None:
    if importlib.util.find_spec("flash_attn") is not None:
        return
    raise SystemExit(
        "Missing flash-attn. Install it in the active venv, for example:\n"
        "  pip install ninja packaging wheel setuptools\n"
        "  pip install flash-attn --no-build-isolation"
    )


def setup_env(root: Path) -> None:
    paths = [root / "HY15", root / "Wan21", root / "shared"]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), existing])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MINWM_OFFLOAD_GENERATOR_BEFORE_VAE", "0")
    os.environ.setdefault("MINWM_VAE_TEMPORAL_CHUNK", "0")
    os.environ.setdefault("MINWM_VAE_CHUNK_OVERLAP", "0")
    os.environ.setdefault("TORCHINDUCTOR_USE_CUDAGRAPHS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def patch_repo(root: Path, script_dir: Path) -> None:
    torchao_requested = env_value("MINWM_TORCHAO_QUANT", "none").lower() not in ("none", "0", "false")
    default_patch = "minwm_4090_patch_torchao.py" if torchao_requested else "minwm_4090_patch.py"
    patch_script = Path(env_value("PATCH_SCRIPT", str(script_dir / default_patch)))
    subprocess.run([sys.executable, str(patch_script), "--repo", str(root)], check=True)


def build_extra_args() -> list[str]:
    extra = ["--low_vram", "--tf32"]

    max_prompts = env_value("MAX_PROMPTS", "0")
    if max_prompts != "0":
        extra += ["--max_prompts", max_prompts]

    compile_mode = env_value("MINWM_COMPILE", "0")
    if compile_mode.lower() not in ("0", "false", "none"):
        extra += ["--compile_generator", "--compile_mode", compile_mode]

    if env_bool("MINWM_CLEANUP_EACH_SAMPLE"):
        extra.append("--cleanup_each_sample")
    if env_bool("MINWM_ASYNC_VIDEO_WRITER"):
        extra.append("--async_video_writer")
    if env_bool("MINWM_LLV2_CACHE_QUANT"):
        extra += ["--llv2_cache_quant", "--llv2_cache_min_numel", env_value("MINWM_LLV2_CACHE_MIN_NUMEL", "16384")]

    torchao_mode = env_value("MINWM_TORCHAO_QUANT", "none")
    if torchao_mode.lower() not in ("none", "0", "false"):
        extra += ["--torchao_quant", torchao_mode]
    if env_bool("MINWM_TORCHAO_STRICT"):
        extra.append("--torchao_quant_strict")

    return extra


def build_command() -> list[str]:
    trajectory_path = env_value("TRAJECTORY_PATH", "")
    trajectory_args = ["--trajectory_path", trajectory_path] if trajectory_path else ["--trajectory", env_value("TRAJECTORY", "w*19")]

    return [
        "torchrun",
        "--master_addr",
        env_value("MASTER_ADDR", "localhost"),
        "--master_port",
        env_value("MASTER_PORT", "29622"),
        "--nproc_per_node=1",
        "--nnodes=1",
        "--node_rank=0",
        "Wan21/wan_inference.py",
        "--config_path",
        env_value("CONFIG_PATH", "Wan21/configs/causal_forcing_dmd_camera.yaml"),
        "--output_folder",
        env_value("OUTPUT_FOLDER", "outputs/wan_action2v_dmd_4090"),
        "--checkpoint_path",
        env_value("CHECKPOINT_PATH", "./ckpts/Wan21/Action2V/dmd/model.pt"),
        "--data_path",
        env_value("DATA_PATH", "Wan21/prompts/demos.txt"),
        "--sp_size",
        env_value("SP_SIZE", "1"),
        "--num_output_frames",
        env_value("NUM_OUTPUT_FRAMES", "20"),
        *trajectory_args,
        *build_extra_args(),
    ]


def main() -> int:
    root = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    require_minwm_root(root)
    patch_repo(root, script_dir)
    setup_env(root)
    check_flash_attn()
    return subprocess.run(build_command(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
