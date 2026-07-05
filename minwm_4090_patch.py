#!/usr/bin/env python3
"""Patch official minWM Wan21 DMD inference for RTX 4090 deployment.

This script does not download models. Run it from, or point it at, a cloned
`shengshu-ai/minWM` repository after you have prepared the base Wan2.1 model
and MIN-Lab/minWM Wan21/Action2V/dmd checkpoint yourself.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


PATCH_MARK = "# ---- minwm 4090 dmd overlay ----"


def _insert_after(text: str, needle: str, payload: str) -> str:
    if payload.strip() in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find patch anchor: {needle!r}")
    return text.replace(needle, needle + payload, 1)


def _replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor: {old!r}")
    return text.replace(old, new, 1)


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _insert_after_line_contains(text: str, contains: str, payload_builder) -> str:
    if contains not in text:
        raise RuntimeError(f"Could not find patch anchor containing: {contains!r}")
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if contains in line:
            payload = payload_builder(line)
            if "".join(payload).strip() in text:
                return text
            lines[index + 1:index + 1] = payload
            return "".join(lines)
    raise RuntimeError(f"Could not find patch anchor containing: {contains!r}")


def _replace_line_contains(text: str, contains: str, payload_builder) -> str:
    if contains not in text:
        raise RuntimeError(f"Could not find patch anchor containing: {contains!r}")
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if contains in line:
            payload = payload_builder(line)
            if "".join(payload).strip() in text:
                return text
            lines[index:index + 1] = payload
            return "".join(lines)
    raise RuntimeError(f"Could not find patch anchor containing: {contains!r}")


def _dedupe_parser_args(text: str, options: tuple[str, ...]) -> str:
    seen: set[str] = set()
    lines = []
    for line in text.splitlines(keepends=True):
        matched = next((option for option in options if f'parser.add_argument("{option}"' in line), None)
        if matched is not None:
            if matched in seen:
                continue
            seen.add(matched)
        lines.append(line)
    return "".join(lines)


def _remove_nested_duplicate_stage(text: str, stage_name: str) -> str:
    """Collapse accidental nested identical profiler contexts from old patches."""
    needle = f"with llv2_runtime.stage('{stage_name}'):"
    lines = text.splitlines(keepends=True)
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        if needle in line and index + 1 < len(lines) and needle in lines[index + 1]:
            outer_indent = _indent_of(line)
            inner_indent = _indent_of(lines[index + 1])
            index += 2
            while index < len(lines):
                nested_line = lines[index]
                if nested_line.startswith(inner_indent):
                    output.append(outer_indent + nested_line[len(inner_indent):])
                    index += 1
                    continue
                break
            continue
        index += 1
    return "".join(output)


def _insert_missing_parser_args(text: str, anchor: str, arg_lines: tuple[str, ...]) -> str:
    missing = []
    for line in arg_lines:
        option = line.split('"', 2)[1]
        if f'parser.add_argument("{option}"' not in text:
            missing.append(line)
    if not missing:
        return text
    return _insert_after(text, anchor, "".join(missing))


def patch_wan_inference(wan_inference: Path) -> None:
    text = wan_inference.read_text(encoding="utf-8")
    original = text
    overlay_options = (
        "--low_vram",
        "--tf32",
        "--compile_generator",
        "--compile_mode",
        "--cleanup_each_sample",
        "--async_video_writer",
        "--llv2_cache_quant",
        "--llv2_cache_min_numel",
        "--max_prompts",
    )
    text = _dedupe_parser_args(text, overlay_options)
    backup = wan_inference.with_suffix(wan_inference.suffix + ".before_4090_overlay")
    if not backup.exists():
        shutil.copy2(wan_inference, backup)
    elif PATCH_MARK in text:
        text = backup.read_text(encoding="utf-8")
        original = text

    text = _remove_nested_duplicate_stage(text, "pipeline_inference")
    text = _remove_nested_duplicate_stage(text, "write_video_submit")

    text = _insert_missing_parser_args(
        text,
        'parser.add_argument("--trajectory_path", type=str, default=None, help="Path to trajectory file (one trajectory string per line, aligned with data_path)")\n',
        (
            'parser.add_argument("--low_vram", action="store_true", help="Use RTX 4090/24GB low-VRAM runtime defaults")\n',
            'parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on Ada GPUs")\n',
            'parser.add_argument("--compile_generator", action="store_true", help="torch.compile the Wan generator for repeated prompts")\n',
            'parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])\n',
            'parser.add_argument("--cleanup_each_sample", action="store_true", help="Free transient tensors and CUDA cache after each video write")\n',
            'parser.add_argument("--async_video_writer", action="store_true", help="Write videos in a background thread so CPU encoding overlaps later prompts")\n',
            'parser.add_argument("--llv2_cache_quant", action="store_true", help="LongLive-2 style INT8 cache/history compression for RTX 4090")\n',
            'parser.add_argument("--llv2_cache_min_numel", type=int, default=16384, help="Minimum tensor size for cache compression")\n',
            'parser.add_argument("--max_prompts", type=int, default=0, help="Limit prompts for smoke tests; 0 means all")\n',
        ),
    )

    text = _insert_after(
        text,
        "args = parser.parse_args()\n",
        f"{PATCH_MARK}\n"
        "import atexit\n"
        "from llv2_minwm_runtime import LongLive2Runtime\n"
        "os.makedirs(args.output_folder, exist_ok=True)\n"
        "os.environ.setdefault(\n"
        "    'MINWM_PROFILE_JSONL',\n"
        "    os.path.join(args.output_folder, f\"minwm_profile_rank{os.environ.get('LOCAL_RANK', '0')}.jsonl\"),\n"
        ")\n"
        "if args.tf32:\n"
        "    torch.backends.cuda.matmul.allow_tf32 = True\n"
        "    torch.backends.cudnn.allow_tf32 = True\n"
        "    torch.set_float32_matmul_precision('high')\n"
        "if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:\n"
        "    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'\n"
        "llv2_runtime = LongLive2Runtime(args)\n"
        "atexit.register(llv2_runtime.close)\n",
    )

    text = _replace_once(
        text,
        "low_memory = get_cuda_free_memory_gb(gpu) < 40\n",
        "low_memory = args.low_vram or get_cuda_free_memory_gb(gpu) < 40\n",
    )

    text = _insert_after(
        text,
        "config = OmegaConf.merge(default_config, config)\n",
        "_minwm_steps = os.environ.get('MINWM_DENOISING_STEP_LIST', '').strip()\n"
        "if _minwm_steps:\n"
        "    config.denoising_step_list = [int(x.strip()) for x in _minwm_steps.split(',') if x.strip()]\n"
        "    print(f\"[4090] override denoising_step_list={list(config.denoising_step_list)}\")\n",
    )

    text = _insert_after_line_contains(
        text,
        "pipeline.generator.to(device=gpu)",
        lambda line: [
            f"{_indent_of(line)}if args.compile_generator:\n",
            f"{_indent_of(line)}    print(f'[4090] torch.compile(generator, mode={{args.compile_mode}})')\n",
            f"{_indent_of(line)}    pipeline.generator = torch.compile(pipeline.generator, mode=args.compile_mode)\n",
        ],
    )

    text = _insert_after_line_contains(
        text,
        "num_prompts = len(dataset)",
        lambda line: [
            f"{_indent_of(line)}if args.max_prompts > 0:\n",
            f"{_indent_of(line)}    from torch.utils.data import Subset\n",
            f"{_indent_of(line)}    dataset = Subset(dataset, range(min(args.max_prompts, len(dataset))))\n",
            f"{_indent_of(line)}    num_prompts = len(dataset)\n",
        ],
    )

    text = _insert_after_line_contains(
        text,
        "# Generate frames",
        lambda line: [
            f"{_indent_of(line)}if args.llv2_cache_quant:\n",
            f"{_indent_of(line)}    if 'clean_latent' in locals():\n",
            f"{_indent_of(line)}        clean_latent = llv2_runtime.prepare_cache_for_use(clean_latent)\n",
            f"{_indent_of(line)}    if 'current_video' in locals():\n",
            f"{_indent_of(line)}        current_video = llv2_runtime.prepare_cache_for_use(current_video)\n",
            f"{_indent_of(line)}    if 'latents' in locals():\n",
            f"{_indent_of(line)}        latents = llv2_runtime.prepare_cache_for_use(latents)\n",
            f"{_indent_of(line)}llv2_runtime.cleanup_cuda()\n",
        ],
    )

    text = _replace_line_contains(
        text,
        "video, latents = pipeline.inference(",
        lambda line: [
            f"{_indent_of(line)}with llv2_runtime.stage('pipeline_inference'):\n",
            f"{_indent_of(line)}    with torch.inference_mode():\n",
            f"{_indent_of(line)}        {line.lstrip()}",
        ],
    )

    if "with llv2_runtime.stage('write_video_submit')" not in text:
        write_anchor = "write_video(output_path, video[0], fps=16)"
        if write_anchor not in text:
            write_anchor = "llv2_runtime.write_video(write_video, output_path, video[0], fps=16)"
        text = _replace_line_contains(
            text,
            write_anchor,
            lambda line: [
                f"{_indent_of(line)}with llv2_runtime.stage('write_video_submit'):\n",
                f"{_indent_of(line)}    llv2_runtime.write_video(write_video, output_path, video[0], fps=16)\n",
            ],
        )

    text = _replace_once(
        text,
        "    # Clear VAE cache\n"
        "    pipeline.vae.model.clear_cache()\n",
        "    # Clear VAE cache when the original Wan VAE is loaded.\n"
        "    if getattr(pipeline.vae, 'model', None) is not None:\n"
        "        pipeline.vae.model.clear_cache()\n",
    )

    text = _insert_after_line_contains(
        text,
        "llv2_runtime.write_video(write_video",
        lambda line: [
            f"{_indent_of(line)}if args.llv2_cache_quant:\n",
            f"{_indent_of(line)}    if 'clean_latent' in locals():\n",
            f"{_indent_of(line)}        clean_latent = llv2_runtime.maybe_compress_cache(clean_latent, name='clean_latent')\n",
            f"{_indent_of(line)}    if 'current_video' in locals():\n",
            f"{_indent_of(line)}        current_video = llv2_runtime.maybe_compress_cache(current_video, name='current_video')\n",
            f"{_indent_of(line)}    if 'latents' in locals():\n",
            f"{_indent_of(line)}        latents = llv2_runtime.maybe_compress_cache(latents, name='latents')\n",
            f"{_indent_of(line)}if args.cleanup_each_sample:\n",
            f"{_indent_of(line)}    if 'video' in locals():\n",
            f"{_indent_of(line)}        del video\n",
            f"{_indent_of(line)}    if 'current_video' in locals():\n",
            f"{_indent_of(line)}        del current_video\n",
            f"{_indent_of(line)}    if 'clean_latent' in locals():\n",
            f"{_indent_of(line)}        del clean_latent\n",
            f"{_indent_of(line)}    if 'latents' in locals():\n",
            f"{_indent_of(line)}        del latents\n",
            f"{_indent_of(line)}    llv2_runtime.cleanup_cuda()\n",
        ],
    )

    if text == original:
        print(f"Already patched: {wan_inference}")
        return
    wan_inference.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(wan_inference), doraise=True)
    except py_compile.PyCompileError:
        shutil.copy2(backup, wan_inference)
        raise
    print(f"Patched: {wan_inference}")
    print(f"Backup:  {backup}")


def patch_causal_inference(causal_inference: Path) -> None:
    """Patch VAE decode OOM with temporal chunk decode and optional offload."""
    if not causal_inference.exists():
        print(f"Skip missing: {causal_inference}")
        return

    text = causal_inference.read_text(encoding="utf-8")
    original = text
    backup = causal_inference.with_suffix(causal_inference.suffix + ".before_4090_overlay")
    if not backup.exists():
        shutil.copy2(causal_inference, backup)
    elif "MINWM_VAE_BACKEND" in text:
        text = backup.read_text(encoding="utf-8")
        original = text
    elif "decode_to_pixel_memory_efficient" in text and "profile_stage('vae_decode')" not in text:
        text = backup.read_text(encoding="utf-8")
        original = text
    elif "MINWM_OFFLOAD_GENERATOR_BEFORE_VAE" in text and "decode_to_pixel_memory_efficient" not in text:
        text = backup.read_text(encoding="utf-8")
        original = text

    def _disable_chunk0_sync(src: str) -> str:
        lines = src.splitlines(True)
        idx = None
        for i in range(len(lines) - 2):
            if (
                lines[i].strip() == "torch.cuda.synchronize()"
                and lines[i + 1].strip() == "_chunk0_t0 = time.perf_counter()"
                and lines[i + 2].strip() == "self.last_chunk0_latency = None"
            ):
                idx = i
                break
        if idx is not None and "MINWM_RECORD_CHUNK0_LATENCY" not in "".join(lines[max(0, idx - 8): idx + 8]):
            block_start = max(0, idx - 2)
            block_end = idx + 3
            lines = lines[:block_start] + [
                "        # Start chunk0 latency timer AFTER text encoder, BEFORE VAE decode.\n",
                "        # Disabled by default for throughput benchmarks because synchronize()\n",
                "        # serializes the CPU and GPU on every prompt.\n",
                "        import os as _minwm_os\n",
                "        _record_chunk0_latency = _minwm_os.environ.get('MINWM_RECORD_CHUNK0_LATENCY', '0').lower() in ('1', 'true', 'yes', 'on')\n",
                "        if _record_chunk0_latency:\n",
                "            torch.cuda.synchronize()\n",
                "            _chunk0_t0 = time.perf_counter()\n",
                "        else:\n",
                "            _chunk0_t0 = None\n",
                "        self.last_chunk0_latency = None\n",
            ] + lines[block_end:]

        idx = None
        for i, line in enumerate(lines):
            if "self.last_chunk0_latency = time.perf_counter() - _chunk0_t0" in line:
                idx = i
                break
        if idx is not None:
            block_start = idx
            while block_start > 0:
                if lines[block_start].strip().startswith("# Capture chunk0 latency"):
                    break
                block_start -= 1
            if "_record_chunk0_latency" not in "".join(lines[block_start: idx + 1]):
                block_end = idx + 1
                lines = lines[:block_start] + [
                    "            # Capture chunk0 latency only when explicitly requested.\n",
                    "            if _record_chunk0_latency and self.last_chunk0_latency is None:\n",
                    "                torch.cuda.synchronize()\n",
                    "                self.last_chunk0_latency = time.perf_counter() - _chunk0_t0\n",
                    "\n",
                ] + lines[block_end:]
        return "".join(lines)

    text = _disable_chunk0_sync(text)

    text = _replace_line_contains(
        text,
        "video = self.vae.decode_to_pixel(output, use_cache=False)",
        lambda line: (
            lambda indent: [
                f"{indent}from llv2_minwm_runtime import decode_to_pixel_memory_efficient, profile_stage\n",
                f"{indent}from lightx2v_minwm_runtime import build_lightx2v_vae_from_env\n",
                f"{indent}_minwm_vae_backend = __import__('os').environ.get('MINWM_VAE_BACKEND', 'wan').lower()\n",
                f"{indent}_minwm_vae_chunk = int(__import__('os').environ.get('MINWM_VAE_TEMPORAL_CHUNK', '0'))\n",
                f"{indent}_minwm_vae_overlap = int(__import__('os').environ.get('MINWM_VAE_CHUNK_OVERLAP', '0'))\n",
                f"{indent}_minwm_vae_offload = __import__('os').environ.get('MINWM_OFFLOAD_GENERATOR_BEFORE_VAE', '0').lower() in ('1', 'true', 'yes')\n",
                f"{indent}_minwm_generator_device = None\n",
                f"{indent}with profile_stage('vae_decode'):\n",
                f"{indent}    if _minwm_vae_backend in ('lightx2v', 'lightx2v_tae', 'lighttae'):\n",
                f"{indent}        if not hasattr(self, '_minwm_lightx2v_vae'):\n",
                f"{indent}            self._minwm_lightx2v_vae = build_lightx2v_vae_from_env()\n",
                f"{indent}        video = self._minwm_lightx2v_vae.decode_to_pixel(output, use_cache=False)\n",
                f"{indent}    elif _minwm_vae_chunk > 0:\n",
                f"{indent}        video = decode_to_pixel_memory_efficient(self.vae, output, use_cache=False, chunk_size=_minwm_vae_chunk, overlap=_minwm_vae_overlap)\n",
                f"{indent}    else:\n",
                f"{indent}        if _minwm_vae_offload and hasattr(self, 'generator'):\n",
                f"{indent}            try:\n",
                f"{indent}                _minwm_generator_device = next(self.generator.parameters()).device\n",
                f"{indent}            except Exception:\n",
                f"{indent}                _minwm_generator_device = None\n",
                f"{indent}            self.generator.to('cpu')\n",
                f"{indent}            if torch.cuda.is_available():\n",
                f"{indent}                torch.cuda.empty_cache()\n",
                f"{indent}        video = self.vae.decode_to_pixel(output, use_cache=False)\n",
                f"{indent}        if _minwm_vae_offload:\n",
                f"{indent}            try:\n",
                f"{indent}                video = video.cpu()\n",
                f"{indent}            except Exception:\n",
                f"{indent}                pass\n",
                f"{indent}            if torch.cuda.is_available():\n",
                f"{indent}                torch.cuda.empty_cache()\n",
                f"{indent}            if _minwm_generator_device is not None and hasattr(self, 'generator'):\n",
                f"{indent}                self.generator.to(_minwm_generator_device)\n",
                f"{indent}                if torch.cuda.is_available():\n",
                f"{indent}                    torch.cuda.empty_cache()\n",
            ]
        )(_indent_of(line)),
    )

    if text == original:
        print(f"Already patched: {causal_inference}")
        return
    causal_inference.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(causal_inference), doraise=True)
    except py_compile.PyCompileError:
        shutil.copy2(backup, causal_inference)
        raise
    print(f"Patched: {causal_inference}")
    print(f"Backup:  {backup}")


def patch_wan_wrapper(wan_wrapper: Path) -> None:
    """Lazy-load the heavy Wan VAE when LightTAE replaces decode."""
    if not wan_wrapper.exists():
        print(f"Skip missing: {wan_wrapper}")
        return

    text = wan_wrapper.read_text(encoding="utf-8")
    original = text
    backup = wan_wrapper.with_suffix(wan_wrapper.suffix + ".before_4090_overlay")
    if not backup.exists():
        shutil.copy2(wan_wrapper, backup)
    elif "MINWM_LAZY_WAN_VAE" in text:
        text = backup.read_text(encoding="utf-8")
        original = text

    text = _insert_after(
        text,
        "import types\n",
        "import os\n",
    )

    text = _replace_once(
        text,
        "        # init model\n"
        "        self.model = _video_vae(\n"
        "            pretrained_path=\"Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\",\n"
        "            z_dim=16,\n"
        "        ).eval().requires_grad_(False)\n",
        "        self.model = None\n"
        "        self._lazy_model_path = \"Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\"\n"
        "        self._lazy_z_dim = 16\n"
        "        backend = os.environ.get(\"MINWM_VAE_BACKEND\", \"wan\").lower()\n"
        "        lazy = os.environ.get(\"MINWM_LAZY_WAN_VAE\", \"1\").lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
        "        if not (lazy and backend in (\"lightx2v\", \"lightx2v_tae\", \"lighttae\")):\n"
        "            self._ensure_model_loaded()\n",
    )

    text = _insert_after(
        text,
        "        if not (lazy and backend in (\"lightx2v\", \"lightx2v_tae\", \"lighttae\")):\n"
        "            self._ensure_model_loaded()\n",
        "\n"
        "    def _ensure_model_loaded(self) -> None:\n"
        "        if self.model is None:\n"
        "            self.model = _video_vae(\n"
        "                pretrained_path=self._lazy_model_path,\n"
        "                z_dim=self._lazy_z_dim,\n"
        "            ).eval().requires_grad_(False)\n",
    )

    text = _replace_once(
        text,
        "    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:\n"
        "        # pixel: [batch_size, num_channels, num_frames, height, width]\n",
        "    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:\n"
        "        self._ensure_model_loaded()\n"
        "        # pixel: [batch_size, num_channels, num_frames, height, width]\n",
    )
    text = _replace_once(
        text,
        "    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:\n"
        "        # from [batch_size, num_frames, num_channels, height, width]\n",
        "    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:\n"
        "        self._ensure_model_loaded()\n"
        "        # from [batch_size, num_frames, num_channels, height, width]\n",
    )

    if text == original:
        print(f"Already patched: {wan_wrapper}")
        return
    wan_wrapper.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(wan_wrapper), doraise=True)
    except py_compile.PyCompileError:
        shutil.copy2(backup, wan_wrapper)
        raise
    print(f"Patched: {wan_wrapper}")
    print(f"Backup:  {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the cloned shengshu-ai/minWM repository",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    wan_inference = repo / "Wan21" / "wan_inference.py"
    if not wan_inference.exists():
        raise SystemExit(f"Missing {wan_inference}; pass --repo /path/to/minWM")
    runtime_src = Path(__file__).with_name("llv2_minwm_runtime.py")
    runtime_dst = repo / "Wan21" / "llv2_minwm_runtime.py"
    if not runtime_src.exists():
        raise SystemExit(f"Missing runtime helper: {runtime_src}")
    shutil.copy2(runtime_src, runtime_dst)
    print(f"Copied: {runtime_dst}")
    torchao_src = Path(__file__).with_name("minwm_torchao_runtime.py")
    if torchao_src.exists():
        torchao_dst = repo / "Wan21" / "minwm_torchao_runtime.py"
        shutil.copy2(torchao_src, torchao_dst)
        print(f"Copied: {torchao_dst}")
    lightx2v_src = Path(__file__).with_name("lightx2v_minwm_runtime.py")
    if lightx2v_src.exists():
        lightx2v_dst = repo / "Wan21" / "lightx2v_minwm_runtime.py"
        shutil.copy2(lightx2v_src, lightx2v_dst)
        print(f"Copied: {lightx2v_dst}")
    standalone_src = Path(__file__).with_name("lightx2v_standalone_loader.py")
    if standalone_src.exists():
        standalone_dst = repo / "Wan21" / "lightx2v_standalone_loader.py"
        shutil.copy2(standalone_src, standalone_dst)
        print(f"Copied: {standalone_dst}")
    patch_wan_inference(wan_inference)
    patch_causal_inference(repo / "Wan21" / "pipeline" / "causal_inference.py")
    patch_wan_wrapper(repo / "Wan21" / "wan_utils" / "wan_wrapper.py")


if __name__ == "__main__":
    main()
