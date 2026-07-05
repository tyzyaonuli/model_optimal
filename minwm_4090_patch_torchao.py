#!/usr/bin/env python3
"""Apply the 4090 overlay plus optional TorchAO quantization hooks."""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

from minwm_4090_patch import (
    _dedupe_parser_args,
    _indent_of,
    _insert_after,
    _insert_after_line_contains,
    _insert_missing_parser_args,
    patch_causal_inference,
    patch_wan_inference,
)


TORCHAO_MARK = "# ---- minwm torchao overlay ----"


def patch_torchao(wan_inference: Path) -> None:
    text = wan_inference.read_text(encoding="utf-8")
    original = text
    text = _dedupe_parser_args(text, ("--torchao_quant", "--torchao_quant_strict"))
    backup = wan_inference.with_suffix(wan_inference.suffix + ".before_torchao_overlay")
    if not backup.exists():
        shutil.copy2(wan_inference, backup)

    text = _insert_missing_parser_args(
        text,
        'parser.add_argument("--llv2_cache_min_numel", type=int, default=16384, help="Minimum tensor size for cache compression")\n',
        (
            'parser.add_argument("--torchao_quant", type=str, default="none", choices=["none", "int8wo", "fp8wo", "fp8dq"], help="Optional TorchAO quantization mode for generator")\n',
            'parser.add_argument("--torchao_quant_strict", action="store_true", help="Fail instead of falling back if TorchAO quantization fails")\n',
        ),
    )

    text = _insert_after(
        text,
        "from llv2_minwm_runtime import LongLive2Runtime\n",
        "from minwm_torchao_runtime import apply_torchao_quantization\n",
    )

    if TORCHAO_MARK not in text:
        text = _insert_after_line_contains(
            text,
            "pipeline.generator.to(device=gpu)",
            lambda line: [
                f"{_indent_of(line)}{TORCHAO_MARK}\n",
                f"{_indent_of(line)}if args.torchao_quant != 'none':\n",
                f"{_indent_of(line)}    with llv2_runtime.stage('torchao_quant_generator'):\n",
                f"{_indent_of(line)}        pipeline.generator = apply_torchao_quantization(\n",
                f"{_indent_of(line)}            pipeline.generator,\n",
                f"{_indent_of(line)}            mode=args.torchao_quant,\n",
                f"{_indent_of(line)}            strict=args.torchao_quant_strict,\n",
                f"{_indent_of(line)}            module_name='generator',\n",
                f"{_indent_of(line)}        )\n",
            ],
        )

    if text == original:
        print(f"Already TorchAO patched: {wan_inference}")
        return
    wan_inference.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(wan_inference), doraise=True)
    except py_compile.PyCompileError:
        shutil.copy2(backup, wan_inference)
        raise
    print(f"TorchAO patched: {wan_inference}")
    print(f"TorchAO backup:  {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the cloned shengshu-ai/minWM repository")
    args = parser.parse_args()

    repo = args.repo.resolve()
    wan_inference = repo / "Wan21" / "wan_inference.py"
    if not wan_inference.exists():
        raise SystemExit(f"Missing {wan_inference}; pass --repo /path/to/minWM")

    runtime_src = Path(__file__).with_name("llv2_minwm_runtime.py")
    runtime_dst = repo / "Wan21" / "llv2_minwm_runtime.py"
    torchao_src = Path(__file__).with_name("minwm_torchao_runtime.py")
    torchao_dst = repo / "Wan21" / "minwm_torchao_runtime.py"
    lightx2v_src = Path(__file__).with_name("lightx2v_minwm_runtime.py")
    lightx2v_dst = repo / "Wan21" / "lightx2v_minwm_runtime.py"
    standalone_src = Path(__file__).with_name("lightx2v_standalone_loader.py")
    standalone_dst = repo / "Wan21" / "lightx2v_standalone_loader.py"
    if not runtime_src.exists():
        raise SystemExit(f"Missing runtime helper: {runtime_src}")
    if not torchao_src.exists():
        raise SystemExit(f"Missing TorchAO helper: {torchao_src}")
    if not lightx2v_src.exists():
        raise SystemExit(f"Missing LightX2V helper: {lightx2v_src}")
    if not standalone_src.exists():
        raise SystemExit(f"Missing LightX2V standalone loader: {standalone_src}")

    shutil.copy2(runtime_src, runtime_dst)
    shutil.copy2(torchao_src, torchao_dst)
    shutil.copy2(lightx2v_src, lightx2v_dst)
    shutil.copy2(standalone_src, standalone_dst)
    print(f"Copied: {runtime_dst}")
    print(f"Copied: {torchao_dst}")
    print(f"Copied: {lightx2v_dst}")
    print(f"Copied: {standalone_dst}")

    patch_wan_inference(wan_inference)
    patch_causal_inference(repo / "Wan21" / "pipeline" / "causal_inference.py")
    patch_torchao(wan_inference)


if __name__ == "__main__":
    main()
