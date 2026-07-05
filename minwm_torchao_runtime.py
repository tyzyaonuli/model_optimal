#!/usr/bin/env python3
"""Optional TorchAO quantization helpers for minWM 4090 experiments."""

from __future__ import annotations

from typing import Callable

import torch


def _count_modules(module: torch.nn.Module, cls: type[torch.nn.Module]) -> int:
    return sum(1 for item in module.modules() if isinstance(item, cls))


def _count_quantizable_modules(module: torch.nn.Module) -> dict[str, int]:
    return {
        "linear": _count_modules(module, torch.nn.Linear),
        "conv2d": _count_modules(module, torch.nn.Conv2d),
        "conv3d": _count_modules(module, torch.nn.Conv3d),
    }


def _load_torchao_quantizer(mode: str) -> tuple[Callable, Callable]:
    try:
        import torchao.quantization as aoq
    except Exception as exc:
        raise RuntimeError(
            "TorchAO is not installed or cannot be imported. Install it with "
            "`pip install torchao` in the active venv."
        ) from exc

    quantize = getattr(aoq, "quantize_", None)
    if quantize is None:
        raise RuntimeError("torchao.quantization.quantize_ is not available in this torchao version.")

    config_names = {
        "int8wo": ("int8_weight_only",),
        "fp8wo": ("float8_weight_only",),
        "fp8dq": ("float8_dynamic_activation_float8_weight",),
    }.get(mode)
    if config_names is None:
        raise ValueError(f"Unknown TorchAO quantization mode: {mode!r}")

    for name in config_names:
        config_factory = getattr(aoq, name, None)
        if config_factory is not None:
            return quantize, config_factory

    available = ", ".join(sorted(name for name in dir(aoq) if "weight" in name or "float8" in name))
    raise RuntimeError(
        f"TorchAO mode {mode!r} is not supported by this installed torchao. "
        f"Looked for {config_names}; available candidates: {available}"
    )


def apply_torchao_quantization(
    module: torch.nn.Module,
    *,
    mode: str,
    strict: bool = False,
    module_name: str = "module",
) -> torch.nn.Module:
    """Apply TorchAO in-place quantization to Linear-heavy modules.

    Modes:
    - int8wo: int8 weight-only, usually safest for memory experiments.
    - fp8wo: fp8 weight-only if the installed TorchAO/PyTorch stack supports it.
    - fp8dq: fp8 dynamic activation + fp8 weight if supported.
    """
    if mode in ("", "0", "none", "false", "False"):
        return module

    module_counts = _count_quantizable_modules(module)
    print(f"[torchao] target={module_name} mode={mode} modules={module_counts}", flush=True)
    try:
        quantize, config_factory = _load_torchao_quantizer(mode)
        config = config_factory()
        quantize(module, config)
        print(f"[torchao] quantized target={module_name} mode={mode}", flush=True)
    except Exception as exc:
        message = f"[torchao] failed target={module_name} mode={mode}: {type(exc).__name__}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        print(message, flush=True)
        print("[torchao] continuing without TorchAO quantization; set MINWM_TORCHAO_STRICT=1 to fail fast", flush=True)
    return module
