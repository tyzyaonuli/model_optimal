#!/usr/bin/env python3
"""Load LightX2V Wan LightTAE files without importing lightx2v.__init__.

LightX2V's package __init__ imports the full pipeline. In mixed minWM
environments that can fail inside optional diffusers/peft/transformers paths
even when the standalone LightTAE files are usable. This loader imports only
`tae.py` and `wan/vae_tiny.py` from a local LightX2V checkout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots = []
    env_root = os.environ.get("LIGHTX2V_REPO") or os.environ.get("LIGHTX2V_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.extend(
        [
            Path("/root/autodl-tmp/workspace/LightX2V"),
            Path("/workspace/LightX2V"),
            Path.cwd() / "LightX2V",
            Path.cwd().parent / "LightX2V",
        ]
    )
    for item in sys.path:
        if item:
            roots.append(Path(item))
    return roots


def find_lightx2v_package_root() -> Path:
    for root in _candidate_roots():
        candidates = [root / "lightx2v", root]
        for candidate in candidates:
            if (candidate / "models" / "video_encoders" / "hf" / "tae.py").exists():
                if (candidate / "models" / "video_encoders" / "hf" / "wan" / "vae_tiny.py").exists():
                    return candidate
    raise FileNotFoundError(
        "Could not find a local LightX2V checkout. Set LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V"
    )


def _ensure_namespace(name: str, path: Path | None = None) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_wan_vae_tiny_class(model_type: str = "taew2_1"):
    package_root = find_lightx2v_package_root()
    _ensure_namespace("lightx2v", package_root)
    _ensure_namespace("lightx2v.models", package_root / "models")
    _ensure_namespace("lightx2v.models.video_encoders", package_root / "models" / "video_encoders")
    _ensure_namespace("lightx2v.models.video_encoders.hf", package_root / "models" / "video_encoders" / "hf")
    _ensure_namespace("lightx2v.models.video_encoders.hf.wan", package_root / "models" / "video_encoders" / "hf" / "wan")

    _load_module(
        "lightx2v.models.video_encoders.hf.tae",
        package_root / "models" / "video_encoders" / "hf" / "tae.py",
    )
    vae_tiny = _load_module(
        "lightx2v.models.video_encoders.hf.wan.vae_tiny",
        package_root / "models" / "video_encoders" / "hf" / "wan" / "vae_tiny.py",
    )
    if model_type == "taew2_1":
        return getattr(vae_tiny, "WanVAE_tiny")
    if model_type == "taew2_2":
        return getattr(vae_tiny, "Wan2_2_VAE_tiny")
    raise ValueError(f"Standalone loader only supports taew2_1/taew2_2, got {model_type!r}")
