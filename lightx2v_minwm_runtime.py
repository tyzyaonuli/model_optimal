#!/usr/bin/env python3
"""LightX2V autoencoder adapter for minWM Wan2.1 latents."""

from __future__ import annotations

from pathlib import Path

import torch


class LightX2VWanTinyVAEWrapper(torch.nn.Module):
    """minWM-compatible wrapper around LightX2V's Wan2.1 LightTAE.

    minWM's WanVAEWrapper.decode_to_pixel receives latents shaped
    [B, F, C, H, W] and returns pixels shaped [B, T, 3, H*8, W*8] in [-1, 1].
    Wan21/wan_inference.py later converts that layout to [B,T,H,W,3] before
    calling torchvision.write_video.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        need_scaled: bool = True,
        parallel: bool = True,
        output_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing LightX2V autoencoder checkpoint: {checkpoint}")
        try:
            from lightx2v_standalone_loader import load_wan_vae_tiny_class

            WanVAE_tiny = load_wan_vae_tiny_class("taew2_1")
        except Exception as exc:
            raise RuntimeError(
                "Could not load LightX2V WanVAE_tiny. Set LIGHTX2V_REPO to your local LightX2V checkout, e.g.:\n"
                "  export LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V"
            ) from exc

        self.dtype = dtype
        self.device = torch.device(device)
        self.parallel = parallel
        self.output_device = torch.device(output_device)
        self.model = WanVAE_tiny(
            vae_path=str(checkpoint),
            dtype=dtype,
            device=str(self.device),
            need_scaled=need_scaled,
        ).eval().requires_grad_(False)
        self.model.to(self.device)

    def _decode_sample(self, sample: torch.Tensor, *, parallel: bool) -> torch.Tensor:
        """Decode one minWM latent sample [F,C,H,W] with LightX2V's own wrapper.

        The source-level WanVAE_tiny.decode contract is [C,F,H,W] in and
        channel-first RGB video out. Do not call taehv.decode_video directly
        here; that bypasses LightX2V's own scaling/layout wrapper and caused
        the observed [5,H,W,20] write_video layout bug.
        """
        del parallel
        return self.model.decode(sample.permute(1, 0, 2, 3).contiguous())

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache:
            print("[lightx2v] use_cache=True is ignored by LightTAE decode", flush=True)
        if latent.ndim != 5:
            raise ValueError(f"Expected minWM latent [B,F,C,H,W], got {tuple(latent.shape)}")

        latent = latent.to(device=self.device, dtype=self.dtype)
        outputs = []
        with torch.inference_mode():
            for sample in latent:
                decoded = self._decode_sample(sample, parallel=self.parallel)
                if decoded.ndim == 5 and decoded.shape[0] == 1:
                    decoded = decoded.squeeze(0)
                if decoded.ndim != 4:
                    raise ValueError(f"Expected LightTAE decoded [3,T,H,W] or [1,3,T,H,W], got {tuple(decoded.shape)}")
                if decoded.shape[0] != 3:
                    raise ValueError(
                        "Expected LightTAE decoded [3,T,H,W]. "
                        f"decoded={tuple(decoded.shape)}, latent_sample={tuple(sample.shape)}, "
                        "so this LightX2V checkout is not returning RGB channel-first video from WanVAE_tiny.decode."
                    )
                video = decoded.permute(1, 0, 2, 3).contiguous().clamp_(-1, 1)
                outputs.append(video.to(device=self.output_device, dtype=torch.float32, non_blocking=True))
                del decoded, video
                if self.output_device.type == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.stack(outputs, dim=0)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LightTAE integration is decode-only for minWM inference.")


def build_lightx2v_vae_from_env() -> LightX2VWanTinyVAEWrapper:
    import os

    checkpoint = os.environ.get("MINWM_LIGHTX2V_VAE_PATH", "")
    if not checkpoint:
        raise RuntimeError("MINWM_LIGHTX2V_VAE_PATH must point to lighttaew2_1.pth or .safetensors")
    dtype_name = os.environ.get("MINWM_LIGHTX2V_DTYPE", "bfloat16").lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise RuntimeError(f"Unsupported MINWM_LIGHTX2V_DTYPE={dtype_name!r}")
    return LightX2VWanTinyVAEWrapper(
        checkpoint,
        dtype=dtype_map[dtype_name],
        device=os.environ.get("MINWM_LIGHTX2V_DEVICE", "cuda"),
        output_device=os.environ.get("MINWM_LIGHTX2V_OUTPUT_DEVICE", "cpu"),
        need_scaled=os.environ.get("MINWM_LIGHTX2V_NEED_SCALED", "1").lower() in ("1", "true", "yes"),
        parallel=os.environ.get("MINWM_LIGHTX2V_PARALLEL", "1").lower() in ("1", "true", "yes"),
    )
