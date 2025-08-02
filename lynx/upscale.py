"""Upscaling helpers."""
from __future__ import annotations

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
from .logger import get_logger

logger = get_logger()


def pick_model_by_quality(quality: str) -> tuple[str, int]:
    """Return ``(model filename, base scale)`` for the selected quality."""
    if quality == "quick":
        return "realesr-general-x4v3.pth", 4
    if quality == "medium":
        return "Swin2SR_ClassicalSR_X4_64.pth", 4
    if quality == "high":
        return "Real_HAT_GAN_SRx4_sharper.pth", 4
    if quality == "super":
        return "net_params_200.pkl", 4
    return "RealESRGAN_x4plus.pth", 4


def build_upsampler(
    model_path: str, base_scale: int, tile: int, fp16: bool, quality: str
) -> RealESRGANer:
    """Construct and return an upsampler instance for the given quality.

    For ``quick`` quality the standard RealESRGAN architecture is used. For
    other qualities we attempt to load the corresponding network modules
    dynamically. If a specialised architecture is unavailable, the RRDBNet
    fallback is used and a warning is logged.
    """

    logger.info("Initialising %s quality upsampler", quality)

    if quality == "quick":
        rrdb = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=base_scale,
        )
        logger.debug("Using RRDBNet architecture")
        return RealESRGANer(
            scale=base_scale,
            model_path=model_path,
            model=rrdb,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=fp16,
        )

    try:
        if quality == "medium":
            from swin2sr.models.network_swin2sr import Swin2SR

            net = Swin2SR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
            )
            logger.debug("Loaded Swin2SR model")
        elif quality == "high":
            from hat.models.hat_arch import HAT

            net = HAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                embed_dim=96,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                window_size_cross=8,
                mlp_ratio=2,
                upsampler="nearest+conv",
            )
            logger.debug("Loaded HAT model")
        elif quality == "super":
            from adcsr_inference import AdcSREnhancer

            net = AdcSREnhancer(model_path)
            logger.debug("Loaded AdcSR model")
            # AdcSR already encapsulates loading, return early
            return net
        else:
            raise ValueError(f"Unknown quality: {quality}")

        weights = torch.load(model_path, map_location="cpu")
        if isinstance(net, torch.nn.Module):
            net.load_state_dict(weights, strict=False)
            net.eval()
        logger.debug("Loaded weights from %s", model_path)
        return net
    except Exception as exc:  # pragma: no cover - optional deps
        logger.warning("Falling back to RealESRGAN for quality %s: %s", quality, exc)
        rrdb = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=base_scale,
        )
        return RealESRGANer(
            scale=base_scale,
            model_path=model_path,
            model=rrdb,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=fp16,
        )
