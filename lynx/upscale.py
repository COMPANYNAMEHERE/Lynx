"""Upscaling helpers."""
from __future__ import annotations

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def pick_model_by_quality(quality: str) -> tuple[str, int]:
    """Return ``(model filename, base scale)`` for the selected quality."""
    if quality == "quick":
        return "realesr-general-x4v3.pth", 4
    if quality == "normal":
        return "Swin2SR_ClassicalSR_X4_64.pth", 4
    if quality == "better":
        return "Real_HAT_GAN_SRx4_sharper.pth", 4
    if quality == "best":
        return "net_params_200.pkl", 4
    return "RealESRGAN_x4plus.pth", 4


def build_upsampler(
    model_path: str, base_scale: int, tile: int, fp16: bool, quality: str
) -> RealESRGANer:
    """Construct and return an upsampler instance for the given quality."""
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=base_scale)
    return RealESRGANer(
        scale=base_scale, model_path=model_path, model=rrdb,
        tile=tile, tile_pad=10, pre_pad=0, half=fp16
    )
