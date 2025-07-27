"""Upscaling helpers."""
from __future__ import annotations

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def pick_model(scale_requested: float) -> tuple[str, int]:
    """Return (model filename, base scale) for the requested upscaling factor."""
    if scale_requested <= 2.1:
        return "RealESRGAN_x2plus.pth", 2
    return "RealESRGAN_x4plus.pth", 4


def build_upsampler(model_path: str, base_scale: int, tile: int, fp16: bool) -> RealESRGANer:
    """Construct and return a RealESRGANer instance."""
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=base_scale)
    return RealESRGANer(
        scale=base_scale, model_path=model_path, model=rrdb,
        tile=tile, tile_pad=10, pre_pad=0, half=fp16
    )
