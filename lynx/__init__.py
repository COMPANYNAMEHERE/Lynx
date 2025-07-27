"""Lynx video upscaling package."""

from .compat import patch_torchvision

# Apply compatibility shims at import time
patch_torchvision()

__all__ = [
    "download",
    "models",
    "upscale",
    "encode",
    "processor",
    "gui",
]
