"""Lynx video upscaling package."""

from .compat import patch_torchvision
from .logger import get_logger

# Apply compatibility shims at import time
patch_torchvision()

# Initialize shared logger early
logger = get_logger()

__all__ = [
    "download",
    "models",
    "upscale",
    "encode",
    "processor",
    "gui",
    "cli",
]
