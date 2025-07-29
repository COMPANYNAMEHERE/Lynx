"""Compatibility utilities for external packages."""
from __future__ import annotations

import sys
from types import ModuleType

from .logger import get_logger

logger = get_logger()


def patch_torchvision() -> None:
    """Ensure ``torchvision.transforms.functional_tensor`` exists.

    Newer versions of torchvision (>=0.17) moved the tensor functions to
    ``torchvision.transforms._functional_tensor``. Older libraries like
    BasicSR still import the old module. This function installs a shim
    so those imports succeed.
    """
    try:  # pragma: no cover - only executed when module missing
        import torchvision.transforms.functional_tensor  # type: ignore
        return
    except Exception:
        pass

    try:
        import torchvision.transforms._functional_tensor as _ft
    except Exception as exc:
        logger.warning("torchvision shim skipped: %s", exc)
        sys.modules.pop("torchvision", None)
        return

    shim = ModuleType("torchvision.transforms.functional_tensor")
    for name in dir(_ft):
        if not name.startswith("_"):
            setattr(shim, name, getattr(_ft, name))
    sys.modules["torchvision.transforms.functional_tensor"] = shim
