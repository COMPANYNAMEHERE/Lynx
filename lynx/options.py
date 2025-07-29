"""Persistent options handling for Lynx GUI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .logger import get_logger

logger = get_logger()

def _default_options_dir() -> Path:
    """Return an OS-appropriate directory for persistent settings."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Lynx"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lynx"


OPTIONS_DIR = _default_options_dir()
OPTIONS_FILE = OPTIONS_DIR / "settings.json"

DEFAULTS: Dict[str, Any] = {
    "output": str(Path("outputs") / "output.mp4"),
    "weights_dir": str(Path("weights")),
    "workdir": str(Path("work")),
    "target_width": 3840,
    "target_height": 2160,
    "tile": 256,
    "cq": 19,
    "codec": "hevc_nvenc",
    "preset": "p5",
    "use_fp16": True,
    "keep_temps": False,
    "prefetch_models": True,
    "strict_model_hash": False,
}


def _validate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate option values, falling back to defaults for invalid ones."""
    clean: Dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if key not in data:
            continue
        val = data[key]
        if isinstance(default, bool):
            if isinstance(val, bool):
                clean[key] = val
        elif isinstance(default, int):
            if isinstance(val, int) and val > 0:
                clean[key] = val
        else:
            if isinstance(val, str) and val:
                clean[key] = val
    return clean


def load_options() -> Dict[str, Any]:
    """Load saved options, falling back to defaults."""
    if OPTIONS_FILE.exists():
        try:
            data = json.loads(OPTIONS_FILE.read_text())
            opts = DEFAULTS.copy()
            opts.update(_validate(data))
            logger.debug("Loaded options from %s", OPTIONS_FILE)
            return opts
        except Exception:
            logger.exception("Failed to load options; using defaults")
    return DEFAULTS.copy()


def save_options(opts: Dict[str, Any]) -> None:
    """Persist options to disk."""
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OPTIONS_FILE.write_text(json.dumps(opts, indent=2))
    logger.debug("Saved options to %s", OPTIONS_FILE)
