"""Persistent options handling for Lynx GUI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

OPTIONS_DIR = Path("options")
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


def load_options() -> Dict[str, Any]:
    """Load saved options, falling back to defaults."""
    if OPTIONS_FILE.exists():
        try:
            data = json.loads(OPTIONS_FILE.read_text())
            opts = DEFAULTS.copy()
            opts.update({k: data.get(k, v) for k, v in DEFAULTS.items()})
            return opts
        except Exception:
            pass
    return DEFAULTS.copy()


def save_options(opts: Dict[str, Any]) -> None:
    """Persist options to disk."""
    OPTIONS_DIR.mkdir(exist_ok=True)
    OPTIONS_FILE.write_text(json.dumps(opts, indent=2))
