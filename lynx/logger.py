from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def _default_log_dir() -> Path:
    """Return an OS-appropriate log directory."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Lynx" / "logs"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "lynx" / "logs"


LOG_DIR = _default_log_dir()
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("lynx")


def setup() -> logging.Logger:
    """Configure and return the shared Lynx logger."""
    if not _logger.handlers:
        _logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = RotatingFileHandler(
            LOG_DIR / "lynx.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
    return _logger


def get_logger() -> logging.Logger:
    """Get the shared Lynx logger."""
    return setup()
