"""High-level helpers for the YouTube preloader workflow."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .logger import get_logger
from .options import DEFAULTS

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from .processor import ProcessingResult
else:  # pragma: no cover - fallback when heavy deps are unavailable
    ProcessingResult = Any  # type: ignore[misc]


@dataclass
class PreloadConfig:
    """Configuration describing a preload job."""

    url: str
    output_path: Path = Path(DEFAULTS["output"])
    target_width: int = DEFAULTS["target_width"]
    target_height: int = DEFAULTS["target_height"]
    workdir: Path = Path(DEFAULTS["workdir"])
    weights_dir: Path = Path(DEFAULTS["weights_dir"])
    tile: int = DEFAULTS["tile"]
    cq: int = DEFAULTS["cq"]
    codec: str = DEFAULTS["codec"]
    preset: str = DEFAULTS["preset"]
    use_fp16: bool = DEFAULTS["use_fp16"]
    keep_temps: bool = DEFAULTS["keep_temps"]
    prefetch_models: bool = DEFAULTS["prefetch_models"]
    strict_model_hash: bool = DEFAULTS["strict_model_hash"]

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("A YouTube URL or local path must be provided")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("Target dimensions must be positive integers")
        if self.tile <= 0:
            raise ValueError("Tile size must be a positive integer")
        if self.cq < 0:
            raise ValueError("Constant quality cannot be negative")
        if not self.codec:
            raise ValueError("Codec must be specified")
        if not self.preset:
            raise ValueError("Preset must be specified")

        self.output_path = Path(self.output_path).expanduser().resolve()
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.weights_dir = Path(self.weights_dir).expanduser().resolve()

    @classmethod
    def from_args(cls, args: Namespace) -> "PreloadConfig":
        """Create a configuration object from CLI arguments."""

        return cls(
            url=args.url,
            output_path=Path(args.output),
            target_width=args.width,
            target_height=args.height,
            workdir=Path(args.workdir),
            weights_dir=Path(args.weights_dir),
            tile=args.tile,
            cq=args.cq,
            codec=args.codec,
            preset=args.preset,
            use_fp16=args.fp16,
            keep_temps=args.keep_temps,
            prefetch_models=args.prefetch_models,
            strict_model_hash=args.strict_model_hash,
        )

    def to_processor_config(self) -> dict[str, object]:
        """Convert the config into the dictionary expected by ``Processor``."""

        return {
            "input": self.url,
            "output": str(self.output_path),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "tile": int(self.tile),
            "cq": int(self.cq),
            "codec": self.codec,
            "preset": self.preset,
            "weights_dir": str(self.weights_dir),
            "workdir": str(self.workdir),
            "use_fp16": bool(self.use_fp16),
            "keep_temps": bool(self.keep_temps),
            "prefetch_models": bool(self.prefetch_models),
            "strict_model_hash": bool(self.strict_model_hash),
        }


PreloadResult = ProcessingResult


class ConsolePreloadHooks:
    """Default hooks that log progress through :mod:`logging`."""

    def __init__(self) -> None:
        self.logger = get_logger()

    def log(self, msg: str) -> None:  # pragma: no cover - pass-through to logger
        self.logger.info(msg)

    def set_progress(self, which: str, done: int, total: int) -> None:
        if total > 0:
            pct = int(done / total * 100)
            self.logger.info("%s %d%% (%d/%d)", which, pct, done, total)
        else:
            self.logger.info("%s %d", which, done)

    def set_status(self, msg: str) -> None:
        self.logger.info(msg)


class VideoPreloader:
    """Coordinate downloading, upscaling and encoding."""

    def __init__(self, config: PreloadConfig, hooks: Optional[Any] = None) -> None:
        self.config = config
        self.hooks = hooks or ConsolePreloadHooks()
        from .processor import Processor

        self._processor = Processor(self.hooks)  # heavy import deferred until needed

    def prepare(self) -> PreloadResult:
        """Execute the preloading pipeline and return a summary."""
        summary = self._processor.run(self.config.to_processor_config())
        if not summary.output_path.exists():
            raise RuntimeError(f"Upscaled video missing: {summary.output_path}")
        return summary
