from __future__ import annotations

import argparse
import sys

from .options import DEFAULTS
from .logger import get_logger


class CLIHooks:
    """Console-based hooks for :class:`Processor`."""

    def __init__(self) -> None:
        self.logger = get_logger()

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def set_progress(self, which: str, done: int, total: int) -> None:
        if total > 0:
            pct = int(done / total * 100)
            self.logger.info("%s %d%% (%d/%d)", which, pct, done, total)

    def set_status(self, msg: str) -> None:
        self.logger.info(msg)

    def confirm_overwrite(self, path: str) -> bool:
        if not sys.stdin.isatty():
            return True
        resp = input(f"Overwrite existing file {path}? [y/N] ")
        return resp.strip().lower().startswith("y")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed command-line arguments."""
    p = argparse.ArgumentParser(description="Headless Lynx video upscaler")
    p.add_argument("input", help="input video file or URL")
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULTS["output"],
        help="output directory or file path",
    )
    p.add_argument("--width", type=int, default=DEFAULTS["target_width"], help="target width")
    p.add_argument("--height", type=int, default=DEFAULTS["target_height"], help="target height")
    p.add_argument("--tile", type=int, default=DEFAULTS["tile"], help="tile size")
    p.add_argument("--cq", type=int, default=DEFAULTS["cq"], help="NVENC constant quality")
    p.add_argument("--codec", default=DEFAULTS["codec"], help="FFmpeg codec")
    p.add_argument("--preset", default=DEFAULTS["preset"], help="NVENC preset")
    p.add_argument("--weights-dir", default=DEFAULTS["weights_dir"], help="Real-ESRGAN weights directory")
    p.add_argument("--workdir", default=DEFAULTS["workdir"], help="temporary working directory")
    p.add_argument("--fp16", action="store_true", default=DEFAULTS["use_fp16"], help="use fp16 upscaling")
    p.add_argument("--keep-temps", action="store_true", default=DEFAULTS["keep_temps"], help="keep temporary files")
    p.add_argument("--no-prefetch", dest="prefetch_models", action="store_false", default=DEFAULTS["prefetch_models"], help="skip model download")
    p.add_argument("--strict-model-hash", action="store_true", default=DEFAULTS["strict_model_hash"], help="fail on weight checksum mismatch")
    p.add_argument("--quality", choices=["quick", "normal", "better", "best"], default=DEFAULTS["model_quality"], help="quality level")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .processor import Processor
    cfg = {
        "input": args.input,
        "output": args.output,
        "target_width": args.width,
        "target_height": args.height,
        "tile": args.tile,
        "cq": args.cq,
        "codec": args.codec,
        "preset": args.preset,
        "weights_dir": args.weights_dir,
        "workdir": args.workdir,
        "use_fp16": args.fp16,
        "keep_temps": args.keep_temps,
        "prefetch_models": args.prefetch_models,
        "strict_model_hash": args.strict_model_hash,
        "model_quality": args.quality,
    }
    hooks = CLIHooks()
    proc = Processor(hooks)
    try:
        proc.run(cfg)
    except Exception as exc:  # pragma: no cover - runtime errors
        hooks.log(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
