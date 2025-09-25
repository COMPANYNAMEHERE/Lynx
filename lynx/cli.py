from __future__ import annotations

import argparse
import sys

from .options import DEFAULTS
from .preloader import ConsolePreloadHooks, PreloadConfig, VideoPreloader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the preloader CLI."""

    p = argparse.ArgumentParser(
        description="Download a YouTube video ahead of time and upscale it locally.",
    )
    p.add_argument(
        "url",
        help="YouTube video URL or path to a pre-downloaded video",
    )
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULTS["output"],
        help="Destination file for the upscaled video",
    )
    p.add_argument(
        "--width",
        type=int,
        default=DEFAULTS["target_width"],
        help="Target width for the upscaled output",
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULTS["target_height"],
        help="Target height for the upscaled output",
    )
    p.add_argument(
        "--tile",
        type=int,
        default=DEFAULTS["tile"],
        help="Tile size for Real-ESRGAN inference",
    )
    p.add_argument(
        "--cq",
        type=int,
        default=DEFAULTS["cq"],
        help="NVENC constant quality setting",
    )
    p.add_argument(
        "--codec",
        default=DEFAULTS["codec"],
        help="FFmpeg video codec to use",
    )
    p.add_argument(
        "--preset",
        default=DEFAULTS["preset"],
        help="NVENC preset to use",
    )
    p.add_argument(
        "--weights-dir",
        default=DEFAULTS["weights_dir"],
        help="Directory where Real-ESRGAN model weights are stored",
    )
    p.add_argument(
        "--workdir",
        default=DEFAULTS["workdir"],
        help="Working directory for downloads and temporary data",
    )
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["use_fp16"],
        help="Toggle fp16 inference when CUDA is available",
    )
    p.add_argument(
        "--keep-temps",
        action="store_true",
        default=DEFAULTS["keep_temps"],
        help="Keep intermediate temporary files",
    )
    p.add_argument(
        "--prefetch-models",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["prefetch_models"],
        help="Pre-download Real-ESRGAN weights before processing",
    )
    p.add_argument(
        "--strict-model-hash",
        action="store_true",
        default=DEFAULTS["strict_model_hash"],
        help="Abort when a weight checksum mismatch is detected",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hooks = ConsolePreloadHooks()
    config = PreloadConfig.from_args(args)
    preloader = VideoPreloader(config, hooks=hooks)

    try:
        result = preloader.prepare()
    except Exception as exc:  # pragma: no cover - runtime errors
        hooks.log(f"Error: {exc}")
        sys.exit(1)

    w, h = result.output_resolution
    hooks.log(
        f"Finished preloading {result.output_path} ({w}x{h}) in {result.elapsed_seconds:.2f} s",
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
