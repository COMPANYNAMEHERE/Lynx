"""Processing pipeline implementation."""
from __future__ import annotations

import math
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover - allow starting GUI without cv2
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = e
else:
    CV2_IMPORT_ERROR = None
import torch

from .download import is_url, yt_download
from .encode import run_ffmpeg_encode
from .models import ensure_model
from .upscale import build_upsampler, pick_model_by_quality
from .logger import get_logger

logger = get_logger()


class UIHooks:
    """Interface expected by ``Processor`` for UI updates."""

    def log(self, msg: str) -> None: ...
    def set_progress(self, which: str, done: int, total: int) -> None: ...
    def set_status(self, msg: str) -> None: ...
    def confirm_overwrite(self, path: str) -> bool: ...


class Processor:
    """Run the upscaling pipeline in a background thread."""

    def __init__(self, ui: UIHooks) -> None:
        self.ui = ui
        self.cancel_event = threading.Event()
        self.output_path: Optional[Path] = None

    def cancel(self) -> None:
        self.cancel_event.set()

    # Internal helpers
    def _log(self, msg: str, *args: object) -> None:
        if args:
            msg = msg % args
        logger.info(msg)
        self.ui.log(msg)

    def _set_bar(self, which: str, done: int, total: int) -> None:
        self.ui.set_progress(which, done, total)

    def run(self, cfg: dict) -> None:
        """Entry point for the processing thread."""
        logger.info("Processing thread started")
        if cv2 is None:
            raise RuntimeError(
                f"OpenCV is unavailable: {CV2_IMPORT_ERROR}. Please install opencv-python"
            )
        try:
            self._log("Checking FFmpeg availability…")
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            raise RuntimeError("FFmpeg not found in PATH. Install it and add ffmpeg/bin to PATH.")

        self._log("Checking GPU runtime…")
        if torch.cuda.is_available():
            self._log(f"CUDA available (PyTorch {torch.version.cuda})")
        else:
            from .env import detect_gpu_info

            info = detect_gpu_info()
            if info:
                self._log(f"⚠ GPU detected ({info}) but PyTorch lacks CUDA support. Running on CPU.")
            else:
                self._log("⚠ No GPU detected. Running on CPU.")

        workdir = Path(cfg["workdir"])
        downloads_dir = workdir / "downloads"
        temp_dir = workdir / "temp"

        out_param = Path(cfg["output"]).expanduser()
        if out_param.suffix:
            output_dir = out_param.parent
            out_file: Optional[Path] = out_param
        else:
            output_dir = out_param
            out_file = None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Output directory: {output_dir}")

        def _sanitize(name: str) -> str:
            return re.sub(r"[^\w\-\. ]+", "_", name).strip()

        if is_url(cfg["input"]):
            inp_path, title = yt_download(
                cfg["input"],
                downloads_dir,
                temp_dir,
                progress_cb=lambda d, t: self._set_bar("download", d, t or 1),
                log_cb=self._log,
            )
            self._log(f"Downloaded source to {inp_path}")
            if out_file is None:
                out_file = output_dir / f"{_sanitize(title)}.mp4"
        else:
            inp_path = Path(cfg["input"]).expanduser().resolve()
            if not inp_path.exists():
                raise RuntimeError(f"Input not found: {inp_path}")
            self._log(f"Using local input {inp_path}")
            if out_file is None:
                out_file = output_dir / f"{_sanitize(inp_path.stem)}.mp4"

        assert out_file is not None
        if out_file.exists():
            if not self.ui.confirm_overwrite(str(out_file)):
                self._log("Output file exists and overwrite not confirmed")
                return

        self._log(f"Output will be written to {out_file}")

        self.output_path = out_file.resolve()

        cap = cv2.VideoCapture(str(inp_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video.")
        self._log("Video opened successfully")

        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes <= 0:
            nframes = None

        self._log(f"Source: {in_w}x{in_h} @ {fps:.3f} fps")

        sx = cfg["target_width"] / in_w
        sy = cfg["target_height"] / in_h
        target_scale = min(sx, sy)
        passthrough = target_scale <= 1.02

        out_w = int(math.floor(in_w * (1 if passthrough else target_scale) / 2) * 2)
        out_h = int(math.floor(in_h * (1 if passthrough else target_scale) / 2) * 2)

        weights_dir = Path(cfg["weights_dir"])
        if cfg.get("prefetch_models"):
            self._log("Prefetching models…")
            for fname in [
                "realesr-general-x4v3.pth",
                "Swin2SR_ClassicalSR_X4_64.pth",
                "Real_HAT_GAN_SRx4_sharper.pth",
                "net_params_200.pkl",
            ]:
                self._log(f"Ensuring model {fname}")
                ensure_model(
                    weights_dir,
                    fname,
                    strict_hash=cfg.get("strict_model_hash", False),
                    progress_cb=lambda d, t: self._set_bar("download", d, t or 1),
                    log_cb=self._log,
                )

        def frames_iter():
            fidx = 0
            self._log("Decoding frames…")
            while True:
                if self.cancel_event.is_set():
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                fidx += 1
                if nframes:
                    self._set_bar("process", fidx, nframes)
                self.ui.set_status(f"Reading… {fidx}/{nframes or '?'}")
                yield rgb
            cap.release()

        if passthrough:
            self._log("Target ≈ source size → pass-through encode (no upscaling).")
            self._log("Encoding using %s to %s", cfg["codec"], self.output_path)
            run_ffmpeg_encode(
                frames_iter(),
                out_w,
                out_h,
                fps,
                self.output_path,
                cfg["codec"],
                cfg["preset"],
                cfg["cq"],
                cancel_event=self.cancel_event,
                log_cb=self._log,
            )
        else:
            model_file, base_scale = pick_model_by_quality(
                cfg.get("model_quality", "normal")
            )
            model_path = ensure_model(
                weights_dir,
                model_file,
                strict_hash=cfg.get("strict_model_hash", False),
                progress_cb=lambda d, t: self._set_bar("download", d, t or 1),
                log_cb=self._log,
            )
            self._log(f"Using model {model_file}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            up = build_upsampler(
                str(model_path),
                base_scale,
                cfg["tile"],
                fp16=(device == "cuda" and cfg.get("use_fp16", False)),
                quality=cfg.get("model_quality", "normal"),
            )
            if device != "cuda":
                self._log("⚠ Running on CPU; this will be slow.")
            self._log("Encoding using %s to %s", cfg["codec"], self.output_path)

            def upscaled_frames():
                fidx = 0
                self._log("Upscaling…")
                while True:
                    if self.cancel_event.is_set():
                        break
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    sr, _ = up.enhance(rgb, outscale=target_scale)
                    if sr.shape[1] != out_w or sr.shape[0] != out_h:
                        sr = cv2.resize(sr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
                    fidx += 1
                    if nframes:
                        self._set_bar("process", fidx, nframes)
                    self.ui.set_status(f"SR… {fidx}/{nframes or '?'}")
                    yield sr
                cap.release()

            run_ffmpeg_encode(
                upscaled_frames(),
                out_w,
                out_h,
                fps,
                self.output_path,
                cfg["codec"],
                cfg["preset"],
                cfg["cq"],
                cancel_event=self.cancel_event,
                log_cb=self._log,
            )

        self._log("Encoding finished")

        logger.info("Processing finished")
