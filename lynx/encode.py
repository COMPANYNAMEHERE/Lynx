"""Video encoding helpers."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional


def run_ffmpeg_encode(
    rgb_frame_iter: Iterable,
    out_w: int,
    out_h: int,
    fps: float,
    out_path: Path,
    codec: str,
    preset: str,
    cq: int,
    cancel_event: threading.Event,
    log_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Encode raw RGB frames to `out_path` using FFmpeg NVENC."""

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{fps:.06f}",
        "-i",
        "-",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    if log_cb:
        log_cb("Starting FFmpeg encode…")
    enc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in rgb_frame_iter:
            if cancel_event.is_set():
                if log_cb:
                    log_cb("Cancel requested. Terminating FFmpeg…")
                enc.terminate()
                break
            try:
                enc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                if log_cb:
                    log_cb("FFmpeg pipe closed unexpectedly")
                break
    finally:
        try:
            enc.stdin.close()
        except Exception:
            pass
        stderr = b""
        if enc.stderr:
            try:
                stderr = enc.stderr.read()
            except Exception:
                pass
        enc.wait()

    if cancel_event.is_set():
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        raise RuntimeError("Operation cancelled.")
    if enc.returncode != 0:
        msg = stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError(f"FFmpeg encode failed: {msg}")
