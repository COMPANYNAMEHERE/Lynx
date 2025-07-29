"""Model management utilities."""
from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .logger import get_logger

logger = get_logger()

MODEL_SPECS = {
    "RealESRGAN_x2plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    },
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    },
}


def _sha256_of_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    strict: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Download ``url`` to ``dest``, reporting progress and supporting cancel."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=dest.name + ".", dir=str(dest.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    logger.info("Fetching %s", dest.name)

    if log_cb:
        log_cb(f"Downloading {dest.name}…")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with contextlib.closing(urllib.request.urlopen(req)) as r, open(tmp_path, "wb") as f:
        total = int(r.headers.get("Content-Length", "0")) or 0
        downloaded = 0
        while True:
            if cancel_event and cancel_event.is_set():
                tmp_path.unlink(missing_ok=True)
                logger.warning("Download of %s cancelled", dest.name)
                raise RuntimeError("Operation cancelled")
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb:
                progress_cb(downloaded, total)
            if cancel_event and cancel_event.is_set():
                tmp_path.unlink(missing_ok=True)
                logger.warning("Download of %s cancelled", dest.name)
                raise RuntimeError("Operation cancelled")

    if expected_sha256:
        got = _sha256_of_file(tmp_path)
        if got.lower() != expected_sha256.lower():
            if strict:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {dest.name}")
            if log_cb:
                log_cb(
                    f"⚠ Warning: checksum mismatch for {dest.name}. Using file anyway."
                )

    tmp_path.replace(dest)
    if log_cb:
        log_cb(f"Saved {dest.name}")
    logger.info("Saved %s", dest.name)


def ensure_model(
    weights_dir: Path,
    model_filename: str,
    strict_hash: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Ensure ``model_filename`` exists in ``weights_dir``.

    The download step can be aborted early via ``cancel_event``.
    """
    spec = MODEL_SPECS.get(model_filename)
    if not spec:
        raise RuntimeError(f"No spec for model {model_filename}")
    if cancel_event and cancel_event.is_set():
        logger.warning("Model download cancelled")
        raise RuntimeError("Operation cancelled")

    dest = weights_dir / model_filename
    if dest.exists() and spec.get("sha256"):
        try:
            if _sha256_of_file(dest).lower() == spec["sha256"].lower():
                if log_cb:
                    log_cb(f"Model OK: {model_filename}")
                if progress_cb:
                    progress_cb(1, 1)
                return dest
            if log_cb:
                log_cb(f"Checksum mismatch on {model_filename}, re-downloading…")
            dest.unlink(missing_ok=True)
        except Exception:
            if log_cb:
                log_cb(f"Could not verify checksum for {model_filename}, re-downloading…")
            dest.unlink(missing_ok=True)

    if not dest.exists():
        _download_with_progress(
            spec["url"],
            dest,
            expected_sha256=spec.get("sha256"),
            strict=strict_hash,
            progress_cb=progress_cb,
            log_cb=log_cb,
            cancel_event=cancel_event,
        )
    return dest
