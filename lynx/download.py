"""Download utilities."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional


def is_url(s: str) -> bool:
    """Return True if the string looks like an HTTP/HTTPS URL."""
    return bool(re.match(r"https?://", s, re.I))


def yt_download(
    url: str,
    downloads_dir: Path,
    temp_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download a YouTube video to `downloads_dir` using yt-dlp."""

    downloads_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp") from exc

    def hook(d: dict):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if progress_cb:
                progress_cb(downloaded, total)
        elif d.get("status") in ("finished", "error"):
            if progress_cb:
                progress_cb(1, 1)

    if log_cb:
        log_cb("Downloading YouTube source…")

    # yt-dlp expects flat lists for postprocessor_args; avoid nested lists
    ydl_opts = {
        "paths": {"home": str(downloads_dir)},
        "outtmpl": "%(title).200B [%(id)s].%(ext)s",
        "format": "bv*+ba/b",
        "merge_output_format": "mkv",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cachedir": False,
        "concurrent_fragment_downloads": 8,
        "postprocessor_args": {"Merger": ["-nostdin"]},
        "overwrites": True,
        "progress_hooks": [hook],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = ydl.prepare_filename(info)
        final = Path(base).with_suffix(".mkv")
        if not final.exists():
            alt = Path(base)
            if alt.exists():
                final = alt
        if not final.exists():
            raise RuntimeError("Download succeeded but final file not found.")
        if log_cb:
            log_cb(f"Downloaded: {final.name}")
        return final.resolve()
