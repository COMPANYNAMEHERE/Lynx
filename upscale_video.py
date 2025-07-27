# upscale_gui.py
# GUI pipeline (Windows-friendly) for:
# - Optional YouTube download (best quality) to a local project folder
# - Real-ESRGAN super-resolution on GPU (CUDA / RTX OK)
# - NVENC encode to MP4
# - Auto-download of model weights into ./weights with optional hash checks
#
# Run:
#   python upscale_gui.py

import argparse
import contextlib
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Optional

# --- GUI (stdlib only) ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Core libs ---
import cv2
import numpy as np
import torch
from tqdm import tqdm
# --- Torchvision compatibility shim (must be before importing realesrgan/basicsr) ---
import sys, types
try:
    import torchvision.transforms.functional_tensor as _ft  # old path exists
except Exception:
    import torchvision.transforms.functional as _F
    _mod = types.ModuleType("torchvision.transforms.functional_tensor")
    # expose only what basicsr needs
    _mod.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _mod
# --- end shim ---

# Real-ESRGAN
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# =========================
# Model specs & downloader
# =========================

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
    progress_cb: Optional[Callable[[int, int], None]] = None,  # (downloaded, total)
    log_cb: Optional[Callable[[str], None]] = None
):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=dest.name + ".", dir=str(dest.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    if log_cb: log_cb(f"Downloading {dest.name}…")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with contextlib.closing(urllib.request.urlopen(req)) as r, open(tmp_path, "wb") as f:
        total = int(r.headers.get("Content-Length", "0")) or 0
        downloaded = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb:
                progress_cb(downloaded, total)

    if expected_sha256:
        got = _sha256_of_file(tmp_path)
        if got.lower() != expected_sha256.lower():
            if strict:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {dest.name}")
            else:
                if log_cb: log_cb(f"⚠ Warning: checksum mismatch for {dest.name}. Using file anyway.")

    tmp_path.replace(dest)
    if log_cb: log_cb(f"Saved {dest.name}")

def ensure_model(
    weights_dir: Path,
    model_filename: str,
    strict_hash: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None
) -> Path:
    spec = MODEL_SPECS.get(model_filename)
    if not spec:
        raise RuntimeError(f"No spec for model {model_filename}")

    dest = weights_dir / model_filename
    if dest.exists() and spec.get("sha256"):
        try:
            if _sha256_of_file(dest).lower() == spec["sha256"].lower():
                if log_cb: log_cb(f"Model OK: {model_filename}")
                if progress_cb: progress_cb(1, 1)
                return dest
            else:
                if log_cb: log_cb(f"Checksum mismatch on {model_filename}, re-downloading…")
                dest.unlink(missing_ok=True)
        except Exception:
            if log_cb: log_cb(f"Could not verify checksum for {model_filename}, re-downloading…")
            dest.unlink(missing_ok=True)

    if not dest.exists():
        _download_with_progress(
            spec["url"], dest,
            expected_sha256=spec.get("sha256"),
            strict=strict_hash,
            progress_cb=progress_cb,
            log_cb=log_cb
        )
    return dest


# =========================
# YouTube download
# =========================

def is_url(s: str) -> bool:
    return bool(re.match(r'https?://', s, re.I))

def yt_download(
    url: str,
    downloads_dir: Path,
    temp_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None
) -> Path:
    """
    Download best video+audio locally with yt-dlp, no global cache.
    Output container: MKV to avoid re-encode (keeps quality).
    Returns final file path.
    """
    downloads_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Force temp files to project-local directories (Windows)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise RuntimeError("yt-dlp not installed in this venv. Run: pip install yt-dlp")

    # Progress hook
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if progress_cb:
                progress_cb(downloaded, total)
        elif d.get("status") in ("finished", "error"):
            if progress_cb:
                progress_cb(1, 1)

    if log_cb: log_cb("Downloading YouTube source…")
    ydl_opts = {
        "paths": {"home": str(downloads_dir)},
        "outtmpl": "%(title).200B [%(id)s].%(ext)s",
        "format": "bv*+ba/b",                      # best video + best audio
        "merge_output_format": "mkv",              # safe container, no re-encode
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cachedir": False,
        "concurrent_fragment_downloads": 8,
        "postprocessor_args": [["-nostdin"]],
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
        if log_cb: log_cb(f"Downloaded: {final.name}")
        return final.resolve()


# =========================
# Super-resolution helpers
# =========================

def pick_model(scale_requested: float):
    # Use x2 for ~2x targets (e.g., 1080->4K); x4 when >2.1x needed
    if scale_requested <= 2.1:
        return "RealESRGAN_x2plus.pth", 2
    else:
        return "RealESRGAN_x4plus.pth", 4

def build_upsampler(model_path: str, base_scale: int, tile: int, fp16: bool) -> RealESRGANer:
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=base_scale)
    return RealESRGANer(
        scale=base_scale, model_path=model_path, model=rrdb,
        tile=tile, tile_pad=10, pre_pad=0, half=fp16
    )

def run_ffmpeg_encode(
    rgb_frame_iter,
    out_w: int, out_h: int, fps: float,
    out_path: Path, codec: str, preset: str, cq: int,
    cancel_event: threading.Event,
    log_cb: Optional[Callable[[str], None]] = None
):
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.06f}",
        "-i", "-",
        "-c:v", codec,
        "-preset", preset,
        "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    if log_cb: log_cb("Starting FFmpeg encode…")
    enc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for frame in rgb_frame_iter:
            if cancel_event.is_set():
                if log_cb: log_cb("Cancel requested. Stopping encode…")
                break
            enc.stdin.write(frame.tobytes())
    finally:
        try:
            enc.stdin.close()
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
        raise RuntimeError("FFmpeg encode failed.")


# =========================
# Worker (processing thread)
# =========================

class Processor:
    def __init__(self, ui):
        self.ui = ui
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def _log(self, msg: str):
        self.ui.log(msg)

    def _set_bar(self, which: str, done: int, total: int):
        self.ui.set_progress(which, done, total)

    def run(self, cfg):
        """
        cfg: dict containing all values from UI
        """
        try:
            # Quick checks
            self._log("Checking FFmpeg availability…")
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception:
                raise RuntimeError("FFmpeg not found in PATH. Install it and add ffmpeg/bin to PATH.")

            self._log("Checking GPU runtime…")
            if not torch.cuda.is_available():
                self._log("⚠ CUDA not detected. Will run on CPU (slow).")

            # Resolve input
            workdir = Path(cfg["workdir"])
            downloads_dir = workdir / "downloads"
            temp_dir = workdir / "temp"
            outputs_dir = Path(cfg["output"]).expanduser().resolve().parent
            outputs_dir.mkdir(parents=True, exist_ok=True)

            if is_url(cfg["input"]):
                inp_path = yt_download(
                    cfg["input"],
                    downloads_dir,
                    temp_dir,
                    progress_cb=lambda d, t: self._set_bar("download", d, t or 1),
                    log_cb=self._log
                )
            else:
                inp_path = Path(cfg["input"]).expanduser().resolve()
                if not inp_path.exists():
                    raise RuntimeError(f"Input not found: {inp_path}")

            out_path = Path(cfg["output"]).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Open input
            cap = cv2.VideoCapture(str(inp_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open input video.")

            in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if nframes <= 0:  # streams with unknown frame count
                nframes = None

            self._log(f"Source: {in_w}x{in_h} @ {fps:.3f} fps")

            # Scale math
            sx = cfg["target_width"]  / in_w
            sy = cfg["target_height"] / in_h
            target_scale = min(sx, sy)
            passthrough = target_scale <= 1.02

            out_w = int(math.floor(in_w * (1 if passthrough else target_scale) / 2) * 2)
            out_h = int(math.floor(in_h * (1 if passthrough else target_scale) / 2) * 2)

            # Prefetch models if requested
            weights_dir = Path(cfg["weights_dir"])
            if cfg["prefetch_models"]:
                self._log("Prefetching models…")
                ensure_model(weights_dir, "RealESRGAN_x2plus.pth",
                             strict_hash=cfg["strict_model_hash"],
                             progress_cb=lambda d,t: self._set_bar("download", d, t or 1),
                             log_cb=self._log)
                ensure_model(weights_dir, "RealESRGAN_x4plus.pth",
                             strict_hash=cfg["strict_model_hash"],
                             progress_cb=lambda d,t: self._set_bar("download", d, t or 1),
                             log_cb=self._log)

            # Frame generators
            def frames_iter():
                fidx = 0
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
                run_ffmpeg_encode(
                    frames_iter(),
                    out_w, out_h, fps,
                    out_path, cfg["nvenc_codec"], cfg["preset"], cfg["cq"],
                    cancel_event=self.cancel_event,
                    log_cb=self._log
                )
            else:
                # Pick model and ensure
                model_file, base_scale = pick_model(target_scale)
                model_path = ensure_model(
                    weights_dir, model_file,
                    strict_hash=cfg["strict_model_hash"],
                    progress_cb=lambda d,t: self._set_bar("download", d, t or 1),
                    log_cb=self._log
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                up = build_upsampler(str(model_path), base_scale, cfg["tile"], fp16=(device=="cuda" and cfg["use_fp16"]))
                if device != "cuda":
                    self._log("⚠ Running on CPU; this will be slow.")

                def upscaled_frames():
                    fidx = 0
                    self._log("Upscaling…")
                    pbar_total = nframes or 0
                    # Manual progress via UI bar
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
                        self.ui.set_status(f"Upscaling… {fidx}/{nframes or '?'}")
                        yield sr
                    cap.release()

                run_ffmpeg_encode(
                    upscaled_frames(),
                    out_w, out_h, fps,
                    out_path, cfg["nvenc_codec"], cfg["preset"], cfg["cq"],
                    cancel_event=self.cancel_event,
                    log_cb=self._log
                )

            if not cfg["keep_temps"]:
                try:
                    for child in (Path(cfg["workdir"]) / "temp").glob("*"):
                        if child.is_file():
                            child.unlink()
                except Exception:
                    pass

            self._log(f"Done → {out_path}")
            self.ui.set_status("✅ Completed")
            messagebox.showinfo("Done", f"Output saved:\n{out_path}")

        except Exception as e:
            self.ui.set_status("❌ Error")
            self._log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))


# =========================
# GUI
# =========================

class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Upscaler (Real-ESRGAN + NVENC)")

        self.processor = Processor(self)

        pad = {"padx": 8, "pady": 4}
        col1 = tk.Frame(root)
        col1.pack(fill="both", expand=True)

        # Input row
        frm_in = tk.Frame(col1)
        frm_in.pack(fill="x", **pad)
        tk.Label(frm_in, text="Input (file or YouTube URL):").pack(anchor="w")
        in_row = tk.Frame(frm_in)
        in_row.pack(fill="x")
        self.var_input = tk.StringVar()
        tk.Entry(in_row, textvariable=self.var_input).pack(side="left", fill="x", expand=True)
        tk.Button(in_row, text="Browse…", command=self.browse_input).pack(side="left", padx=6)

        # Output row
        frm_out = tk.Frame(col1)
        frm_out.pack(fill="x", **pad)
        tk.Label(frm_out, text="Output file (.mp4):").pack(anchor="w")
        out_row = tk.Frame(frm_out)
        out_row.pack(fill="x")
        self.var_output = tk.StringVar(value=str((Path("outputs") / "output_4k.mp4").resolve()))
        tk.Entry(out_row, textvariable=self.var_output).pack(side="left", fill="x", expand=True)
        tk.Button(out_row, text="Save As…", command=self.browse_output).pack(side="left", padx=6)

        # Target size
        frm_size = tk.Frame(col1)
        frm_size.pack(fill="x", **pad)
        tk.Label(frm_size, text="Target resolution:").grid(row=0, column=0, sticky="w")
        self.var_w = tk.IntVar(value=3840)
        self.var_h = tk.IntVar(value=2160)
        tk.Label(frm_size, text="Width").grid(row=0, column=1)
        tk.Entry(frm_size, width=8, textvariable=self.var_w).grid(row=0, column=2)
        tk.Label(frm_size, text="Height").grid(row=0, column=3)
        tk.Entry(frm_size, width=8, textvariable=self.var_h).grid(row=0, column=4)

        # Weights & workdir
        frm_dirs = tk.Frame(col1)
        frm_dirs.pack(fill="x", **pad)
        tk.Label(frm_dirs, text="Weights folder:").grid(row=0, column=0, sticky="w")
        self.var_weights = tk.StringVar(value=str(Path("weights").resolve()))
        tk.Entry(frm_dirs, textvariable=self.var_weights).grid(row=0, column=1, sticky="ew")
        tk.Button(frm_dirs, text="…", command=self.browse_weights).grid(row=0, column=2, padx=4)
        tk.Label(frm_dirs, text="Work folder:").grid(row=1, column=0, sticky="w")
        self.var_work = tk.StringVar(value=str(Path("work").resolve()))
        tk.Entry(frm_dirs, textvariable=self.var_work).grid(row=1, column=1, sticky="ew")
        tk.Button(frm_dirs, text="…", command=self.browse_work).grid(row=1, column=2, padx=4)
        frm_dirs.grid_columnconfigure(1, weight=1)

        # Settings
        frm_set = tk.Frame(col1, relief="groove", bd=1)
        frm_set.pack(fill="x", **pad)
        tk.Label(frm_set, text="Settings").grid(row=0, column=0, sticky="w", pady=(2,6))

        self.var_tile = tk.IntVar(value=256)
        self.var_cq = tk.IntVar(value=19)
        self.var_codec = tk.StringVar(value="hevc_nvenc")
        self.var_preset = tk.StringVar(value="p5")
        self.var_fp16 = tk.BooleanVar(value=True)
        self.var_keep_temps = tk.BooleanVar(value=False)
        self.var_prefetch = tk.BooleanVar(value=False)
        self.var_strict_hash = tk.BooleanVar(value=False)

        tk.Label(frm_set, text="Tile").grid(row=1, column=0, sticky="e")
        tk.Entry(frm_set, width=6, textvariable=self.var_tile).grid(row=1, column=1, sticky="w")

        tk.Label(frm_set, text="CQ").grid(row=1, column=2, sticky="e")
        tk.Entry(frm_set, width=6, textvariable=self.var_cq).grid(row=1, column=3, sticky="w")

        tk.Label(frm_set, text="Codec").grid(row=2, column=0, sticky="e")
        ttk.Combobox(frm_set, textvariable=self.var_codec, values=["hevc_nvenc","h264_nvenc"], width=12, state="readonly").grid(row=2, column=1, sticky="w")

        tk.Label(frm_set, text="Preset").grid(row=2, column=2, sticky="e")
        ttk.Combobox(frm_set, textvariable=self.var_preset, values=[f"p{i}" for i in range(1,8)], width=6, state="readonly").grid(row=2, column=3, sticky="w")

        tk.Checkbutton(frm_set, text="Use FP16 (RTX)", variable=self.var_fp16).grid(row=3, column=0, sticky="w", pady=2)
        tk.Checkbutton(frm_set, text="Keep temps", variable=self.var_keep_temps).grid(row=3, column=1, sticky="w")
        tk.Checkbutton(frm_set, text="Prefetch models", variable=self.var_prefetch).grid(row=3, column=2, sticky="w")
        tk.Checkbutton(frm_set, text="Strict model hash", variable=self.var_strict_hash).grid(row=3, column=3, sticky="w")

        # Progress
        frm_prog = tk.Frame(col1)
        frm_prog.pack(fill="x", **pad)
        tk.Label(frm_prog, text="Download progress").pack(anchor="w")
        self.bar_dl = ttk.Progressbar(frm_prog, maximum=100)
        self.bar_dl.pack(fill="x")

        tk.Label(frm_prog, text="Process progress").pack(anchor="w", pady=(8,0))
        self.bar_proc = ttk.Progressbar(frm_prog, maximum=100)
        self.bar_proc.pack(fill="x")

        # Status + log
        self.var_status = tk.StringVar(value="Idle")
        tk.Label(col1, textvariable=self.var_status).pack(anchor="w", **pad)
        self.txt_log = tk.Text(col1, height=10)
        self.txt_log.pack(fill="both", expand=True, **pad)

        # Run/Cancel
        frm_btn = tk.Frame(col1)
        frm_btn.pack(fill="x", **pad)
        self.btn_run = tk.Button(frm_btn, text="Start", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_cancel = tk.Button(frm_btn, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

    # --- UI helpers ---
    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.root.update_idletasks()

    def set_progress(self, which: str, done: int, total: int):
        if total <= 0:
            return
        value = max(0, min(100, int(done * 100 / total)))
        if which == "download":
            self.bar_dl["value"] = value
        else:
            self.bar_proc["value"] = value
        self.root.update_idletasks()

    def set_status(self, msg: str):
        self.var_status.set(msg)
        self.root.update_idletasks()

    # --- Browsers ---
    def browse_input(self):
        path = filedialog.askopenfilename(title="Select input video")
        if path:
            self.var_input.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(title="Save output as", defaultextension=".mp4",
                                            filetypes=[("MP4", "*.mp4")])
        if path:
            self.var_output.set(path)

    def browse_weights(self):
        path = filedialog.askdirectory(title="Select weights folder")
        if path:
            self.var_weights.set(path)

    def browse_work(self):
        path = filedialog.askdirectory(title="Select work folder")
        if path:
            self.var_work.set(path)

    # --- Run / Cancel ---
    def collect_cfg(self):
        inp = self.var_input.get().strip()
        out = self.var_output.get().strip()
        if not inp:
            raise RuntimeError("Please set an input (file or YouTube URL).")
        if not out:
            raise RuntimeError("Please choose an output file.")
        return {
            "input": inp,
            "output": out,
            "target_width": int(self.var_w.get()),
            "target_height": int(self.var_h.get()),
            "weights_dir": self.var_weights.get().strip(),
            "workdir": self.var_work.get().strip(),
            "tile": int(self.var_tile.get()),
            "cq": int(self.var_cq.get()),
            "nvenc_codec": self.var_codec.get(),
            "preset": self.var_preset.get(),
            "use_fp16": bool(self.var_fp16.get()),
            "keep_temps": bool(self.var_keep_temps.get()),
            "prefetch_models": bool(self.var_prefetch.get()),
            "strict_model_hash": bool(self.var_strict_hash.get()),
        }

    def start(self):
        try:
            cfg = self.collect_cfg()
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.set_status("Starting…")
        self.bar_dl["value"] = 0
        self.bar_proc["value"] = 0
        self.txt_log.delete("1.0", "end")

        self.processor = Processor(self)
        t = threading.Thread(target=self.processor.run, args=(cfg,), daemon=True)
        t.start()

    def cancel(self):
        if self.processor:
            self.processor.cancel()
            self.set_status("Cancelling…")
        self.btn_cancel.config(state="disabled")
        self.btn_run.config(state="normal")


def main():
    # Allow running as a normal script (no CLI needed)
    root = tk.Tk()
    app = App(root)
    root.minsize(720, 600)
    root.mainloop()

if __name__ == "__main__":
    main()
