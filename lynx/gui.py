"""Tkinter GUI for the Lynx upscaler."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import subprocess
from .processor import Processor
from .models import ensure_model


class SplashScreen:
    """Simple splash window with status text and progress bar."""

    def __init__(self, root: tk.Tk) -> None:
        self.top = tk.Toplevel(root)
        self.top.title("Lynx Loading")
        self.top.resizable(False, False)
        self.top.geometry("360x120")
        self.top.attributes("-topmost", True)
        self.var_msg = tk.StringVar(value="Starting…")
        tk.Label(self.top, textvariable=self.var_msg).pack(pady=10)
        self.bar = ttk.Progressbar(self.top, maximum=100, length=300)
        self.bar.pack(pady=10)
        self.top.update()

    def update(self, msg: str, value: int) -> None:
        self.var_msg.set(msg)
        self.bar["value"] = value
        self.top.update_idletasks()

    def close(self) -> None:
        self.top.destroy()


def preload(root: tk.Tk) -> None:
    """Run basic preflight checks while showing a splash screen."""

    splash = SplashScreen(root)

    # 1. Check FFmpeg
    splash.update("Checking FFmpeg…", 10)
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except Exception:
        pass

    # 2. Ensure models exist (download if needed)
    weights = Path("weights")
    for idx, model in enumerate(["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"]):
        splash.update(f"Loading {model}…", 30 + idx * 30)
        try:
            ensure_model(weights, model,
                         progress_cb=lambda d, t, base=idx: splash.update(
                             f"Loading {model}…",
                             30 + base * 30 + int((d / (t or 1)) * 30)
                         ))
        except Exception:
            continue

    splash.update("Starting UI…", 100)
    splash.close()


class App:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Lynx Upscaler")
        self.processor: Optional[Processor] = None

        pad = {"padx": 6, "pady": 2}

        col1 = tk.Frame(root)
        col1.pack(fill="both", expand=True)

        frm_in = tk.Frame(col1)
        frm_in.pack(fill="x", **pad)
        self.var_input = tk.StringVar()
        tk.Label(frm_in, text="Input file / URL").pack(anchor="w")
        tk.Entry(frm_in, textvariable=self.var_input, width=60).pack(side="left", fill="x", expand=True)
        tk.Button(frm_in, text="Browse", command=self.browse_input).pack(side="left")

        frm_out = tk.Frame(col1)
        frm_out.pack(fill="x", **pad)
        self.var_output = tk.StringVar()
        tk.Label(frm_out, text="Output file").pack(anchor="w")
        tk.Entry(frm_out, textvariable=self.var_output, width=60).pack(side="left", fill="x", expand=True)
        tk.Button(frm_out, text="Browse", command=self.browse_output).pack(side="left")

        frm_w = tk.Frame(col1)
        frm_w.pack(fill="x", **pad)
        self.var_w = tk.IntVar(value=3840)
        self.var_h = tk.IntVar(value=2160)
        tk.Label(frm_w, text="Target Width").grid(row=0, column=0, sticky="e")
        tk.Entry(frm_w, textvariable=self.var_w, width=6).grid(row=0, column=1, sticky="w")
        tk.Label(frm_w, text="Height").grid(row=0, column=2, sticky="e")
        tk.Entry(frm_w, textvariable=self.var_h, width=6).grid(row=0, column=3, sticky="w")

        frm_paths = tk.Frame(col1)
        frm_paths.pack(fill="x", **pad)
        self.var_weights = tk.StringVar(value=str(Path("weights")))
        self.var_work = tk.StringVar(value=str(Path("work")))
        tk.Button(frm_paths, text="Weights…", command=self.browse_weights).grid(row=0, column=0)
        tk.Entry(frm_paths, textvariable=self.var_weights, width=40).grid(row=0, column=1, sticky="w")
        tk.Button(frm_paths, text="Work…", command=self.browse_work).grid(row=1, column=0)
        tk.Entry(frm_paths, textvariable=self.var_work, width=40).grid(row=1, column=1, sticky="w")

        frm_set = tk.Frame(col1)
        frm_set.pack(fill="x", **pad)
        tk.Label(frm_set, text="Settings").grid(row=0, column=0, sticky="w", pady=(2, 6))

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
        ttk.Combobox(frm_set, textvariable=self.var_codec, values=["hevc_nvenc", "h264_nvenc"], width=12, state="readonly").grid(row=2, column=1, sticky="w")

        tk.Label(frm_set, text="Preset").grid(row=2, column=2, sticky="e")
        ttk.Combobox(frm_set, textvariable=self.var_preset, values=[f"p{i}" for i in range(1, 8)], width=6, state="readonly").grid(row=2, column=3, sticky="w")

        tk.Checkbutton(frm_set, text="Use FP16 (RTX)", variable=self.var_fp16).grid(row=3, column=0, sticky="w", pady=2)
        tk.Checkbutton(frm_set, text="Keep temps", variable=self.var_keep_temps).grid(row=3, column=1, sticky="w")
        tk.Checkbutton(frm_set, text="Prefetch models", variable=self.var_prefetch).grid(row=3, column=2, sticky="w")
        tk.Checkbutton(frm_set, text="Strict model hash", variable=self.var_strict_hash).grid(row=3, column=3, sticky="w")

        frm_prog = tk.Frame(col1)
        frm_prog.pack(fill="x", **pad)
        tk.Label(frm_prog, text="Download progress").pack(anchor="w")
        self.bar_dl = ttk.Progressbar(frm_prog, maximum=100)
        self.bar_dl.pack(fill="x")
        tk.Label(frm_prog, text="Process progress").pack(anchor="w", pady=(8, 0))
        self.bar_proc = ttk.Progressbar(frm_prog, maximum=100)
        self.bar_proc.pack(fill="x")

        self.var_status = tk.StringVar(value="Idle")
        tk.Label(col1, textvariable=self.var_status).pack(anchor="w", **pad)
        self.txt_log = tk.Text(col1, height=10)
        self.txt_log.pack(fill="both", expand=True, **pad)

        frm_btn = tk.Frame(col1)
        frm_btn.pack(fill="x", **pad)
        self.btn_run = tk.Button(frm_btn, text="Start", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_cancel = tk.Button(frm_btn, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

    def log(self, msg: str) -> None:
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.root.update_idletasks()

    def set_progress(self, which: str, done: int, total: int) -> None:
        if total <= 0:
            return
        value = max(0, min(100, int(done * 100 / total)))
        if which == "download":
            self.bar_dl["value"] = value
        else:
            self.bar_proc["value"] = value
        self.root.update_idletasks()

    def set_status(self, msg: str) -> None:
        self.var_status.set(msg)
        self.root.update_idletasks()

    def browse_input(self) -> None:
        path = filedialog.askopenfilename(title="Select input video")
        if path:
            self.var_input.set(path)

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save output as", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.var_output.set(path)

    def browse_weights(self) -> None:
        path = filedialog.askdirectory(title="Select weights folder")
        if path:
            self.var_weights.set(path)

    def browse_work(self) -> None:
        path = filedialog.askdirectory(title="Select work folder")
        if path:
            self.var_work.set(path)

    def collect_cfg(self) -> dict:
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

    def start(self) -> None:
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

    def cancel(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.set_status("Cancelling…")
        self.btn_cancel.config(state="disabled")
        self.btn_run.config(state="normal")


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    preload(root)
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(0, root.attributes, "-topmost", False)
    app = App(root)
    root.minsize(720, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
