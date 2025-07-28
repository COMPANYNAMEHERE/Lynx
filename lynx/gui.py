"""Tkinter GUI for the Lynx upscaler."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover - import failure check
    raise RuntimeError(
        "customtkinter is required for the GUI. Install it with 'pip install customtkinter'."
    ) from e
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
from .logger import get_logger

logger = get_logger()


class Tooltip:
    """Simple tooltip for widgets."""

    def __init__(self, widget: ctk.CTkBaseClass, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: Optional[ctk.CTkToplevel] = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event: object | None = None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tip = ctk.CTkToplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        ctk.CTkLabel(self.tip, text=self.text, fg_color="#ffffe0", text_color="black", corner_radius=2).pack(padx=4, pady=2)

    def hide(self, _event: object | None = None) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None

import subprocess
from .processor import Processor
from .models import ensure_model
from .options import load_options, save_options, DEFAULTS


class SplashScreen:
    """Simple splash window with status text, progress bar and cancel."""

    def __init__(self, root: ctk.CTk, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.top = ctk.CTkToplevel(root)
        self.top.title("Lynx Loading")
        self.top.resizable(False, False)
        self.top.geometry("360x120")
        self.top.attributes("-topmost", True)
        self.var_msg = ctk.StringVar(value="Starting…")
        ctk.CTkLabel(self.top, textvariable=self.var_msg).pack(pady=10)
        self.bar = ctk.CTkProgressBar(self.top, width=300)
        self.bar.pack(pady=10)
        self.bar.set(0)
        ctk.CTkButton(self.top, text="Cancel", command=self.cancel).pack(pady=(0, 8))
        self.top.update()

    def update(self, msg: str, value: int) -> None:
        self.var_msg.set(msg)
        self.bar.set(value / 100)
        self.top.update_idletasks()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.var_msg.set("Cancelling…")
        self.top.update()

    def close(self) -> None:
        self.top.destroy()


def preload(root: ctk.CTk) -> bool:
    """Show a temporary splash screen while verifying runtime.

    Returns ``True`` if startup completed, ``False`` if cancelled.
    """

    logger.info("Performing startup checks")
    cancel_event = threading.Event()
    splash = SplashScreen(root, cancel_event)

    # 1. Check FFmpeg
    splash.update("Checking FFmpeg…", 10)
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except Exception:
        pass
    if cancel_event.is_set():
        splash.close()
        return False

    # 2. Ensure models exist (download if needed)
    weights = Path("weights")
    for idx, model in enumerate(["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"]):
        if cancel_event.is_set():
            splash.close()
            return False

        splash.update(f"Loading {model}…", 30 + idx * 30)

        def prog(d: int, t: int, base: int = idx) -> None:
            splash.update(
                f"Loading {model}…",
                30 + base * 30 + int((d / (t or 1)) * 30),
            )

        try:
            ensure_model(
                weights,
                model,
                progress_cb=prog,
                cancel_event=cancel_event,
            )
        except Exception:
            if cancel_event.is_set():
                splash.close()
                return False
            continue

    if cancel_event.is_set():
        splash.close()
        return False

    splash.update("Starting UI…", 100)
    logger.info("Startup checks complete")
    splash.close()
    return True


class App:
    """Main application window."""

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        root.title("Lynx Upscaler")
        self.processor: Optional[Processor] = None
        self.opts = load_options()
        logger.debug("Options loaded: %s", self.opts)

        menubar = tk.Menu(root)
        opt_menu = tk.Menu(menubar, tearoff=0)
        opt_menu.add_command(label="Options", command=self.open_options)
        menubar.add_cascade(label="Settings", menu=opt_menu)
        root.config(menu=menubar)

        pad = {"padx": 6, "pady": 2}

        col1 = ctk.CTkFrame(root)
        col1.pack(fill="both", expand=True)

        frm_in = ctk.CTkFrame(col1)
        frm_in.pack(fill="x", **pad)
        self.var_input = ctk.StringVar()
        lbl_in = ctk.CTkLabel(frm_in, text="Video file or YouTube link:")
        lbl_in.pack(anchor="w")
        ent_in = ctk.CTkEntry(frm_in, textvariable=self.var_input, width=400)
        ent_in.pack(side="left", fill="x", expand=True)
        btn_in = ctk.CTkButton(frm_in, text="Browse", command=self.browse_input)
        btn_in.pack(side="left")
        Tooltip(ent_in, "Choose a local video or paste a YouTube URL")

        frm_out = ctk.CTkFrame(col1)
        frm_out.pack(fill="x", **pad)
        self.var_output = ctk.StringVar(value=self.opts.get("output", DEFAULTS["output"]))
        lbl_out = ctk.CTkLabel(frm_out, text="Save output to:")
        lbl_out.pack(anchor="w")
        ent_out = ctk.CTkEntry(frm_out, textvariable=self.var_output, width=400)
        ent_out.pack(side="left", fill="x", expand=True)
        btn_out = ctk.CTkButton(frm_out, text="Browse", command=self.browse_output)
        btn_out.pack(side="left")
        Tooltip(ent_out, "Destination video file")

        frm_w = ctk.CTkFrame(col1)
        frm_w.pack(fill="x", **pad)
        self.var_w = ctk.IntVar(value=int(self.opts.get("target_width", DEFAULTS["target_width"])))
        self.var_h = ctk.IntVar(value=int(self.opts.get("target_height", DEFAULTS["target_height"])))
        ctk.CTkLabel(frm_w, text="Output width").grid(row=0, column=0, sticky="e")
        ent_w = ctk.CTkEntry(frm_w, textvariable=self.var_w, width=80)
        ent_w.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(frm_w, text="Height").grid(row=0, column=2, sticky="e")
        ent_h = ctk.CTkEntry(frm_w, textvariable=self.var_h, width=80)
        ent_h.grid(row=0, column=3, sticky="w")
        Tooltip(ent_w, "Desired output width in pixels")
        Tooltip(ent_h, "Desired output height in pixels")

        frm_paths = ctk.CTkFrame(col1)
        frm_paths.pack(fill="x", **pad)
        self.var_weights = ctk.StringVar(value=self.opts.get("weights_dir", DEFAULTS["weights_dir"]))
        self.var_work = ctk.StringVar(value=self.opts.get("workdir", DEFAULTS["workdir"]))
        ctk.CTkLabel(frm_paths, text="Model folder").grid(row=0, column=0, sticky="e")
        self.cmb_weights = ctk.CTkComboBox(frm_paths, variable=self.var_weights, values=["Browse…"], width=200, state="readonly")
        self.cmb_weights.grid(row=0, column=1, sticky="w")
        self.cmb_weights.bind("<<ComboboxSelected>>", self.browse_weights)
        Tooltip(self.cmb_weights, "Where model weights are stored")
        ctk.CTkLabel(frm_paths, text="Work folder").grid(row=1, column=0, sticky="e")
        self.cmb_work = ctk.CTkComboBox(frm_paths, variable=self.var_work, values=["Browse…"], width=200, state="readonly")
        self.cmb_work.grid(row=1, column=1, sticky="w")
        self.cmb_work.bind("<<ComboboxSelected>>", self.browse_work)
        Tooltip(self.cmb_work, "Temporary working directory")

        frm_set = ctk.CTkFrame(col1)
        frm_set.pack(fill="x", **pad)
        ctk.CTkLabel(frm_set, text="Settings").grid(row=0, column=0, sticky="w", pady=(2, 6))

        self.var_tile = ctk.IntVar(value=int(self.opts.get("tile", DEFAULTS["tile"])))
        self.var_cq = ctk.IntVar(value=int(self.opts.get("cq", DEFAULTS["cq"])))
        self.var_codec = ctk.StringVar(value=self.opts.get("codec", DEFAULTS["codec"]))
        self.var_preset = ctk.StringVar(value=self.opts.get("preset", DEFAULTS["preset"]))
        self.var_fp16 = ctk.BooleanVar(value=bool(self.opts.get("use_fp16", DEFAULTS["use_fp16"])))
        self.var_keep_temps = ctk.BooleanVar(value=bool(self.opts.get("keep_temps", DEFAULTS["keep_temps"])))
        self.var_prefetch = ctk.BooleanVar(value=bool(self.opts.get("prefetch_models", DEFAULTS["prefetch_models"])))
        self.var_strict_hash = ctk.BooleanVar(value=bool(self.opts.get("strict_model_hash", DEFAULTS["strict_model_hash"])))

        ctk.CTkLabel(frm_set, text="Tile").grid(row=1, column=0, sticky="e")
        ctk.CTkEntry(frm_set, width=60, textvariable=self.var_tile).grid(row=1, column=1, sticky="w")

        ctk.CTkLabel(frm_set, text="CQ").grid(row=1, column=2, sticky="e")
        ctk.CTkEntry(frm_set, width=60, textvariable=self.var_cq).grid(row=1, column=3, sticky="w")

        ctk.CTkLabel(frm_set, text="Codec").grid(row=2, column=0, sticky="e")
        ctk.CTkComboBox(frm_set, variable=self.var_codec, values=["hevc_nvenc", "h264_nvenc"], width=120, state="readonly").grid(row=2, column=1, sticky="w")

        ctk.CTkLabel(frm_set, text="Preset").grid(row=2, column=2, sticky="e")
        ctk.CTkComboBox(frm_set, variable=self.var_preset, values=[f"p{i}" for i in range(1, 8)], width=60, state="readonly").grid(row=2, column=3, sticky="w")

        ctk.CTkCheckBox(frm_set, text="Use FP16 (RTX only)", variable=self.var_fp16).grid(row=3, column=0, sticky="w", pady=2)
        ctk.CTkCheckBox(frm_set, text="Keep temp files", variable=self.var_keep_temps).grid(row=3, column=1, sticky="w")
        ctk.CTkCheckBox(frm_set, text="Download models now", variable=self.var_prefetch).grid(row=3, column=2, sticky="w")
        ctk.CTkCheckBox(frm_set, text="Verify model hash", variable=self.var_strict_hash).grid(row=3, column=3, sticky="w")

        frm_prog = ctk.CTkFrame(col1)
        frm_prog.pack(fill="x", **pad)
        ctk.CTkLabel(frm_prog, text="Download progress").pack(anchor="w")
        self.bar_dl = ctk.CTkProgressBar(frm_prog)
        self.bar_dl.pack(fill="x")
        self.bar_dl.set(0)
        ctk.CTkLabel(frm_prog, text="Process progress").pack(anchor="w", pady=(8, 0))
        self.bar_proc = ctk.CTkProgressBar(frm_prog)
        self.bar_proc.pack(fill="x")
        self.bar_proc.set(0)

        self.var_status = ctk.StringVar(value="Idle")
        ctk.CTkLabel(col1, textvariable=self.var_status).pack(anchor="w", **pad)
        self.txt_log = ctk.CTkTextbox(col1, height=200)
        self.txt_log.pack(fill="both", expand=True, **pad)

        frm_btn = ctk.CTkFrame(col1)
        frm_btn.pack(fill="x", **pad)
        self.btn_run = ctk.CTkButton(frm_btn, text="Start", command=self.start)
        self.btn_run.pack(side="left")
        self.btn_cancel = ctk.CTkButton(frm_btn, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

        self.apply_options()
        logger.debug("UI initialized")

    def log(self, msg: str) -> None:
        logger.info(msg)
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.root.update_idletasks()

    def set_progress(self, which: str, done: int, total: int) -> None:
        if total <= 0:
            return
        value = max(0, min(100, int(done * 100 / total)))
        if which == "download":
            self.bar_dl.set(value / 100)
        else:
            self.bar_proc.set(value / 100)
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

    def browse_weights(self, _event: object | None = None) -> None:
        path = filedialog.askdirectory(title="Select model folder")
        if path:
            self.var_weights.set(path)
            self.cmb_weights.configure(values=[path, "Browse…"])
            self.cmb_weights.set(path)
        else:
            self.cmb_weights.set(self.var_weights.get())

    def browse_work(self, _event: object | None = None) -> None:
        path = filedialog.askdirectory(title="Select work folder")
        if path:
            self.var_work.set(path)
            self.cmb_work.configure(values=[path, "Browse…"])
            self.cmb_work.set(path)
        else:
            self.cmb_work.set(self.var_work.get())

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

    def apply_options(self) -> None:
        """Update widgets from saved options."""
        self.var_output.set(self.opts["output"])
        self.var_w.set(int(self.opts["target_width"]))
        self.var_h.set(int(self.opts["target_height"]))
        self.var_weights.set(self.opts["weights_dir"])
        self.cmb_weights.configure(values=[self.var_weights.get(), "Browse…"])
        self.cmb_weights.set(self.var_weights.get())
        self.var_work.set(self.opts["workdir"])
        self.cmb_work.configure(values=[self.var_work.get(), "Browse…"])
        self.cmb_work.set(self.var_work.get())
        self.var_tile.set(int(self.opts["tile"]))
        self.var_cq.set(int(self.opts["cq"]))
        self.var_codec.set(self.opts["codec"])
        self.var_preset.set(self.opts["preset"])
        self.var_fp16.set(bool(self.opts["use_fp16"]))
        self.var_keep_temps.set(bool(self.opts["keep_temps"]))
        self.var_prefetch.set(bool(self.opts["prefetch_models"]))
        self.var_strict_hash.set(bool(self.opts["strict_model_hash"]))
        logger.debug("Applied options to UI")

    def open_options(self) -> None:
        if getattr(self, "opt_win", None):
            self.opt_win.lift()
            return
        self.opt_win = ctk.CTkToplevel(self.root)
        self.opt_win.title("Options")
        logger.debug("Options window opened")
        tab = ctk.CTkTabview(self.opt_win)
        tab.pack(fill="both", expand=True, padx=10, pady=10)
        paths = tab.add("Paths")
        advanced = tab.add("Encoding")

        vars_map = {
            "output": (ctk.StringVar(value=self.opts["output"]), "Default output", paths),
            "weights_dir": (ctk.StringVar(value=self.opts["weights_dir"]), "Weights folder", paths),
            "workdir": (ctk.StringVar(value=self.opts["workdir"]), "Work folder", paths),
            "target_width": (ctk.IntVar(value=self.opts["target_width"]), "Width", advanced),
            "target_height": (ctk.IntVar(value=self.opts["target_height"]), "Height", advanced),
            "tile": (ctk.IntVar(value=self.opts["tile"]), "Tile", advanced),
            "cq": (ctk.IntVar(value=self.opts["cq"]), "CQ", advanced),
            "codec": (ctk.StringVar(value=self.opts["codec"]), "Codec", advanced),
            "preset": (ctk.StringVar(value=self.opts["preset"]), "Preset", advanced),
            "use_fp16": (ctk.BooleanVar(value=self.opts["use_fp16"]), "Use FP16", advanced),
            "keep_temps": (ctk.BooleanVar(value=self.opts["keep_temps"]), "Keep temps", advanced),
            "prefetch_models": (ctk.BooleanVar(value=self.opts["prefetch_models"]), "Prefetch models", advanced),
            "strict_model_hash": (ctk.BooleanVar(value=self.opts["strict_model_hash"]), "Strict hash", advanced),
        }
        self.opt_vars = {k: v[0] for k, v in vars_map.items()}

        rows = {}
        for key, (var, label, frame) in vars_map.items():
            r = rows.setdefault(frame, 0)
            if isinstance(var, ctk.BooleanVar):
                ctk.CTkCheckBox(frame, text=label, variable=var).grid(row=r, column=0, sticky="w", padx=6, pady=2, columnspan=2)
            else:
                ctk.CTkLabel(frame, text=label).grid(row=r, column=0, sticky="e", padx=6, pady=2)
                ctk.CTkEntry(frame, textvariable=var, width=200).grid(row=r, column=1, sticky="w", padx=6, pady=2)
            rows[frame] += 1

        frm_btn = ctk.CTkFrame(self.opt_win)
        frm_btn.pack(pady=6)
        ctk.CTkButton(frm_btn, text="Defaults", command=self.reset_options).pack(side="left", padx=4)
        ctk.CTkButton(frm_btn, text="Save", command=self.save_options).pack(side="left", padx=4)
        ctk.CTkButton(frm_btn, text="Close", command=self.close_options).pack(side="left", padx=4)

    def reset_options(self) -> None:
        for k, var in self.opt_vars.items():
            var.set(DEFAULTS[k])
        logger.debug("Options reset to defaults")

    def save_options(self) -> None:
        for k, var in self.opt_vars.items():
            self.opts[k] = var.get()
        save_options(self.opts)
        self.apply_options()
        logger.debug("Options saved")

    def close_options(self) -> None:
        if getattr(self, "opt_win", None):
            self.opt_win.destroy()
            self.opt_win = None
            logger.debug("Options window closed")

    def start(self) -> None:
        try:
            cfg = self.collect_cfg()
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            logger.error("Invalid settings: %s", e)
            return

        logger.info("Starting processing")
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.set_status("Starting…")
        self.bar_dl.set(0)
        self.bar_proc.set(0)
        self.txt_log.delete("1.0", "end")

        self.processor = Processor(self)
        t = threading.Thread(target=self.processor.run, args=(cfg,), daemon=True)
        t.start()
        logger.debug("Processing thread launched")

    def cancel(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.set_status("Cancelling…")
            logger.info("Cancel requested")
        self.btn_cancel.config(state="disabled")
        self.btn_run.config(state="normal")
        logger.debug("Cancel handler completed")


def main() -> None:
    ctk.set_appearance_mode("Dark")
    logger.info("Launching GUI")
    root = ctk.CTk()
    root.withdraw()

    orig_destroy = root.destroy
    orig_quit = root.quit

    def report_callback_exception(exc: type[BaseException], val: BaseException, tb: object) -> None:
        logger.exception("Tkinter callback error", exc_info=(exc, val, tb))
        messagebox.showerror("Error", f"{exc.__name__}: {val}")

    def logged_destroy(*args: object, **kwargs: object) -> None:
        logger.debug("root.destroy called", stack_info=True)
        return orig_destroy(*args, **kwargs)

    def logged_quit(*args: object, **kwargs: object) -> None:
        logger.debug("root.quit called", stack_info=True)
        return orig_quit(*args, **kwargs)

    root.bind("<Destroy>", lambda e: logger.debug("<Destroy> event for %s", e.widget))

    root.report_callback_exception = report_callback_exception  # type: ignore[attr-defined]
    root.destroy = logged_destroy  # type: ignore[assignment]
    root.quit = logged_quit  # type: ignore[assignment]

    if not preload(root):
        logger.info("Startup cancelled")
        root.destroy()
        return

    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(0, root.attributes, "-topmost", False)

    def on_close() -> None:
        logger.info("Window close requested")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    app = App(root)
    root.minsize(720, 600)
    logger.info("Entering mainloop")
    try:
        root.mainloop()
    finally:
        logger.info("GUI closed")


if __name__ == "__main__":
    main()
