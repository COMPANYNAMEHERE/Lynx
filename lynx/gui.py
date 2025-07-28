"""PyQt5 GUI for the Lynx upscaler."""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess
import torch

from .download import is_url, yt_title
from .models import ensure_model, MODEL_SPECS

from .processor import Processor
from .options import load_options, save_options, DEFAULTS
from .logger import get_logger

logger = get_logger()


class LogHandler(logging.Handler):
    """Forward log records to a ``QPlainTextEdit``."""

    def __init__(self, widget: QtWidgets.QPlainTextEdit) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI
        msg = self.format(record)
        self.widget.appendPlainText(msg)
        self.widget.verticalScrollBar().setValue(
            self.widget.verticalScrollBar().maximum()
        )


class OptionsDialog(QtWidgets.QDialog):
    """Dialog for editing persistent options."""

    def __init__(self, opts: dict, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.opts = opts.copy()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        paths = QtWidgets.QWidget()
        enc = QtWidgets.QWidget()
        tabs.addTab(paths, "Paths")
        tabs.addTab(enc, "Encoding")

        form_paths = QtWidgets.QFormLayout(paths)
        form_enc = QtWidgets.QFormLayout(enc)

        self.ed_out = QtWidgets.QLineEdit(self.opts.get("output", DEFAULTS["output"]))
        self.ed_weights = QtWidgets.QLineEdit(
            self.opts.get("weights_dir", DEFAULTS["weights_dir"])
        )
        self.ed_work = QtWidgets.QLineEdit(self.opts.get("workdir", DEFAULTS["workdir"]))
        form_paths.addRow("Default output", self.ed_out)
        form_paths.addRow("Weights folder", self.ed_weights)
        form_paths.addRow("Work folder", self.ed_work)

        self.sp_w = QtWidgets.QSpinBox()
        self.sp_w.setRange(64, 16384)
        self.sp_w.setValue(int(self.opts.get("target_width", DEFAULTS["target_width"])))
        self.sp_h = QtWidgets.QSpinBox()
        self.sp_h.setRange(64, 16384)
        self.sp_h.setValue(int(self.opts.get("target_height", DEFAULTS["target_height"])))
        self.sp_tile = QtWidgets.QSpinBox()
        self.sp_tile.setRange(16, 1024)
        self.sp_tile.setValue(int(self.opts.get("tile", DEFAULTS["tile"])))
        self.sp_cq = QtWidgets.QSpinBox()
        self.sp_cq.setRange(0, 51)
        self.sp_cq.setValue(int(self.opts.get("cq", DEFAULTS["cq"])))
        self.cmb_codec = QtWidgets.QComboBox()
        self.cmb_codec.addItems(["hevc_nvenc", "h264_nvenc"])
        self.cmb_codec.setCurrentText(self.opts.get("codec", DEFAULTS["codec"]))
        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.addItems([f"p{i}" for i in range(1, 8)])
        self.cmb_preset.setCurrentText(self.opts.get("preset", DEFAULTS["preset"]))
        self.chk_fp16 = QtWidgets.QCheckBox()
        self.chk_fp16.setChecked(bool(self.opts.get("use_fp16", DEFAULTS["use_fp16"])))
        self.chk_keep = QtWidgets.QCheckBox()
        self.chk_keep.setChecked(bool(self.opts.get("keep_temps", DEFAULTS["keep_temps"])))
        self.chk_prefetch = QtWidgets.QCheckBox()
        self.chk_prefetch.setChecked(bool(self.opts.get("prefetch_models", DEFAULTS["prefetch_models"])))
        self.chk_strict = QtWidgets.QCheckBox()
        self.chk_strict.setChecked(bool(self.opts.get("strict_model_hash", DEFAULTS["strict_model_hash"])))

        form_enc.addRow("Width", self.sp_w)
        form_enc.addRow("Height", self.sp_h)
        form_enc.addRow("Tile", self.sp_tile)
        form_enc.addRow("CQ", self.sp_cq)
        form_enc.addRow("Codec", self.cmb_codec)
        form_enc.addRow("Preset", self.cmb_preset)
        form_enc.addRow("Use FP16", self.chk_fp16)
        form_enc.addRow("Keep temp files", self.chk_keep)
        form_enc.addRow("Prefetch models", self.chk_prefetch)
        form_enc.addRow("Verify model hash", self.chk_strict)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        btn_row.addStretch(1)
        btn_defaults = QtWidgets.QPushButton("Defaults")
        btn_save = QtWidgets.QPushButton("Save")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_defaults)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_close)
        btn_defaults.clicked.connect(self.set_defaults)
        btn_save.clicked.connect(self.accept)
        btn_close.clicked.connect(self.reject)

    def set_defaults(self) -> None:
        self.ed_out.setText(DEFAULTS["output"])
        self.ed_weights.setText(DEFAULTS["weights_dir"])
        self.ed_work.setText(DEFAULTS["workdir"])
        self.sp_w.setValue(int(DEFAULTS["target_width"]))
        self.sp_h.setValue(int(DEFAULTS["target_height"]))
        self.sp_tile.setValue(int(DEFAULTS["tile"]))
        self.sp_cq.setValue(int(DEFAULTS["cq"]))
        self.cmb_codec.setCurrentText(DEFAULTS["codec"])
        self.cmb_preset.setCurrentText(DEFAULTS["preset"])
        self.chk_fp16.setChecked(bool(DEFAULTS["use_fp16"]))
        self.chk_keep.setChecked(bool(DEFAULTS["keep_temps"]))
        self.chk_prefetch.setChecked(bool(DEFAULTS["prefetch_models"]))
        self.chk_strict.setChecked(bool(DEFAULTS["strict_model_hash"]))

    def accept(self) -> None:  # pragma: no cover - UI
        self.opts["output"] = self.ed_out.text().strip()
        self.opts["weights_dir"] = self.ed_weights.text().strip()
        self.opts["workdir"] = self.ed_work.text().strip()
        self.opts["target_width"] = self.sp_w.value()
        self.opts["target_height"] = self.sp_h.value()
        self.opts["tile"] = self.sp_tile.value()
        self.opts["cq"] = self.sp_cq.value()
        self.opts["codec"] = self.cmb_codec.currentText()
        self.opts["preset"] = self.cmb_preset.currentText()
        self.opts["use_fp16"] = self.chk_fp16.isChecked()
        self.opts["keep_temps"] = self.chk_keep.isChecked()
        self.opts["prefetch_models"] = self.chk_prefetch.isChecked()
        self.opts["strict_model_hash"] = self.chk_strict.isChecked()
        save_options(self.opts)
        logger.debug("Options saved: %s", self.opts)
        super().accept()


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lynx Upscaler")
        self.processor: Optional[Processor] = None
        self.thread: Optional[threading.Thread] = None
        self.opts = load_options()
        logger.debug("Options loaded: %s", self.opts)
        self._init_ui()

    # UI construction
    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        form_in = QtWidgets.QHBoxLayout()
        layout.addLayout(form_in)
        self.ed_in = QtWidgets.QLineEdit()
        form_in.addWidget(QtWidgets.QLabel("Video file or YouTube link:"))
        form_in.addWidget(self.ed_in)
        btn_in = QtWidgets.QPushButton("Browse")
        form_in.addWidget(btn_in)
        btn_in.clicked.connect(self.browse_input)

        form_out = QtWidgets.QHBoxLayout()
        layout.addLayout(form_out)
        self.ed_out = QtWidgets.QLineEdit(self.opts.get("output", DEFAULTS["output"]))
        form_out.addWidget(QtWidgets.QLabel("Save output to:"))
        form_out.addWidget(self.ed_out)
        btn_out = QtWidgets.QPushButton("Browse")
        form_out.addWidget(btn_out)
        btn_out.clicked.connect(self.browse_output)

        form_model = QtWidgets.QHBoxLayout()
        layout.addLayout(form_model)
        form_model.addWidget(QtWidgets.QLabel("Model:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(list(MODEL_SPECS.keys()))
        form_model.addWidget(self.cmb_model)
        self.btn_model_dl = QtWidgets.QPushButton("Download")
        form_model.addWidget(self.btn_model_dl)
        self.lbl_model_status = QtWidgets.QLabel()
        form_model.addWidget(self.lbl_model_status)
        self.btn_model_dl.clicked.connect(self.download_model)
        self.cmb_model.currentIndexChanged.connect(self.update_model_status)
        self.update_model_status()

        self.bar_dl = QtWidgets.QProgressBar()
        self.bar_proc = QtWidgets.QProgressBar()
        layout.addWidget(QtWidgets.QLabel("Download progress"))
        layout.addWidget(self.bar_dl)
        layout.addWidget(QtWidgets.QLabel("Process progress"))
        layout.addWidget(self.bar_proc)

        self.status_box = QtWidgets.QPlainTextEdit(readOnly=True)
        layout.addWidget(self.status_box)

        self.log_widget = QtWidgets.QPlainTextEdit(readOnly=True)
        layout.addWidget(self.log_widget, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)

        self.status_label = QtWidgets.QLabel("Idle")
        self.statusBar().addWidget(self.status_label)

        menu = self.menuBar().addMenu("Settings")
        act_opts = menu.addAction("Options")
        act_opts.triggered.connect(self.open_options)

        # Attach logger to text widget
        handler = LogHandler(self.log_widget)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger("lynx").addHandler(handler)
        logger.debug("UI initialized")
        self.update_status_box()

    # UI helper methods
    def browse_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input video")
        if path:
            self.ed_in.setText(path)
            self.auto_set_output()

    def auto_set_output(self) -> None:
        """Set the output path based on the input title if using default name."""
        out_text = self.ed_out.text().strip()
        default_out = DEFAULTS["output"]
        if not out_text or out_text == default_out:
            inp = self.ed_in.text().strip()
            if not inp:
                return
            if is_url(inp):
                title = yt_title(inp)
            else:
                title = Path(inp).stem
            safe = "".join(c for c in title if c.isalnum() or c in " -_()").rstrip()
            if not safe:
                safe = "output"
            out_path = Path(default_out).parent / f"{safe}.mp4"
            self.ed_out.setText(str(out_path))

    def browse_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save output as", str(Path("outputs") / "output.mp4"), "MP4 files (*.mp4)"
        )
        if path:
            self.ed_out.setText(path)

    def open_options(self) -> None:
        dlg = OptionsDialog(self.opts, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.opts = dlg.opts
            save_options(self.opts)
            self.apply_options()

    def apply_options(self) -> None:
        self.ed_out.setText(self.opts.get("output", DEFAULTS["output"]))
        self.update_status_box()

    def update_status_box(self) -> None:
        """Check environment and display status messages."""
        lines = []
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            lines.append("FFmpeg: OK")
        except Exception:
            lines.append("FFmpeg: not found")

        if torch.cuda.is_available():
            lines.append("CUDA: available")
        else:
            lines.append("CUDA: not detected (CPU mode)")

        weights_dir = Path(self.opts.get("weights_dir", DEFAULTS["weights_dir"]))
        model_file = getattr(self, "cmb_model", None)
        model_name = model_file.currentText() if model_file else "RealESRGAN_x4plus.pth"
        model_path = weights_dir / model_name
        lines.append(
            f"Model: present" if model_path.exists() else f"Model: missing {model_name}"
        )
        if model_file:
            self.update_model_status()

        self.status_box.setPlainText("\n".join(lines))
        logger.debug("Status updated: %s", lines)

    def update_model_status(self) -> None:
        """Update the label showing whether the selected model is present."""
        weights_dir = Path(self.opts.get("weights_dir", DEFAULTS["weights_dir"]))
        model_name = self.cmb_model.currentText()
        if (weights_dir / model_name).exists():
            self.lbl_model_status.setText("Model online")
            self.lbl_model_status.setStyleSheet("color: green")
        else:
            self.lbl_model_status.setText("Model offline")
            self.lbl_model_status.setStyleSheet("color: red")

    def download_model(self) -> None:
        """Fetch the currently selected model."""
        model_name = self.cmb_model.currentText()
        weights_dir = Path(self.opts.get("weights_dir", DEFAULTS["weights_dir"]))
        try:
            ensure_model(
                weights_dir,
                model_name,
                strict_hash=self.opts.get("strict_model_hash", False),
                progress_cb=lambda d, t: self.set_progress("download", d, t or 1),
                log_cb=self.log,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Download failed", str(e))
            logger.error("Model download failed: %s", e)
        self.update_model_status()

    def log(self, msg: str) -> None:
        logger.info(msg)

    def set_progress(self, which: str, done: int, total: int) -> None:
        if total <= 0:
            return
        val = max(0, min(100, int(done * 100 / total)))
        if which == "download":
            self.bar_dl.setValue(val)
        else:
            self.bar_proc.setValue(val)

    def set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    # Processing controls
    def collect_cfg(self) -> dict:
        inp = self.ed_in.text().strip()
        out = self.ed_out.text().strip()
        if not inp:
            raise RuntimeError("Please set an input (file or YouTube URL).")
        if not out:
            raise RuntimeError("Please choose an output file.")
        cfg = self.opts.copy()
        cfg.update({"input": inp, "output": out, "model": self.cmb_model.currentText()})
        return cfg

    def start(self) -> None:
        try:
            self.auto_set_output()
            cfg = self.collect_cfg()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid settings", str(e))
            logger.error("Invalid settings: %s", e)
            return
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar_dl.setValue(0)
        self.bar_proc.setValue(0)
        self.log_widget.clear()
        self.set_status("Starting…")
        self.processor = Processor(self)
        self.thread = threading.Thread(target=self.processor.run, args=(cfg,), daemon=True)
        self.thread.start()
        logger.debug("Processing thread launched")

    def cancel(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.set_status("Cancelling…")
            logger.info("Cancel requested")
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.processor:
            self.processor.cancel()
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    logger.info("Entering mainloop")
    app.exec_()
    logger.info("GUI closed")


if __name__ == "__main__":
    main()
