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

from .processor import Processor
from .options import load_options, save_options, DEFAULTS
from .logger import get_logger

logger = get_logger()


class LogHandler(logging.Handler, QtCore.QObject):
    """Forward log records to a ``QPlainTextEdit`` safely across threads."""

    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, widget: QtWidgets.QPlainTextEdit) -> None:
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self)
        self.widget = widget
        self.log_signal.connect(self._append)

    @QtCore.pyqtSlot(str)
    def _append(self, msg: str) -> None:  # pragma: no cover - UI
        self.widget.appendPlainText(msg)
        self.widget.verticalScrollBar().setValue(
            self.widget.verticalScrollBar().maximum()
        )

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI
        msg = self.format(record)
        self.log_signal.emit(msg)


class ProcessorThread(QtCore.QThread):
    """Qt thread running the ``Processor``."""

    def __init__(self, proc: Processor, cfg: dict) -> None:
        super().__init__()
        self.proc = proc
        self.cfg = cfg
        self.error: Optional[Exception] = None

    def run(self) -> None:  # pragma: no cover - integration
        try:
            self.proc.run(self.cfg)
        except Exception as e:
            self.error = e
            logger.exception("Processing error")


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
        btn_out = QtWidgets.QPushButton("Browse")
        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.ed_out)
        row_out.addWidget(btn_out)
        wrap_out = QtWidgets.QWidget()
        wrap_out.setLayout(row_out)

        self.ed_weights = QtWidgets.QLineEdit(
            self.opts.get("weights_dir", DEFAULTS["weights_dir"])
        )
        btn_weights = QtWidgets.QPushButton("Browse")
        row_weights = QtWidgets.QHBoxLayout()
        row_weights.addWidget(self.ed_weights)
        row_weights.addWidget(btn_weights)
        wrap_weights = QtWidgets.QWidget()
        wrap_weights.setLayout(row_weights)

        self.ed_work = QtWidgets.QLineEdit(self.opts.get("workdir", DEFAULTS["workdir"]))
        btn_work = QtWidgets.QPushButton("Browse")
        row_work = QtWidgets.QHBoxLayout()
        row_work.addWidget(self.ed_work)
        row_work.addWidget(btn_work)
        wrap_work = QtWidgets.QWidget()
        wrap_work.setLayout(row_work)

        form_paths.addRow("Output folder", wrap_out)
        form_paths.addRow("Weights folder", wrap_weights)
        form_paths.addRow("Work folder", wrap_work)

        btn_out.clicked.connect(lambda: self._pick_dir(self.ed_out))
        btn_weights.clicked.connect(lambda: self._pick_dir(self.ed_weights))
        btn_work.clicked.connect(lambda: self._pick_dir(self.ed_work))

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

    def _pick_dir(self, edit: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if path:
            edit.setText(path)

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

    log_signal = QtCore.pyqtSignal(str)
    progress_signal = QtCore.pyqtSignal(str, int, int)
    status_signal = QtCore.pyqtSignal(str)
    overwrite_request = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lynx Upscaler")
        self.processor: Optional[Processor] = None
        self.thread: Optional[QtCore.QThread] = None
        self.opts = load_options()
        logger.debug("Options loaded: %s", self.opts)
        self._init_ui()
        self.overwrite_request.connect(self._on_overwrite_request)
        self._overwrite_event = threading.Event()
        self._overwrite_answer = False

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

        quality_row = QtWidgets.QHBoxLayout()
        layout.addLayout(quality_row)
        self.btn_quality = QtWidgets.QPushButton()
        # allow user to pick one of four model quality tiers
        self._init_quality_button()
        quality_row.addWidget(QtWidgets.QLabel("Quality"))
        quality_row.addWidget(self.btn_quality)

        self.bar_dl = QtWidgets.QProgressBar()
        self.bar_dl.setRange(0, 100)
        self.bar_dl.setValue(0)
        self.bar_proc = QtWidgets.QProgressBar()
        self.bar_proc.setRange(0, 100)
        self.bar_proc.setValue(0)
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

        # cross-thread GUI updates
        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.status_signal.connect(self.status_label.setText)

        logger.debug("UI initialized")
        self.update_status_box()

    # UI helper methods
    def browse_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input video")
        if path:
            self.ed_in.setText(path)

    def browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(Path("outputs"))
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
        self.btn_quality.setText(self.opts.get("model_quality", DEFAULTS["model_quality"]).title())
        self.update_status_box()

    def _init_quality_button(self) -> None:
        """Populate the quality picker menu with model descriptions."""
        quality = self.opts.get("model_quality", DEFAULTS["model_quality"])
        self.btn_quality.setText(quality.title())
        # show mapping between friendly names and actual model architectures
        self.btn_quality.setToolTip(
            "quick=RealESRGAN, normal=Swin2SR, better=HAT, best=AdcSR"
        )
        menu = QtWidgets.QMenu(self.btn_quality)
        items = [
            ("quick", "Quick – RealESRGAN"),
            ("normal", "Normal – Swin2SR"),
            ("better", "Better – HAT"),
            ("best", "Best – AdcSR"),
        ]
        for name, label in items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, n=name: self.set_quality(n))
        self.btn_quality.setMenu(menu)

    def set_quality(self, quality: str) -> None:
        """Store chosen model quality tier and update button text."""
        self.opts["model_quality"] = quality
        self.btn_quality.setText(quality.title())
        save_options(self.opts)

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
            from .env import detect_gpu_info

            info = detect_gpu_info()
            if info:
                lines.append("CUDA: GPU detected but PyTorch CPU-only")
            else:
                lines.append("CUDA: not detected (CPU mode)")

        weights_dir = Path(self.opts.get("weights_dir", DEFAULTS["weights_dir"]))
        model = weights_dir / "RealESRGAN_x4plus.pth"
        lines.append(
            "Model: present" if model.exists() else "Model: missing RealESRGAN_x4plus.pth"
        )

        self.status_box.setPlainText("\n".join(lines))
        logger.debug("Status updated: %s", lines)

    @QtCore.pyqtSlot(str)
    def _append_log(self, msg: str) -> None:  # pragma: no cover - UI
        self.log_widget.appendPlainText(msg)
        self.log_widget.verticalScrollBar().setValue(
            self.log_widget.verticalScrollBar().maximum()
        )

    @QtCore.pyqtSlot(str, int, int)
    def _update_progress(self, which: str, done: int, total: int) -> None:
        bar = self.bar_dl if which == "download" else self.bar_proc
        if total <= 0:
            bar.setRange(0, 0)
            return
        bar.setRange(0, 100)
        val = max(0, min(100, int(done * 100 / total)))
        bar.setValue(val)

    def log(self, msg: str) -> None:
        self.log_signal.emit(msg)

    def set_progress(self, which: str, done: int, total: int) -> None:
        self.progress_signal.emit(which, done, total)

    def set_status(self, msg: str) -> None:
        self.status_signal.emit(msg)

    @QtCore.pyqtSlot(str)
    def _on_overwrite_request(self, path: str) -> None:
        ans = QtWidgets.QMessageBox.question(
            self, "Overwrite file", f"{path} exists. Overwrite?"
        ) == QtWidgets.QMessageBox.Yes
        self._overwrite_answer = ans
        self._overwrite_event.set()

    def confirm_overwrite(self, path: str) -> bool:
        if QtCore.QThread.currentThread() is self.thread():
            return (
                QtWidgets.QMessageBox.question(
                    self, "Overwrite file", f"{path} exists. Overwrite?"
                )
                == QtWidgets.QMessageBox.Yes
            )
        self._overwrite_event.clear()
        self.overwrite_request.emit(path)
        self._overwrite_event.wait()
        return bool(self._overwrite_answer)

    # Processing controls
    def collect_cfg(self) -> Optional[dict]:
        inp = self.ed_in.text().strip()
        out = self.ed_out.text().strip()
        ok = True
        if not inp:
            self.ed_in.setStyleSheet("border: 1px solid red;")
            ok = False
        else:
            self.ed_in.setStyleSheet("")
        if not out:
            self.ed_out.setStyleSheet("border: 1px solid red;")
            ok = False
        else:
            self.ed_out.setStyleSheet("")
        if not ok:
            return None
        cfg = self.opts.copy()
        cfg.update({"input": inp, "output": out})
        return cfg

    def start(self) -> None:
        cfg = self.collect_cfg()
        if cfg is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid settings",
                "Please correct the highlighted fields.",
            )
            logger.error("Invalid settings")
            return
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar_dl.setRange(0, 100)
        self.bar_dl.setValue(0)
        self.bar_proc.setRange(0, 100)
        self.bar_proc.setValue(0)
        self.log_widget.clear()
        self.set_status("Starting…")
        self.processor = Processor(self)
        self.thread = ProcessorThread(self.processor, cfg)
        self.thread.finished.connect(self.processing_finished)
        self.thread.start()
        logger.debug("Processing thread launched")

    def processing_finished(self) -> None:  # pragma: no cover - UI
        error = None
        out_file: Optional[Path] = None
        if self.thread:
            self.thread.wait()
            error = self.thread.error
        if self.processor:
            path = getattr(self.processor, "output_path", None)
            if path:
                out_file = Path(path)
        self.thread = None
        self.processor = None
        self.bar_dl.setRange(0, 100)
        self.bar_dl.setValue(0)
        self.bar_proc.setRange(0, 100)
        self.bar_proc.setValue(0)
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if error:
            QtWidgets.QMessageBox.critical(self, "Error", str(error))
            logger.error("Processing failed: %s", error)
            self.set_status("Error")
        elif self.status_label.text().startswith("Cancelling"):
            self.set_status("Cancelled")
        else:
            self.set_status("Finished")
            if out_file:
                ans = QtWidgets.QMessageBox.question(
                    self,
                    "Open folder",
                    f"Open output folder?\n{out_file.parent}",
                )
                if ans == QtWidgets.QMessageBox.Yes:
                    QtGui.QDesktopServices.openUrl(
                        QtCore.QUrl.fromLocalFile(str(out_file.parent))
                    )

    def cancel(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.set_status("Cancelling…")
            logger.info("Cancel requested")
            if self.thread:
                self.thread.wait()
        self.btn_cancel.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.bar_dl.setRange(0, 100)
        self.bar_dl.setValue(0)
        self.bar_proc.setRange(0, 100)
        self.bar_proc.setValue(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.processor:
            self.processor.cancel()
            if self.thread:
                self.thread.wait()
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
