from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional


def detect_gpu_info() -> Optional[str]:
    """Return a short string describing a detected NVIDIA GPU or None."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
            line = out.strip().splitlines()[0] if out.strip() else ""
            if line:
                return line
            return "nvidia-smi present"
        except Exception:
            pass
    if shutil.which("nvcc"):
        try:
            out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
            m = re.search(r"release ([0-9.]+)", out)
            ver = m.group(1) if m else ""
            return f"nvcc {ver}".strip()
        except Exception:
            pass
    return None
