# Lynx

Local YouTube upscaling toolkit using Real‑ESRGAN and NVENC.  The
application ships with a PyQt5 interface that stores its settings under your
OS configuration directory, e.g. `~/.config/lynx/settings.json` or
`%LOCALAPPDATA%\Lynx\settings.json`.

## Quick start

1. Install Python 3.11. PyTorch will be installed automatically by the setup script.
   The script detects CUDA via `nvidia-smi` or `nvcc` and installs the matching build.
2. Run the setup script and follow the prompts to create or update the conda environment.  The script shows a summary of your CUDA status and waits for input before exiting:
   ```bash
   bash setup.sh
   ```
   Activate it with `conda activate lynx`.
   The script logs all actions to `setup/setup.log` for troubleshooting.
3. Run the GUI:
   ```bash
   python main.py
   ```
   On launch the window shows a status box indicating whether CUDA and
   required weights are detected. If something is missing, follow the
   instructions in the log output. If you see "GPU detected but PyTorch
   CPU-only", reinstall PyTorch with CUDA support.

## Directory layout

These folders are created for you and kept in version control as placeholders:

- `weights/` – Real‑ESRGAN model files.
- `work/` – temporary downloads and intermediate chunks.
  - `work/downloads/` – YouTube downloads.
  - `work/temp/` – transient files cleaned after each run.
- `outputs/` – final encoded videos.
- `setup.sh` – creates/updates the conda environment and installs PyTorch.
- `setup/` – contains `setup.log`.
- `logs/` – legacy location for run logs. Recent versions store logs under
  the user's OS data directory (e.g. `~/.local/state/lynx/logs` or
  `%LOCALAPPDATA%\Lynx\logs`).
- `options/` – legacy settings directory. Current releases use the user's OS
  configuration directory instead (e.g. `~/.config/lynx`).

Feel free to remove the placeholder files if you wish; the application will recreate any missing directories at runtime.

## Self tests

Run the bundled unit tests to verify basic functionality:

```bash
python tests/tester.py
```

## Troubleshooting

Common issues:

- **Missing torchvision functional_tensor** – Import ``lynx`` first so the included
  shim can patch torchvision automatically.
- **GPU detected but PyTorch CPU-only** – Rerun ``setup.sh`` or reinstall
  PyTorch with CUDA support.
- **`conda` not found** – Install Miniconda or Anaconda and make sure ``conda``
  is on your ``PATH``.
- **`yt-dlp` missing** – Install it inside the environment with
  ``pip install yt-dlp``.
- **FFmpeg not detected** – Install FFmpeg with NVENC support and ensure it is
  available on your ``PATH``.
