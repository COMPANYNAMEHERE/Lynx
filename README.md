# Lynx

Lynx now focuses on preloading YouTube videos for smooth offline playback.
Given a URL the tool downloads the entire stream to disk and immediately
upscales it with Real‑ESRGAN before handing the result to FFmpeg for fast
NVENC encoding.  The legacy GUI is still available, but the recommended entry
point is the new preloader CLI.

## Quick start

1. Install Python 3.11. The setup script installs PyTorch for your detected CUDA
   version using the official CUDA wheels.
2. Run the setup script and follow the prompts to create or update the conda environment.  When resetting an existing environment you can choose to redownload all packages.  Each package installs with its own progress display.  When it finishes it prints the exact commands to run next:
   ```bash
   bash setup.sh
   ```
   The log of all actions is saved to `setup/setup.log` for troubleshooting (git ignores this file by default).
3. Activate the environment and run the preloader CLI:
   ```bash
   conda activate lynx
   python -m lynx.cli "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
   ```
   The command downloads the video, upscales it according to your chosen
   dimensions and writes the result to the `outputs/` directory.  Use
   `python -m lynx.cli -h` to see all available options.

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
- **GPU detected but PyTorch CPU-only** – Rerun ``setup.sh``. If it still installs
  the CPU build, reinstall manually for your CUDA version, e.g.:
  ```bash
  conda run -n lynx pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **`conda` not found** – Install Miniconda or Anaconda and make sure ``conda``
  is on your ``PATH``.
- **`yt-dlp` missing** – Install it inside the environment with
  ``pip install yt-dlp``.
- **FFmpeg not detected** – Install FFmpeg with NVENC support and ensure it is
  available on your ``PATH``.
