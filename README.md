# Lynx

Local YouTube upscaling toolkit using Real‑ESRGAN and NVENC.  The
application ships with a PyQt5 interface that stores its settings under your
OS configuration directory, e.g. `~/.config/lynx/settings.json` or
`%LOCALAPPDATA%\Lynx\settings.json`.

## Quick start

1. Install Python 3.11. PyTorch will be installed automatically by the setup script.
2. Create a conda environment and install the requirements using the helper script:
   ```bash
   bash setup.sh
   ```
   Activate it with `conda activate lynx`.
3. Run the GUI:
   ```bash
   python main.py
   ```
   On launch the window shows a status box indicating whether CUDA and
   required weights are detected. If something is missing, follow the
   instructions in the log output.

## Directory layout

These folders are created for you and kept in version control as placeholders:

- `weights/` – Real‑ESRGAN model files.
- `work/` – temporary downloads and intermediate chunks.
  - `work/downloads/` – YouTube downloads.
  - `work/temp/` – transient files cleaned after each run.
- `outputs/` – final encoded videos.
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

If you encounter `ModuleNotFoundError: torchvision.transforms.functional_tensor`,
the package includes a shim that loads automatically. Ensure you import
``lynx`` before using Real-ESRGAN modules so the patch can take effect.
