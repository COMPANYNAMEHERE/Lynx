# Lynx

Local YouTube Upscaling toolkit using Real-ESRGAN and NVENC.

## Quick start

1. Install Python 3.11 and PyTorch with CUDA support.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the GUI:
   ```bash
   python -m lynx
   ```

## Directory layout

These folders are created for you and kept in version control as placeholders:

- `weights/` – Real‑ESRGAN model files.
- `work/` – temporary downloads and intermediate chunks.
  - `work/downloads/` – YouTube downloads.
  - `work/temp/` – transient files cleaned after each run.
- `outputs/` – final encoded videos.
- `logs/` – run logs and diagnostics.

Feel free to remove the placeholder files if you wish; the application will recreate any missing directories at runtime.

## Self tests

Run the bundled unit tests to verify basic functionality:

```bash
python tester.py
```
