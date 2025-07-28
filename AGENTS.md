# AGENTS.md

Lynx is organised around small "agents" that each handle one part of the
upscaling pipeline. The key modules live under `lynx/`:

- `download.py` – fetch a YouTube video or accept a local file.
- `models.py` – ensure Real‑ESRGAN weights are present.
- `upscale.py` – run Real‑ESRGAN on incoming frames.
- `encode.py` – stream frames to FFmpeg NVENC.
- `processor.py` – orchestrate the above pieces.
- `gui.py` – simple Tkinter interface.

The project aims to run fully offline once dependencies and model weights are
available locally. Unit tests are provided under `tests/` and can be executed
with `python tests/tester.py`.

Guidelines for contributions:

- Keep modules focused on a single task and use type hints.
- Avoid heavyweight dependencies when possible.
- Ensure new features work without requiring internet access at runtime.

Environment assumptions:

- Python 3.11
- PyTorch with CUDA 11.8
- FFmpeg with NVENC support

See `README.md` for setup instructions.
