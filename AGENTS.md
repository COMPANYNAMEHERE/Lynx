# AGENTS.md

> Design notes for the upscaler pipeline, modeled as a set of small “agents” with clear responsibilities and IO contracts.

## Overview

The app turns a **local file or YouTube URL** into an **upscaled 4K (or custom) MP4** using:
- Real‑ESRGAN (GPU, tiles, optional FP16)
- FFmpeg (NVENC)
- Local‑only storage for downloads and temp files

### Project goal

Provide a completely local pipeline for taking any video (file or YouTube link)
and producing a high quality upscale using open models. The code is organised as
small "agents" with clear IO contracts so that new features like denoising or
interpolation can be slotted in easily. Keep contributions focused on a single
agent when possible.

### Directory layout (local-only)
weights/ # model files (.pth) auto-downloaded + verified
work/
downloads/ # yt-dlp outputs (kept)
temp/ # transient chunks (cleaned unless --keep_temps)
outputs/ # encoded results

### Code layout
```
lynx/            # package with all modules
  download.py    # yt-dlp helpers and URL checks
  models.py      # weight fetching and verification
  encode.py      # FFmpeg NVENC interface
  upscale.py     # Real-ESRGAN helpers
  processor.py   # pipeline controller
  gui.py         # Tkinter UI
```


---

## Agents

### 1) SourceResolver Agent
**Purpose:** Decide if `-i` is a URL or file and return a local path.
- **Input:** `input: str`
- **Output:** `Path` to playable file
- **Code:** `is_url()`, part of `Processor.run()`

### 2) Download Agent (YouTube)
**Purpose:** Fetch highest-quality video+audio and merge (no re‑encode).
- **Input:** `url: str`, `downloads_dir`, `temp_dir`
- **Output:** Local `.mkv` (or original container) in `work/downloads/`
- **Side effects:** All temp stays under `work/temp/`
- **Progress:** Hook updates GUI bar
- **Code:** `yt_download()`

### 3) WeightsManager Agent (Models)
**Purpose:** Ensure Real‑ESRGAN weights exist; fetch + SHA‑256 verify if missing.
- **Input:** `weights_dir`, `model_filename`
- **Output:** Local model path
- **Flags:** `--prefetch_models`, `--strict_model_hash`
- **Code:** `ensure_model()`, `_download_with_progress()`

### 4) Probe Agent (Inspect)
**Purpose:** Read input FPS/size/count.
- **Input:** Local video path
- **Output:** `(width, height, fps, nframes)`
- **Code:** `cv2.VideoCapture` usage in `Processor.run()`

### 5) Upscaler Agent (SR)
**Purpose:** Choose model (x2/x4), tile, FP16, run Real‑ESRGAN.
- **Input:** Frames (RGB), `target_scale`
- **Output:** Upscaled frames (RGB), resized to even dims
- **Code:** `pick_model()`, `build_upsampler()`, `up.enhance()`

### 6) Encoder Agent (Mux/Encode)
**Purpose:** Stream RGB frames → FFmpeg NVENC (HEVC/H.264).
- **Input:** RGB frames iterator, fps, codec, preset, cq
- **Output:** Final video in `outputs/`
- **Code:** `run_ffmpeg_encode()`

### 7) Workspace Agent (Storage)
**Purpose:** Create folders, keep downloads, clean temps (unless `--keep_temps`).
- **Code:** `Processor.run()` cleanup section

### 8) Orchestrator UI Agent (GUI)
**Purpose:** Drive the pipeline, binding progress, status, cancel.
- **Code:** `App` (Tkinter), `Processor`

---

## Data Flow
Input (file/URL)
└─> SourceResolver
├─ file ────────────────────────┐
└─ URL ─> Download Agent ───────┘
(work/downloads/*.mkv)

Probe Agent (cv2) ─> decide passthrough vs SR ─> pick model (x2/x4)

[SR path]
Frames (cv2) ─> Upscaler Agent (tiles, FP16) ─> RGB out
└───────────────> Encoder Agent (NVENC) ─> outputs/*.mp4

[Passthrough]
Frames (cv2) ────────────────────────────────┘



---

## Contracts (key functions)

- `yt_download(url, downloads_dir, temp_dir, progress_cb, log_cb) -> Path`
- `ensure_model(weights_dir, filename, strict_hash, progress_cb, log_cb) -> Path`
- `build_upsampler(model_path, base_scale, tile, fp16) -> RealESRGANer`
- `run_ffmpeg_encode(rgb_iter, out_w, out_h, fps, out_path, codec, preset, cq, cancel_event, log_cb)`

Each agent reports via:
- `log_cb(str)` for messages,
- `progress_cb(done:int, total:int)` for bars,
- `cancel_event` for graceful shutdown.

---

## Presets (recommended)

| Profile      | Codec        | Preset | CQ  | Tile | FP16 |
|--------------|--------------|--------|-----|------|------|
| Balanced     | hevc_nvenc   | p5     | 19  | 256  | On   |
| Fast         | h264_nvenc   | p3     | 21  | 224  | On   |
| Max Quality  | hevc_nvenc   | p7     | 18  | 256  | On   |

*Tile*: raise to 320 if VRAM allows; drop to 200–224 if OOM.

---

## Preflight Checks

- **FFmpeg in PATH** with `h264_nvenc` / `hevc_nvenc`.
- **PyTorch + CUDA** match (`torch`/`torchvision` built for same CUDA).
- **Weights** writable at `weights/`.
- **Disk space** in `work/` and `outputs/`.

---

## Common Errors & Fixes

**`ModuleNotFoundError: torchvision.transforms.functional_tensor`**
  Newer torchvision versions moved these helpers. Lynx ships a shim that
  installs automatically on `import lynx`. Ensure you import the package
  before using Real-ESRGAN, or use the BasicSR repo listed in
  `requirements.txt`.

- **`FFmpeg encode failed`**  
  Check codec support: `ffmpeg -hide_banner -encoders | findstr nvenc`. Try `h264_nvenc`, lower preset, or ensure drivers are up to date.

- **CUDA not detected**  
  You’ll run on CPU (very slow). Install NVIDIA drivers + CUDA‑enabled PyTorch wheels.

- **OOM during SR**  
  Lower `Tile`, ensure FP16 is enabled, close other VRAM users.

---

## Extending Agents

- **Batch Agent**: read a list of URLs/files and queue jobs.  
- **Denoise/DeBlock Agent**: optional pre‑filter via FFmpeg (`-vf pp=ac` or `nlmeans`).  
- **Interpolation Agent**: add RIFE for motion smoothing/slow‑mo.
- **TensorRT/ONNX Agent**: export Real‑ESRGAN for extra speed.

---

## Contribution guidelines

- Keep modules small and self contained. Each agent should expose a single
  function or class responsible for one task.
- Use type hints and docstrings for all public functions.
- Ensure new features run without internet access once dependencies and weights
  are present locally.
- Prefer standard library or lightweight dependencies.

---

## Environment

- Python 3.11 (64‑bit)
- PyTorch CUDA 11.8 wheels (for RTX 2080 Super)
- FFmpeg (NVENC)
- Packages per `requirements.txt`

---

## Quick Tests

- **Local file**: small 10–30s 1080p clip → 4K output.
- **YouTube**: short trailer (rights‑cleared) → verify download, SR, encode.
- Run the unit suite with `python tester.py` to check helper functions.

---

## Versioning

- App SemVer starts at `0.1.0`.  
- Note any agent‑contract changes here.

### Changelog
- `0.1.0` – Initial GUI orchestrator; auto model fetch; YouTube download; NVENC encode.

