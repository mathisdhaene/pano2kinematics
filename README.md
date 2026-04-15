# pano2kinematics


**Real-time markerless upper-body kinematics from a single 360° camera**, using
**Neural Localizer Fields (NLF)** and **YOLO**.


---

## Features

* Single 360° camera input (equirectangular)
* Real-time inference (CPU or GPU)
* YOLO-based human detection & tracking
* NLF-based anatomical landmark regression
* GStreamer shared-memory input
* No markers, no MoCap system

---

## Requirements (summary)

* Linux (tested on Ubuntu 22.04 / 24.04)
* Python **3.10**
* **uv** for environment management
* **CPU-only** supported (GPU optional)
* No `sudo` required for the Python environment
* System GStreamer / GI packages are required for live mode

---

## 1. Python environment

### 1.1 Install uv (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal and verify:

```bash
uv --version
```

---

### 1.2 Create the environment

From the `deps/pano2kinematics` directory:

```bash
uv venv --python 3.10 --system-site-packages
uv sync
```

This will:

* create `.venv` next to the project files
* install all Python dependencies from `pyproject.toml` and `uv.lock`
* keep the environment on Python 3.10
* expose system GI bindings via `--system-site-packages`
* replace the old conda/mamba workflow; `environment.yml` is historical now

---

### 1.3 Sanity check

```bash
uv run python - <<'PY'
import torch, cv2, smplx, pyrender, transformers, ultralytics
print("All imports OK")
PY
```

---

## 2. GStreamer / GI bindings

The live pipeline uses **GStreamer** via Python GI bindings.

```bash
sudo apt install --yes \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-libav \
  python3-gi \
  python3-gi-cairo \
  libgirepository1.0-dev
```

Quick test:

```bash
uv run python - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
print("GStreamer OK")
PY
```

---

## 3. Model weights

### 3.1 Expected directory structure

```text
weights/
├── nlf
│   ├── nlf_l_multi_0.3.2.torchscript
│   └── nlf_s_multi.torchscript
└── yolo_models
```

---

### 3.2 NLF weights

Download the TorchScript models from:

👉 [https://github.com/isarandi/nlf/releases](https://github.com/isarandi/nlf/releases)

Place them in:

```bash
weights/nlf/
```

---

### 3.3 YOLO weights

YOLO weights are handled by **Ultralytics** and are typically downloaded automatically.

If you prefer manual control, place them in:

```bash
weights/yolo_models/
```

---

## 4. Canonical vertex templates (NLF)

NLF requires **canonical SMPL / SMPL-H / SMPL-X vertex templates**.

These are included in the repository:

```text
pipeline_nlf/canonical_verts/
├── smpl.npy
├── smplh.npy
├── smplh16.npy
└── smplx.npy
```

---

## 5. EquiLib (projection)

This repository includes a local copy of **EquiLib** for equirectangular-to-perspective projection:

👉 [https://github.com/haruishi43/equilib](https://github.com/haruishi43/equilib)

EquiLib is vendored directly in the source tree to ensure reproducibility.

---

## 6. Running the pipeline (live)

There are **two live-related scripts**:

* `legacy/live_cpu.py`
* `live_gpu.py`

`live_gpu.py` is the main entry point. The CPU script is kept in `legacy/` as a reference implementation.

---

### CPU version

```bash
uv run python legacy/live_cpu.py \
  --live \
  --shm-socket /tmp/theta_bgr.sock \
  --fps 30 \
  --device cpu \
  --yolo weights/yolo_models/yolo11m-pose.pt \
  --tracker bytetrack.yaml \
  --bio-cfg configs/biomeca.yaml \
  --nlf-weights weights/nlf/nlf_s_multi.torchscript \
  --display-only
```

---

### GPU version

```bash
uv run python live_gpu.py \
  --live \
  --shm-socket /tmp/theta_bgr.sock \
  --frame-width 3840 \
  --frame-height 1920 \
  --fps 30 \
  --device cuda \
  --yolo weights/yolo_models/yolo11m-pose.pt \
  --tracker bytetrack.yaml \
  --bio-cfg configs/biomeca.yaml \
  --nlf-weights weights/nlf/nlf_s_multi.torchscript \
  --display-only
```

---

## Acknowledgements

This project builds upon the following open-source projects:

* **Neural Localizer Fields (NLF)**
  [https://github.com/isarandi/nlf](https://github.com/isarandi/nlf)

* **Ultralytics YOLO**
  [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

* **EquiLib**
  [https://github.com/haruishi43/equilib](https://github.com/haruishi43/equilib)

Users are encouraged to cite the original works when using these components in academic publications.

---

## Project status

This repository is used for ongoing research and experimentation.
The API and scripts may evolve as the pipeline is refined.
