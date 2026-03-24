# pano2kinematics

**Real-time markerless upper-body kinematics from a single 360° camera**, using
**Neural Localizer Fields (NLF)** and **YOLO**.

Part of the **PANORAMICS** (Panoramic Analysis for Natural Observation and Real-time Assessment of Markerless Integrated Capture Systems) umbrella project.

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
* **CPU-only** supported (GPU optional)
* No `sudo` required
* `conda-forge` only (HPC / CNRS friendly)

---

## 1. Python environment

### 1.1 Install Miniforge (once)

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Restart your terminal and verify:

```bash
conda --version
```

---

### 1.2 Configure conda (conda-forge only)

```bash
cat > ~/.condarc <<'EOF'
channels:
  - conda-forge
channel_priority: strict
default_channels: []
custom_channels: {}
custom_multichannels: {}
auto_activate_base: false
EOF
```

---

### 1.3 Install mamba (recommended)

```bash
conda install -n base -c conda-forge mamba
```

---

### 1.4 Create the environment

The repository already provides a ready-to-use environment file.

From the repository root:

```bash
mamba env create -f environment.yml
conda activate py10
```

---

### 1.5 Install PyTorch

#### CPU-only

```bash
mamba install -c conda-forge pytorch-cpu torchvision torchdata
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

### 1.6 Install pip-only dependencies

```bash
mamba install -c conda-forge pip setuptools wheel
```

```bash
python -m pip install \
  smplx \
  pyrender==0.1.45 \
  transformers \
  ultralytics
```

---

### 1.7 Sanity check

```bash
python - <<'PY'
import torch, cv2, smplx, pyrender, transformers, ultralytics
print("All imports OK")
PY
```

---

## 2. GStreamer / GI bindings

The live pipeline uses **GStreamer** via Python GI bindings.

Install inside the conda environment:

```bash
conda activate py10
mamba install -c conda-forge pygobject gst-python gstreamer
```

Quick test:

```bash
python - <<'PY'
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
python3 legacy/live_cpu.py \
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
python3 live_gpu.py \
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
