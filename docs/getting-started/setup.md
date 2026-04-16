# Setup & Requirements

pycasa is structured around **optional extras** — install only the dependencies your workflow actually needs. The base package provides core utilities; each feature group (I/O, detection, YOLO, tracking) is an opt-in extra.

## Python Requirement

- Python `>=3.10`

## Install pycasa

=== "From GitHub (recommended)"

    ```bash
    pip install "pycasa[io,detection,tracking,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
    ```

=== "From a local clone"

    ```bash
    git clone https://github.com/DFL-KamLab/pycasa.git
    cd pycasa
    pip install ".[io,detection,tracking,yolo]"
    ```

!!! tip "Full-workflow install"
    If you are just getting started, install all extras at once with `[io,detection,tracking,yolo]`. This gives you everything needed to run the examples in this documentation.

---

## Feature Extras Matrix

| Extra | What it enables | Install command |
|-------|----------------|-----------------|
| *(none)* | Core session model, basic utilities | `pip install "pycasa @ git+..."` |
| `io` | `load_default_data()` with HuggingFace Hub downloads | `pip install "pycasa[io] @ git+..."` |
| `detection` | `detect_moving_cells()` and `digital_washing()` (`scikit-image`, `scipy`) | `pip install "pycasa[detection] @ git+..."` |
| `tracking` | `tracking.sort()` (`scipy` assignment solver) | `pip install "pycasa[tracking] @ git+..."` |
| `yolo` | `detection.yolov5()` (`torch`, `torchvision`, `ultralytics`, `matplotlib`) | `pip install "pycasa[yolo] @ git+..."` |
| `io,detection,tracking,yolo` | Full pipeline (all of the above) | `pip install "pycasa[io,detection,tracking,yolo] @ git+..."` |

---

## YOLOv5 Source Checkout

For standard `.pt` checkpoints (non-TorchScript), pycasa needs access to a local YOLOv5 source tree. Clone it alongside pycasa:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

pycasa looks for the checkout in two places:

1. `../yolov5` relative to the pycasa project root.
2. The path in the `PYCASA_YOLOV5_REPO` environment variable (if set).

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PYCASA_DATA` | Cache root for `load_default_data()` downloads | `~/.pycasa_data` |
| `PYCASA_PROJECT_ROOT` | Explicit project-root override for weight and path resolution | Auto-detected |
| `PYCASA_YOLOV5_REPO` | Path to a local YOLOv5 source checkout | `../yolov5` (relative) |

=== "Linux / macOS"

    ```bash
    export PYCASA_DATA="$HOME/.pycasa_data"
    export PYCASA_YOLOV5_REPO="/path/to/yolov5"
    ```

=== "Windows PowerShell"

    ```powershell
    $env:PYCASA_DATA="$HOME\.pycasa_data"
    $env:PYCASA_YOLOV5_REPO="C:\path\to\yolov5"
    ```

---

## Installation Verification

Run the following to confirm your setup is working end-to-end:

```python
import pycasa as pc

self = pc.io.load_default_data(verbose=False)
self.preprocessing.binarization.otsu(show_progress=False, verbose=False)
self.info()
```

`self.info()` prints a concise session summary showing which video arrays, detections, tracks, and metadata are present. A successful output looks like:

```
Casa Session
  video  : original_video (100, H, W, 3) | grayscale_video | binary_video
  detections : (none)
  tracks     : (none)
  meta       : sampling_rate=50.0 | frame_range=[0, 99]
```

If this runs without import or runtime errors, your installation is ready for all standard workflows.

!!! note "First-run downloads"
    `load_default_data()` downloads a subset of the HC004 dataset from HuggingFace Hub on first run and caches it locally. Subsequent calls use the cache and require no network access.
