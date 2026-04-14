# Setup & Requirements

This page covers regular `pip` installation for `pycasa` and explains which optional dependencies to install based on your workflow.

## Python Requirement

- Python `>=3.10`

## Install pycasa

Install directly from GitHub:

```bash
pip install "git+https://github.com/DFL-KamLab/pycasa.git"
```

Install from a local clone:

```bash
git clone https://github.com/DFL-KamLab/pycasa.git
cd pycasa
pip install .
```

## Feature Extras Matrix

Install only what you need:

- Base only (core loading + core utilities):

```bash
pip install "git+https://github.com/DFL-KamLab/pycasa.git"
```

- I/O extras (`huggingface_hub` for `load_default_data` downloads):

```bash
pip install "pycasa[io] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

- Detection extras (`scikit-image`, `scipy`):

```bash
pip install "pycasa[detection] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

- Tracking extras (`scipy`):

```bash
pip install "pycasa[tracking] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

- YOLO extras (`torch`, `torchvision`, `ultralytics`, and related runtime libraries):

```bash
pip install "pycasa[yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

- Full workflow extras:

```bash
pip install "pycasa[io,detection,tracking,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## Requirement Notes

- `pycasa.io.load_default_data()` uses `huggingface_hub` and downloads a default dataset subset.
- `self.detection.yolov5()` may download managed weights into `yolov5-weights/`.
- For standard `.pt` checkpoints, pycasa may need a local YOLOv5 source checkout:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

The repository is expected either:

- next to `pycasa` as `../yolov5`, or
- at a custom path via `PYCASA_YOLOV5_REPO`.

## Environment Variables

- `PYCASA_DATA`:
  default-data cache root for `load_default_data`.
- `PYCASA_PROJECT_ROOT`:
  explicit project-root override for weight/path resolution.
- `PYCASA_YOLOV5_REPO`:
  path to a local YOLOv5 checkout.

Example:

```bash
# Linux/macOS
export PYCASA_DATA="$HOME/.pycasa_data"
export PYCASA_YOLOV5_REPO="/path/to/yolov5"
```

```powershell
# Windows PowerShell
$env:PYCASA_DATA="$HOME\\.pycasa_data"
$env:PYCASA_YOLOV5_REPO="C:\\path\\to\\yolov5"
```

## Installation Verification

Run:

```python
import pycasa as pc
self = pc.io.load_default_data(verbose=False)
self.preprocessing.binarization.otsu(show_progress=False, verbose=False)
self.info()
```

If this completes without import/runtime errors, your installation is ready for most workflows.
