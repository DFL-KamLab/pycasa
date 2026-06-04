# pycasa

**Version 0.0.1**

`pycasa` is a Python toolkit for computer-assisted semen analysis workflows. It supports loading microscopy videos, preprocessing frames, running detection and tracking, computing motility metrics, assessing predictions against groundtruth, and visualizing results.

## Why pycasa

- Fluent session API centered on a single `Casa` object.
- Modular namespaces for each workflow stage.
- Support for both default reference data and custom video pipelines.

## Quick Install

```bash
pip install "git+https://github.com/DFL-KamLab/pycasa.git"
```

For the full default-data + YOLO example:

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## Starter Example

```python
import pycasa as pc

self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolo()
self.info()
```

## Package Structure

```text
pycasa/
  casa/            # Fluent wrappers (self.io, self.detection, ...)
  io/              # Video/default-data loading implementations
  preprocessing/   # Grayscale, normalization, binarization implementations
  detection/       # Detection backends (moving-cells, digital washing, Urbano, YOLO v5/v26)
  tracking/        # Tracking backends (SORT, DeepSORT, JPDAF)
  motility/        # Motility parameter computation
  assessment/      # Prediction-vs-groundtruth evaluation
  visualization/   # Plotting and interactive analysis
  _core/           # Session schema/validation primitives
  utils/           # Shared helper utilities
```

## Documentation Website

For detailed setup, examples, and API references, use the website:

- Home: https://dfl-kamlab.github.io/pycasa/
- Setup & Requirements: https://dfl-kamlab.github.io/pycasa/getting-started/setup/
- Examples: https://dfl-kamlab.github.io/pycasa/examples/default-data-otsu-yolo/
- API Guide: https://dfl-kamlab.github.io/pycasa/api/casa-session/
