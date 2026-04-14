# Quickstart

This quickstart gives you a short, working path from loading data to running detection.

## Install

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## Minimal End-to-End Script

```python
import pycasa as pc

self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolov5()
self.info()
```

## What This Does

- Loads default data into a `Casa` session.
- Builds a binary video using Otsu thresholding.
- Runs YOLOv5 inference and stores detections in session state.
- Prints a concise session summary.

## Next Steps

- For custom inputs, see [Load Custom Video](../examples/custom-video.md).
- For trajectories, see [Detection + SORT Tracking](../examples/detection-tracking.md).
- For evaluation, see [Motility + Assessment](../examples/motility-assessment.md).
