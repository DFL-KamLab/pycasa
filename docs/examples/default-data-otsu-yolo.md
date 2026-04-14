# Example: Default Data + Otsu + YOLO

This is the fastest way to run a realistic pycasa pipeline.

## Install

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

This install path includes `matplotlib` via the `yolo` extra, which is required by `self.visualization.plot_frame(...)`.

## Script

```python
import pycasa as pc

self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolov5()
self.visualization.plot_frame(frame_index=5, show_detections=True)
self.info()
```

## Outputs

- Loaded video and metadata under `self.get_video()` and `self.get_meta()`.
- Binary frames stored as `binary_video` in session video state.
- YOLO detections stored under `self.get_detections()`.
- Visualization metadata recorded in `self.get_meta()["last_visualization"]`.

## Notes

- First run can trigger downloads for default data and managed YOLO weights.
- If detections already exist, the active predicted method is replaced by `yolov5`.
