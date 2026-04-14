# Example: Detection + SORT Tracking

This workflow creates trajectories from detections.

## Install

```bash
pip install "pycasa[io,detection,tracking] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## Script

```python
import pycasa as pc

self = pc.io.load_default_data()
self.detection.detect_moving_cells(method="cv-mog2")
self.tracking.sort(show_progress=True, verbose=True)
self.info()
```

## Why This Pairing

- `detect_moving_cells` gives a lightweight detection backend.
- `tracking.sort` converts detections into frame-linked trajectories.

## Validation Checks

- Confirm detections exist with `self.get_detections()`.
- Confirm sort output exists with `self.get_tracks(backend="sort")`.
