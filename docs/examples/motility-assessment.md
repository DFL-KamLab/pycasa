# Example: Motility + Assessment

This example computes motility metrics and compares predicted detections against groundtruth.

## Install

```bash
pip install "pycasa[io,detection,tracking] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## Script

```python
import pycasa as pc

self = pc.io.load_default_data()
self.detection.detect_moving_cells(method="cv-mog2")
self.tracking.sort()
self.motility.standard_motility_parameters()
self.assessment.classification()
self.info()
```

## What You Get

- Motility summaries in `self.get_motility()`.
- Assessment/evaluation outputs in `self.get_assessment()`.

## Notes

- Assessment requires both predicted detections and groundtruth.
- Default data includes a compatible groundtruth directory.
