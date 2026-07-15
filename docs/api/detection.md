# API: Detection

Detection is the localization stage of the pipeline. It identifies candidate cells/objects per frame and writes one active predicted detection source into the session for tracking and assessment.

**Single-result policy:** only one predicted detection method is kept at a time. Running a second detector overwrites the first and emits a warning.

All detection methods:
- Store detections in the session, accessible via `self.get_detections()`.
- Return the same `Casa` instance for fluent chaining.

Detection rows follow the legacy normalized format:

```
[label, norm_centroid_x, norm_centroid_y, norm_bbox_width, norm_bbox_height]
```

All coordinates are normalized to `[0, 1]` relative to frame width/height.

!!! note "This page is generated from the code"
    Signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

::: pycasa.casa.detection.detection_wrapper._SessionDetectionNamespace.detect_moving_cells

---

::: pycasa.casa.detection.detection_wrapper._SessionDetectionNamespace.digital_washing

---

::: pycasa.casa.detection.detection_wrapper._SessionDetectionNamespace.urbano_detection

---

::: pycasa.casa.detection.detection_wrapper._SessionDetectionNamespace.yolo
