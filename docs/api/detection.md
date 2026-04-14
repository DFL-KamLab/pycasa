# API: Detection

Purpose:
Detection is the localization stage of the pipeline. It identifies candidate cells/objects per frame and writes one active predicted detection source into the session for tracking and assessment.

## Public Methods In This Section

- `self.detection.detect_moving_cells(...)`
- `self.detection.digital_washing(...)`
- `self.detection.yolov5(...)`

## `detect_moving_cells`

Supports methods such as `cv-gmg`, `cv-mog`, `cv-mog2`, and `gm`.

```python
self.detection.detect_moving_cells(method="cv-mog2")
```

## `digital_washing`

A dedicated workflow with configurable thresholds and filtering parameters.

```python
self.detection.digital_washing(show_progress=True)
```

## `yolov5`

Runs YOLOv5 with managed or custom weights.

```python
self.detection.yolov5(
    weights="sys-casa_yolov5s.pt",
    conf=0.15,
    download=True,
)
```

Important runtime notes:

- Managed weights may be downloaded to `yolov5-weights/`.
- For standard `.pt` fallback loading, a local YOLOv5 clone may be required.
- `PYCASA_YOLOV5_REPO` can point to that checkout.

## Output Behavior

- Writes detections under `casa["detections"][<method>]`.
- Keeps only one active predicted detection method.
- Updates `casa["meta"]["last_detection"]`.
