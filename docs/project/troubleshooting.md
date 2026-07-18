# Troubleshooting

Common setup and runtime issues when working with pycasa.

## Import Errors for Optional Dependencies

Symptom:

- missing module errors during `load_default_data`, detection, or tracking.

Fix:

- install matching extras for your workflow:

```bash
pip install "pycasa[io,detection,tracking,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

## `load_default_data` Cannot Download

Symptom:

- errors resolving default dataset files.

Fix:

- confirm network access and Hugging Face availability.
- set `PYCASA_DATA` to a writable location.
- retry with `download=True` or `force_download=True`.

## YOLO Weight / Repository Errors

Symptom:

- `FileNotFoundError` about managed/custom weights or local YOLOv5 source checkout.

Fix:

1. Confirm requested weight name is valid (for managed weights).
2. Ensure weights are available locally or allow download.
3. If required, clone YOLOv5 and set:

```powershell
$env:PYCASA_YOLOV5_REPO="C:\\path\\to\\yolov5"
```

## No Detections or Empty Tracks

Symptom:

- detection output is empty, or tracking returns no trajectories.

Fix:

- visualize sample frames to inspect contrast/object visibility.
- adjust detection thresholds/method.
- verify frame ranges and video quality.
- rerun tracking after changing detection backend.

## Assessment Output Missing

Symptom:

- assessment is empty or unavailable.

Fix (detection assessment):

- ensure both predicted detections and groundtruth **detections** are present before calling:

```python
self.assessment.evaluate_detections()
```

Fix (track assessment):

- `evaluate_tracks()` needs **at least two track sets**. Load imported groundtruth tracks with `load_video(..., groundtruth_tracks_path=...)` and run at least one tracker (e.g. `self.tracking.sort()`); if fewer than two sets are present it is skipped with a warning.
- it uses the optional `motmetrics` dependency, installed on demand the first time you call it.

```python
self.assessment.evaluate_tracks()
```
