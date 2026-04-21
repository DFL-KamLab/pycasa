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

## Public Methods In This Section

- `self.detection.detect_moving_cells(...)`
- `self.detection.digital_washing(...)`
- `self.detection.yolov5(...)`

---

## `self.detection.detect_moving_cells(...)`

Legacy-parity moving-cell detection using background subtraction or Gaussian mixture modeling.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"cv-gmg"` | Background extraction backend. One of `"cv-gmg"`, `"cv-mog"`, `"cv-mog2"`, or `"gm"`. See method details below. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress warnings. |
| `**kwargs` | | | Optional per-method tuning parameters (see below). |

**Keyword Arguments (`**kwargs`)**

| Parameter | Type | Default | Applies to | Description |
|-----------|------|---------|------------|-------------|
| `number_training_frames` | `int` | `20` | all methods | Number of warm-up frames used to initialize background model. Detections are skipped for these frames. |
| `blob_min_pixel_area` | `int` | `20` | all methods | Minimum connected-component area (in pixels) required to keep a blob as a detection. Smaller blobs are discarded. |
| `threshold` | `float` | `3` | `gm` only | Sigma threshold for the Gaussian mixture foreground/background separation. Higher values are more conservative. |
| `med_filter_size` | `int` | `3` | `gm` only | Kernel size for median filtering applied before Gaussian mixture processing. |
| `disk_size` | `int` | `6` | `gm` only | Radius of the disk structuring element used for morphological dilation/erosion post-processing. |

**Method Details**

| Method | Backend | Notes |
|--------|---------|-------|
| `"cv-gmg"` | `cv2.bgsegm.createBackgroundSubtractorGMG` | Requires `opencv-contrib-python`. Uses GMG (Godbehere-Matsukawa-Goldberg) subtractor. |
| `"cv-mog"` | `cv2.bgsegm.createBackgroundSubtractorMOG` | Requires `opencv-contrib-python`. Uses MOG mixture-of-Gaussians subtractor. |
| `"cv-mog2"` | `cv2.createBackgroundSubtractorMOG2` | Available in standard `opencv-python`. Uses MOG2 subtractor. |
| `"gm"` | Custom Gaussian mixture | Pure NumPy/SciPy implementation with median filtering and morphological post-processing. Supports `threshold`, `med_filter_size`, `disk_size` kwargs. |

**Output**

Detections are stored in the session and accessible via `self.get_detections()`. An intermediate per-frame binary foreground mask is also retained internally.

**Returns**

`Casa` — the same session instance.

**Raises**

- `ImportError` — if `"cv-gmg"` or `"cv-mog"` is requested and `opencv-contrib-python` is not installed.
- `ValueError` — if an unsupported method string is passed.

**Example**

```python
# Using OpenCV MOG2 background subtractor
self.detection.detect_moving_cells(method="cv-mog2")

# Using Gaussian mixture with custom parameters
self.detection.detect_moving_cells(
    method="gm",
    number_training_frames=30,
    threshold=2.5,
    disk_size=4,
)
```

---

## `self.detection.digital_washing(...)`

Digital Washing detection algorithm combining Gaussian-mixture motion extraction, log-based binarization, and Hu-moment shape classification.

The algorithm operates in stages:
1. Gaussian-mixture motion extraction (same core as `gm` method above).
2. Log-filter-based binarization of the grayscale video.
3. Motion/background separation to remove static blobs.
4. Feature extraction (shape, Hu moments) per blob.
5. Local detector fusion to produce final candidate detections.
6. Border exclusion by `border_margin_px`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `motion_threshold` | `float` | `3.0` | Sigma threshold for Gaussian-mixture motion extraction. Higher values reduce sensitivity to motion. |
| `number_training_frames` | `int` | `20` | Warm-up frame count used by the motion and background separation stages. Detections are not generated for these frames. |
| `blob_min_pixel_area` | `int` | `20` | Minimum connected-component area (in pixels) to keep during feature extraction. |
| `k_val` | `float` | `1.7` | Standard-deviation multiplier used in the local detector fusion rules. Controls how tightly candidates must cluster around the mean feature distribution. |
| `border_margin_px` | `int` | `20` | Exclusion margin in pixels from the frame border. Detections within this margin are discarded. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar during iterative processing stages. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress overwrite warnings. |

**Output**

Detections are stored in the session and accessible via `self.get_detections()`. Intermediate binary videos are also retained internally:

| Key | Description |
|-----|-------------|
| `digital_washing_motion_video` | Binary motion mask (uint8, 0/255). |
| `digital_washing_binarized_video` | Log-method binarization result (uint8, 0/255). |
| `digital_washing_background_video` | Background blobs after motion removal (uint8, 0/255). |

**Returns**

`Casa` — the same session instance.

**Raises**

- `TypeError` — if `casa["video"]["original_video"]` is not a numpy array.
- `ValueError` — if `original_video` has unsupported dimensions.

**Example**

```python
self.detection.digital_washing(
    motion_threshold=3.0,
    number_training_frames=20,
    blob_min_pixel_area=20,
    k_val=1.7,
    border_margin_px=20,
)
```

---

## `self.detection.yolov5(...)`

Run YOLOv5 object detection on the in-memory video using managed weights downloaded from HuggingFace or a custom local checkpoint.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `str` | `"sys-casa_yolov5s.pt"` | Weight selector. Pass a managed weight name (see table below) for automatic download/caching, or a path to a local `.pt` file. |
| `conf` | `float` | `0.15` | Detection confidence threshold. Only detections with confidence ≥ this value are kept. |
| `download` | `bool` | `True` | Automatically download missing managed weights from the dataset repository. |
| `force_download` | `bool` | `False` | Re-download managed weights even when a cached local file exists. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar during per-frame inference. |
| `verbose` | `bool` | `True` | Print start/end summaries and confidence statistics. Does not suppress warnings. |

**Managed Weight Names**

Two weight sets are available. All names follow the pattern `<set>_<model>.pt`:

| Set | Models | Description |
|-----|--------|-------------|
| `sys-casa` | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` | Weights trained on the CASA semen analysis dataset. |
| `sys-opt` | `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x` | Optimized variant weights. |

Full list: `sys-casa_yolov5n.pt`, `sys-casa_yolov5s.pt`, `sys-casa_yolov5m.pt`, `sys-casa_yolov5l.pt`, `sys-casa_yolov5x.pt`, `sys-opt_yolov5n.pt`, `sys-opt_yolov5s.pt`, `sys-opt_yolov5m.pt`, `sys-opt_yolov5l.pt`, `sys-opt_yolov5x.pt`

**Output**

Detections are stored in the session as frame-indexed normalized detection rows and are accessible via `self.get_detections()`.

Detection output format: `[class_id, norm_center_x, norm_center_y, norm_width, norm_height]`

**Returns**

`Casa` — the same session instance.

**Raises**

- `ImportError` — if optional YOLO runtime dependencies (`torch`, `torchvision`, etc.) are not installed.
- `FileNotFoundError` — if a custom weight path does not exist, or if managed weights are not cached and `download=False`.
- `TypeError` — if `original_video` exists but is not a numpy array.
- `ValueError` — if `weights` is an empty string.

**Notes**

- Managed weights are downloaded to `yolov5-weights/` relative to the project root.
- For standard `.pt` checkpoints (non-TorchScript), a local YOLOv5 source checkout is required. Set the `PYCASA_YOLOV5_REPO` environment variable to point to it, or clone YOLOv5 as a sibling directory (`../yolov5`).
- The `delete_temp` parameter is a legacy compatibility flag and has no effect in the current implementation.

**Example**

```python
# Default managed weights
self.detection.yolov5()

# Larger model with lower confidence threshold
self.detection.yolov5(weights="sys-opt_yolov5m.pt", conf=0.10)

# Custom local checkpoint
self.detection.yolov5(weights="/path/to/my_weights.pt", download=False)
```

## Output Behavior

All detection methods store their output in the session. Only one predicted detection source is kept at a time — running a second method overwrites the previous result and emits a yellow warning. Retrieve detections with `self.get_detections()`.

To inspect detections after running:

```python
detections = self.get_detections()
```
