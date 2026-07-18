# Example: Load Custom Video

Use this workflow when you have your own microscopy video file and want to run pycasa on it — with optional groundtruth annotations and custom frame range or calibration settings.

## Install

```bash
pip install "pycasa[detection] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

Add `[yolo]` if you plan to use YOLOv5 for detection.

---

## Supported formats

!!! tip "Supported video formats"
    pycasa reads video files via OpenCV. The following container formats are supported:

    `.avi` · `.mp4` · `.mov` · `.mkv` · `.flv` · `.wmv`

---

## Groundtruth format

Groundtruth **detections** and groundtruth **tracks** are imported independently, via two separate arguments.

### Groundtruth detections — `groundtruth_detections_path`

A folder of one plain-text file per frame. Each file contains one detection row per line in normalized YOLO format:

```
<label> <norm_cx> <norm_cy> <norm_w> <norm_h>
```

The frame index for each file is taken from the **last integer group** in its file name, so `frame-170.txt`, `82_frame_170.txt`, etc. all work (no fixed naming or zero-padding required). Loaded under `casa["detections"]["groundtruth"]` and used by `assessment.classification()`.

### Groundtruth tracks — `groundtruth_tracks_path`

A folder of per-frame files in the same YOLO layout but **prefixed with a persistent track id**, where the same id recurring across frames defines an identity:

```
<track_id> <label> <norm_cx> <norm_cy> <norm_w> <norm_h>
```

Loaded under `casa["tracks"]["groundtruth_tracks"]` as `{track_id: {frame: [center_x, center_y]}}` (centers only, for now), with ids normalized to `t0, t1, ...`. Access with `self.get_groundtruth_tracks()`. This imported truth coexists with any tracker output and appears as its own source in the timelapse, `info()`, motility, and the motility visualizations.

All coordinates are normalized to `[0, 1]` relative to frame width/height. Frames with no detections can be represented by an empty file or omitted entirely.

---

## Output preview

<div class="screenshot" markdown>

![Screenshot: plot_frame output on a custom-loaded video with moving-cell detections overlaid](../assets/screenshots/custom-video-detections.png)
*`plot_frame(show_detections=True)` — detections from `detect_moving_cells()` overlaid on a custom video frame.*

</div>

---

## Script

```python
import pycasa as pc

# Load a custom video with optional groundtruth and calibration
self = pc.io.load_video(
    video_path="path/to/video.avi",
    groundtruth_detections_path="path/to/groundtruth_detections_folder",  # optional
    groundtruth_tracks_path="path/to/groundtruth_tracks_folder",          # optional
    initial_frame=0,
    final_frame=300,                # None = read to end of file
    sampling_rate=50.0,             # frames per second (optional override)
    um_per_px=0.25,                 # pixel-to-micron calibration (optional)
    magnification="20x",            # metadata tag only, no effect on computation
)

# Convert to grayscale
self.preprocessing.grayscale(overwrite=False)

# Binarize — try Otsu or any other method
self.preprocessing.binarization.otsu()

# Detect with the lightweight background-subtraction backend
self.detection.detect_moving_cells(method="cv-mog2")

# Print session summary
self.info()
```

---

## Key `load_video` parameters

| Parameter | Type | Default | When to use |
|-----------|------|---------|-------------|
| `video_path` | `str` | *(required)* | Path to the input video file. |
| `groundtruth_detections_path` | `str \| None` | `None` | Folder of per-frame `.txt` detection files (`label cx cy w h`). Enables `assessment.classification()`. |
| `groundtruth_tracks_path` | `str \| None` | `None` | Folder of per-frame `.txt` track files (`track_id label cx cy w h`). Imported truth tracks; access via `self.get_groundtruth_tracks()`. |
| `initial_frame` / `final_frame` | `int \| None` | `0` / `None` | Clip to a sub-range of the video. Frames outside the range are not loaded. |
| `sampling_rate` | `float \| None` | `None` | Override the FPS stored in the video container. Used by motility metric computation. |
| `um_per_px` | `float \| None` | `None` | Pixel-to-micron calibration factor. Required for micron-unit motility output. Can also be set later with `self.set_um_per_px(value)`. |

---

## Tips

- Use `final_frame=None` to read until the end of the video.
- `overwrite=False` on preprocessing calls preserves the previous result if the step was already run.
- If `sampling_rate` is not known, pycasa reads it from the container. Verify with `self.get_meta()["sampling_rate"]` after load.
- Set `um_per_px` at load time or later with `self.set_um_per_px(0.24)` before running motility.

---

## What to try next

- [Detection + SORT Tracking](detection-tracking.md) — compare detection backends and build trajectories.
- [Motility + Assessment](motility-assessment.md) — compute CASA metrics and evaluate against groundtruth.
