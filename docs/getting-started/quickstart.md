# Quickstart

This quickstart walks you from a fresh install to a working detection pipeline in under five minutes. By the end you will have loaded the default semen analysis dataset, binarized the video, run YOLO detection, and inspected the results.

## Install

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

!!! note "First-run downloads"
    On first run, `load_default_data()` downloads a small subset of the HC004 dataset from HuggingFace Hub (~a few MB) and `yolo()` downloads the default managed weights. Both are cached locally and skipped on subsequent runs.

---

## Minimal pipeline

```python
import pycasa as pc

self = pc.io.load_default_data()          # load HC004 subset; returns a Casa session
self.preprocessing.binarization.otsu()    # build a binary video via Otsu thresholding
self.detection.yolo()                     # run YOLO inference on every frame (YOLO26 by default)
self.info()                               # print a concise session summary
```

**What each line does:**

| Line | What happens |
|------|-------------|
| `load_default_data()` | Downloads (or reads from cache) the HC004 video and groundtruth annotations; stores frames in `casa["video"]["original_video"]`. |
| `binarization.otsu()` | Converts the video to grayscale then applies Otsu global thresholding; stores the result as `binary_video`. |
| `detection.yolo()` | Downloads managed weights if needed and runs inference frame-by-frame; stores normalized detections in `casa["detections"]["yolo26"]` by default (or `["yolov5"]` when `yolo_model="yolov5"`). |
| `self.info()` | Prints counts of loaded arrays, detected objects, and metadata keys so you can confirm each stage ran. |

---

## Full pipeline

Once you are comfortable with the minimal script, here is the complete seven-step pipeline:

```python
import pycasa as pc

# --- Load ---
self = pc.io.load_default_data()

# --- Preprocess ---
self.preprocessing.grayscale()
self.preprocessing.binarization.otsu()

# --- Detect ---
self.detection.yolo()

# --- Track ---
self.tracking.sort(max_age=25, min_hits=3, iou_threshold=0.1)

# --- Motility ---
# um_per_px is auto-set to 0.24 by load_default_data(); call self.set_um_per_px(...) for custom videos
self.motility.standard_motility_parameters()

# --- Assess ---
self.assessment.evaluate_detections(match_min_distance_pixel=20)   # detections vs GT detections
# self.assessment.evaluate_tracks()   # tracks vs GT tracks (MOTA/IDF1); needs groundtruth_tracks_path

# --- Visualize ---
self.visualization.timelapse(
    video_type="original",
    show_detections=True,
    show_tracks=True,
)
```

---

## Inspecting results

After running the pipeline, pull data out of the session with the getter methods:

```python
meta       = self.get_meta()        # sampling rate, frame range, last_* fields
detections = self.get_detections()  # active predicted detection dict
tracks     = self.get_tracks(backend="sort")  # SORT trajectories by source
motility   = self.get_motility()    # standard motility parameter dicts
assessment = self.get_assessment()  # ["detection"]: tp/fp/fn/precision/recall/F1; ["tracking"]: MOTA/IDF1 pairs
```

---

## Next steps

| Guide | What it adds |
|-------|-------------|
| [Load Custom Video](../examples/custom-video.md) | Load your own AVI/MP4 file with custom frame ranges and calibration. |
| [Detection + SORT Tracking](../examples/detection-tracking.md) | Compare all three detection backends and tune SORT parameters. |
| [Motility + Assessment](../examples/motility-assessment.md) | Compute micron-calibrated CASA metrics and evaluate detector performance. |
| [Default Data + Otsu + YOLO](../examples/default-data-otsu-yolo.md) | Deep-dive into this exact pipeline with step-by-step annotations. |
