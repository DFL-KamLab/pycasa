# Example: Default Data + Otsu + YOLO

This is the fastest way to get a realistic pycasa pipeline running. It uses the built-in HC004 default dataset — a publicly available semen analysis video subset — requires no local files, and exercises the most commonly used detection backend (YOLOv5).

## Install

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

The `yolo` extra pulls in `matplotlib`, which is required for visualization calls.

---

## Script

```python
import pycasa as pc

# Load the default HC004 dataset subset from HuggingFace Hub (cached after first run)
self = pc.io.load_default_data()

# Binarize the video using Otsu global thresholding
self.preprocessing.binarization.otsu()

# Run YOLOv5 inference on every frame (downloads managed weights on first run)
self.detection.yolov5()

# Plot a single frame with bounding-box detections overlaid
self.visualization.plot_frame(frame_index=5, show_detections=True)

# Print a session summary
self.info()
```

---

## Output preview

<div class="screenshot" markdown>

![Screenshot: plot_frame output showing YOLOv5 bounding-box detections overlaid on a semen analysis video frame](../assets/screenshots/yolov5-plot-frame.png)
*`plot_frame(frame_index=5, show_detections=True)` — YOLOv5 bounding boxes overlaid on frame 5 of the HC004 dataset.*

</div>

---

## Step-by-step breakdown

**`pc.io.load_default_data()`**

Downloads (or reads from the local cache at `~/.pycasa_data`) a subset of the HC004 microscopy video and its matching groundtruth annotation folder. The video is stored as a NumPy array under `casa["video"]["original_video"]` in BGR color order. Sampling rate and frame range are stored in `casa["meta"]`.

**`preprocessing.binarization.otsu()`**

Converts the original video to grayscale first, then applies Otsu's global threshold to produce a binary (0/255 uint8) video stored as `casa["video"]["binary_video"]`. Otsu works well for semen analysis videos because the sperm cells create a bimodal intensity distribution against the bright background.

**`detection.yolov5()`**

Downloads the default managed weights (`sys-casa_yolov5s.pt`, trained on CASA semen data) from HuggingFace if not already cached. Runs inference on each frame and stores normalized detections — `[class_id, norm_cx, norm_cy, norm_w, norm_h]` — in `casa["detections"]["yolov5"]`. Confidence threshold defaults to `0.15`.

**`visualization.plot_frame(frame_index=5, show_detections=True)`**

Opens a static matplotlib figure showing frame 5 of the original video with YOLOv5 bounding boxes overlaid. Useful for a quick visual sanity-check before running the full pipeline.

**`self.info()`**

Prints a compact summary of what the session holds: video arrays loaded, active detections, and key metadata fields.

---

## Otsu binary video

<div class="screenshot" markdown>

![Screenshot: side-by-side plot_frame showing original and binarized video frames](../assets/screenshots/otsu-binarized.png)
*`plot_frame(image_type="original+binarized")` — original frame (left) alongside the Otsu binary output (right).*

</div>

---

## Inspecting the session

```python
# What metadata was recorded?
meta = self.get_meta()
print(meta["sampling_rate"])       # frames per second
print(meta["last_detection"])      # method used, frames processed, detection counts

# How many detections are there per frame?
detections = self.get_detections()
frame_0_dets = detections[0]       # list of [class_id, cx, cy, w, h] rows for frame 0
print(f"Frame 0: {len(frame_0_dets)} detections")

# Compare against groundtruth
gt = self.get_groundtruth()
print(f"Groundtruth frames loaded: {len(gt)}")
```

---

## Notes

- First run may trigger downloads for both the dataset and the YOLO weights. Subsequent runs use the cache and complete offline.
- Calling a second detection method after `yolov5()` will overwrite the active predicted detections and emit a warning.
- The `plot_frame` call requires `matplotlib`. If you installed with `[io,yolo]`, it is already included.

---

## What to try next

- Add `self.tracking.sort()` after detection to get trajectories — see [Detection + SORT Tracking](detection-tracking.md).
- Add `self.assessment.classification()` to score YOLOv5 against the bundled groundtruth — see [Motility + Assessment](motility-assessment.md).
- Swap in your own video — see [Load Custom Video](custom-video.md).
