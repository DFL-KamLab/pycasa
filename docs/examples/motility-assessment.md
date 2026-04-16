# Example: Motility + Assessment

This example shows the full analytical half of the pycasa pipeline: computing standard CASA motility parameters from SORT trajectories, and scoring the detector against groundtruth annotations.

## Install

```bash
pip install "pycasa[io,detection,tracking] @ git+https://github.com/DFL-KamLab/pycasa.git"
```

---

## Complete script

```python
import pycasa as pc

# 1. Load default data (includes groundtruth annotations)
self = pc.io.load_default_data()

# 2. Detect and track
self.detection.detect_moving_cells(method="cv-mog2")
self.tracking.sort()

# 3. Set pixel-to-micron calibration before motility computation
self.set_um_per_px(0.24)   # µm per pixel — adjust to match your microscope setup

# 4. Compute standard CASA motility metrics
self.motility.standard_motility_parameters(
    window_size=10,    # trajectory points per sliding window
    overlap=0.2,       # fraction of window that overlaps with the previous window
)

# 5. Score predicted detections against groundtruth
self.assessment.classification(match_min_distance_pixel=20)

self.info()
```

---

## Output preview

<div class="screenshot" markdown>

![Screenshot: motility radar chart showing aggregate mean values for VCL, VSL, VAP, LIN, ALH, WOB, STR, MAD](../assets/screenshots/motility-radar.png)
*`motility_radar()` — spider chart of aggregate motility metric means across all active SORT tracks.*

</div>

<div class="screenshot" markdown>

![Screenshot: interactive motility calculator panel showing track selector and 8 metric history tiles](../assets/screenshots/interactive-motility-calculator.png)
*`interactive_motility_calculator()` — per-track window explorer with VCL, VSL, VAP, LIN, ALH, WOB, STR, and MAD history tiles.*

</div>

---

## Motility metrics reference

`standard_motility_parameters` computes these eight metrics per sliding window over each track:

| Metric | Full Name | Unit | What it measures |
|--------|-----------|------|-----------------|
| **VCL** | Curvilinear Velocity | µm/s | Speed along the actual curvilinear path. |
| **VSL** | Straight-Line Velocity | µm/s | Net displacement speed (first → last point). |
| **VAP** | Average Path Velocity | µm/s | Speed along the smoothed mean path. |
| **LIN** | Linearity | — | `VSL / VCL`. Near 1 = highly linear motion. |
| **ALH** | Amplitude of Lateral Head Displacement | µm | Half-range of lateral oscillations around the mean path. |
| **WOB** | Wobble | — | `VAP / VCL`. Deviation from the mean path. |
| **STR** | Straightness | — | `VSL / VAP`. Straightness of the mean path. |
| **MAD** | Mean Angular Displacement | degrees | Average turning angle per frame step. |

!!! note "Unit conversion"
    Velocity metrics (VCL, VSL, VAP) and ALH are reported in **µm/s** and **µm** when `um_per_px` is set and `conversion_required=True` (the default). Without calibration they remain in pixel units. Set calibration with `self.set_um_per_px(value)` before calling `standard_motility_parameters()`.

---

## Reading the motility output

```python
motility = self.get_motility()

# Output is keyed by detection source → track_id → metric → list of per-window values
source = "moving_cells"   # or "yolov5", "groundtruth", etc.
track_motility = motility["standard_motility_parameters"][source]

# Inspect one track
for track_id, params in list(track_motility.items())[:2]:
    vcl_windows = params["VCL"]   # one float per sliding window
    vsl_windows = params["VSL"]
    print(f"Track {track_id}: {len(vcl_windows)} windows, "
          f"mean VCL={sum(vcl_windows)/len(vcl_windows):.1f} µm/s")
    print(f"  frame_ranges: {params['frame_ranges']}")
```

The output structure:

```python
{
    "standard_motility_parameters": {
        "moving_cells": {
            42: {
                "VCL": [38.2, 40.1, 37.8],   # one value per window
                "VSL": [22.4, 24.0, 21.9],
                "VAP": [30.1, 31.5, 29.8],
                "LIN": [0.59, 0.60, 0.58],
                "ALH": [1.8, 1.9, 1.7],
                "WOB": [0.79, 0.79, 0.80],
                "STR": [0.74, 0.76, 0.73],
                "MAD": [18.3, 17.9, 19.1],
                "frame_ranges": "0-9, 8-17, 16-25",
            },
            ...
        }
    }
}
```

---

## Reading the assessment output

!!! note "Groundtruth requirement"
    Assessment compares predicted detections against groundtruth. The default dataset includes groundtruth annotations. For custom videos, pass `groundtruth_path=...` to `load_video()`.

```python
assessment = self.get_assessment()
clf = assessment["classification"]

print(f"True positives  : {clf['tp']}")
print(f"False positives : {clf['fp']}")
print(f"False negatives : {clf['fn']}")
print(f"Precision       : {clf['precision']:.1f}%")
print(f"Recall          : {clf['recall']:.1f}%")
print(f"F1 score        : {clf['F1']:.1f}%")
print(f"Evaluated frames: {clf['evaluated_frames']}")
```

| Metric | Definition |
|--------|-----------|
| `tp` | Predicted detections matched to a groundtruth within the distance threshold. |
| `fp` | Predicted detections with no close groundtruth match. |
| `fn` | Groundtruth detections with no close predicted match. |
| `precision` | `tp / (tp + fp)` as a percentage. |
| `recall` | `tp / (tp + fn)` as a percentage. |
| `F1` | Harmonic mean of precision and recall as a percentage. |
| `evaluated_frames` | Frames where both predicted and groundtruth detections were present. |

---

## What to try next

- [Visualization API](../api/visualization.md) — use `motility_radar()` and `motility_density_scatter()` to plot aggregate metric summaries.
- [Interactive motility calculator](../api/visualization.md#selfvisualizationinteractive_motility_calculator) — explore per-track, per-window metric histories in an interactive panel.
- [Assessment API](../api/assessment.md) — full parameter reference for `classification()`.
