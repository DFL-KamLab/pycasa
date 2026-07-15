# Casa Session Model

`Casa` is the central session object in pycasa. It carries video data, detections, tracks, motility metrics, and evaluation outputs through a fluent API.

## Public Interface In This Section

- Constructor: `pc.io.load_video(...)`, `pc.io.load_default_data(...)` return `Casa`.
- Session namespaces: `self.io`, `self.preprocessing`, `self.detection`, `self.tracking`, `self.motility`, `self.assessment`, `self.visualization`.
- Session inspection/state methods:
  `info`, `copy`, `set_um_per_px`, `get_casa`, `get_meta`, `get_video`, `get_detections`,
  `get_groundtruth`, `get_tracks`, `get_motility`, `get_assessment`.

## Core Pattern

```python
import pycasa as pc

self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolo()
```

Each method updates internal session state and returns the same `Casa` object for chaining.

## Session Namespaces

- `self.io` — session creation and data-ingestion layer. Use this first to load video frames, optional groundtruth labels, sampling rate, and calibration metadata.
- `self.preprocessing` — frame transformation layer. Use this to derive grayscale/normalized/binary variants before running detector and tracker stages.
- `self.detection` — object localization layer. Produces one active predicted detection set at a time (e.g., moving-cells, digital washing, Urbano, or YOLO — YOLOv5 / YOLO26).
- `self.tracking` — trajectory-building layer. Converts detections over time into track sequences.
- `self.motility` — quantitative analysis layer. Computes motility metrics over track windows.
- `self.assessment` — evaluation layer. Compares predicted detections with groundtruth detections.
- `self.visualization` — inspection/reporting layer. Produces frame plots, timelapse playback, and motility-oriented visual summaries.

Each namespace has its own API page (see the nav). The concrete session-level state and getter methods are documented below.

!!! note "This section is generated from the code"
    The method signatures and descriptions below are rendered directly from the
    `Casa` class docstrings, so they always match the installed version.

## Session Methods

::: pycasa.casa.casa.Casa
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - info
        - copy
        - set_um_per_px
        - get_casa
        - get_meta
        - get_video
        - get_detections
        - get_groundtruth
        - get_tracks
        - get_motility
        - get_assessment

## Public API and Helper Rules

- Use namespace methods as the stable public API (`self.detection.yolo()`, etc.).
- Treat underscore-prefixed modules/functions (e.g., `pycasa._core`, helpers in `pycasa.utils`) as internal implementation details.
- Fluent wrappers in `pycasa.casa.*` orchestrate state updates; algorithm implementations live in `pycasa.<namespace>`.
- Session provenance is tracked through `meta["last_*"]` fields (`last_detection`, `last_tracking`, `last_motility`, `last_assessment`, `last_visualization`).

## State Flow

Typical progression:

1. `io` populates `meta`, `video`, and optional groundtruth.
2. `preprocessing` writes derived video arrays.
3. `detection` writes active predicted detections.
4. `tracking` builds trajectories.
5. `motility` computes trajectory metrics.
6. `assessment` evaluates predictions against groundtruth.
