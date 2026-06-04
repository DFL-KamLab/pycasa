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

## State Methods

### `info()`

Print a structured summary of the current CASA session and return `self`.

**Returns**

`Casa` — the same session instance (for chaining).

**Example**

```python
self.info()
```

---

### `copy()`

Return a deep-copied `Casa` session independent from this instance. Modifying the copy does not affect the original.

**Returns**

`Casa` — a new, independent session object.

**Example**

```python
session_copy = self.copy()
```

---

### `set_um_per_px(um_per_px)`

Set the microns-per-pixel calibration value on the current session.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `um_per_px` | `float` | Positive finite calibration value. Required for motility unit conversion (px/s → µm/s). |

**Returns**

`Casa` — the same session instance (for chaining).

**Raises**

- `ValueError` — if `um_per_px` is not a positive finite number.

**Example**

```python
self.set_um_per_px(0.24)
```

---

## Getter Methods

### `get_casa()`

Return the validated CASA session dictionary.

**Returns**

`dict` — the full session dictionary with all populated keys.

---

### `get_meta()`

Return the session metadata dictionary (`casa['meta']`).

**Returns**

`dict` — metadata including `video_path`, `width`, `height`, `total_number_frame`, `sampling_rate`, `um_per_px`, `magnification`, and all `last_*` provenance fields.

---

### `get_video()`

Return the video data dictionary (`casa['video']`).

**Returns**

`dict` — contains `original_video`, `grayscale_video`, `normalized_video`, `binary_video`, `initial_frame`, `final_frame`, `number_frame_used`, and any other derived video arrays.

---

### `get_detections(include_groundtruth=False)`

Return detection data from the current session.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_groundtruth` | `bool` | `False` | When `False`, returns only active predicted detections. When `True`, returns the full `casa['detections']` dict including groundtruth. |

**Returns**

`dict` — frame-indexed detections. Each key is a frame index string; each value is a list of detection rows in legacy normalized format `[label, norm_x, norm_y, norm_w, norm_h]`.

---

### `get_groundtruth()`

Return `casa['detections']['groundtruth']`.

**Returns**

`dict` — frame-indexed groundtruth detections, or an empty dict if no groundtruth was loaded.

---

### `get_tracks(backend=None)`

Return tracking output from the current session.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str \| None` | `None` | When `None`, returns all tracks under `casa['tracks']`. When set (e.g., `"sort"`), returns only `casa['tracks']['sort']`. |

**Returns**

`dict` — tracks keyed by detection source name.

**Example**

```python
sort_tracks = self.get_tracks(backend="sort")
```

---

### `get_motility()`

Return `casa['motility']`.

**Returns**

`dict` — motility output, keyed by method name (e.g., `"standard_motility_parameters"`).

---

### `get_assessment()`

Return `casa['assessment']`.

**Returns**

`dict` — assessment output including classification metrics and logs.

---

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
