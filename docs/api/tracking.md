# API: Tracking

Tracking links frame-level detections over time into trajectories so downstream motility and visualization steps can operate on motion continuity rather than isolated per-frame points.

pycasa implements three interchangeable tracking backends, each with its own trade-offs:

| Backend | Algorithm | Best for |
|---|---|---|
| `sort` | Kalman filter + Hungarian IoU assignment ([Bewley et al. 2016](https://arxiv.org/abs/1602.00763)) | Fast, simple, robust default. |
| `deepsort` | SORT + appearance embeddings ([Wojke et al. 2017](https://github.com/nwojke/deep_sort)) | Same as SORT when appearance is uninformative; supports custom ReID features. |
| `jpdaf` | Joint Probabilistic Data Association Filter ([Urbano et al. 2017](https://doi.org/10.1109/TMI.2016.2630720)) | Dense, occluded sperm fields with overlapping trajectories. |

All three write to the standard `casa["tracks"][backend][source]` schema, so downstream motility and visualization steps work identically regardless of which one you choose. Calling a new backend clears `casa["tracks"]` and a yellow warning is emitted to flag the overwrite.

## Public Methods In This Section

- `self.tracking.sort(...)`
- `self.tracking.deepsort(...)`
- `self.tracking.jpdaf(...)`

---

## `self.tracking.sort(...)`

Run SORT tracking over the active predicted detections (and optionally groundtruth).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_gt` | `bool` | `False` | If `True`, skip tracking the groundtruth detection source. When `False`, SORT runs on both groundtruth and predicted detections (if both are available). |
| `delete_temp` | `bool` | `True` | Legacy compatibility flag. No temporary files are created by the in-process implementation, so this has no effect. |
| `max_age` | `int` | `25` | Maximum number of consecutive frames a track is allowed to go without a matching detection before it is dropped. Higher values allow tracks to "coast" through occlusions. |
| `min_hits` | `int` | `3` | Minimum number of consecutive detections required before a new track is confirmed and included in output. Filters out spurious short-lived detections. |
| `iou_threshold` | `float` | `0.1` | Intersection-over-Union (IoU) overlap required between a detection and a track prediction to count as a valid association. Lower values allow looser matching, which is useful for small or low-overlap bounding boxes. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress warnings. |

**Algorithm Details**

For each frame, SORT:

1. Predicts each active track's next bounding box using a Kalman filter.
2. Computes the IoU matrix between predicted boxes and current detections.
3. Solves the assignment problem using the Hungarian algorithm (or LAP if available).
4. Updates matched tracks, marks unmatched tracks as missed (incrementing their age), and initializes new tracks for unmatched detections.
5. Drops tracks that exceed `max_age` without a match.
6. Only outputs tracks that have reached `min_hits` consecutive hits.

**Output**

Tracking results are keyed by detection source name. Each track ID maps to a dict of `{frame_index: [x, y, w, h]}` coordinate data. Retrieve results with:

```python
sort_tracks = self.get_tracks(backend="sort")
```

**Returns**

`Casa` — the same session instance.

**Example**

```python
# Track both groundtruth and predicted detections
self.tracking.sort(skip_gt=False, max_age=25, min_hits=3, iou_threshold=0.1)

# Inspect output
sort_tracks = self.get_tracks(backend="sort")
```

**Notes**

- If no detections are available, sort still runs on groundtruth (unless `skip_gt=True`).
- Re-run tracking after changing the detection method to ensure tracks correspond to current detections.
- The LAP package (`lap`) is used for the assignment step when available; otherwise SciPy's `linear_sum_assignment` is used as a fallback.

---

## `self.tracking.deepsort(...)`

Run DeepSORT tracking over the active predicted detections (and optionally groundtruth). DeepSORT extends SORT with an appearance-feature matching cascade in addition to the Kalman/IoU gate.

On first invocation, the upstream `nwojke/deep_sort` repository is automatically cloned to `~/.pycasa/deepsort/` and used in-process — no manual install required.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_gt` | `bool` | `False` | If `True`, skip tracking the groundtruth detection source and only track predicted detections. |
| `max_age` | `int` | `30` | Maximum number of consecutive missed frames before a track is dropped. |
| `n_init` | `int` | `3` | Minimum consecutive detections required before a track is confirmed (equivalent to SORT's `min_hits`). |
| `max_iou_distance` | `float` | `0.7` | Maximum IoU **distance** for gating (distance = 1 − IoU). The default `0.7` corresponds to a minimum IoU of `0.3`. |
| `max_cosine_distance` | `float` | `1.0` | Appearance-feature gate threshold. The default `1.0` effectively disables appearance-based matching, falling back to Kalman/IoU-only association — the recommended setting for biological cells whose visual appearance is nearly identical across tracks. Lower values (e.g. `0.3`) tighten appearance matching when meaningful ReID features are provided. |
| `nn_budget` | `int \| None` | `None` | Maximum number of appearance features stored per track for the nearest-neighbour appearance metric. `None` means unlimited. |
| `initial_frame` | `int` | `0` | Offset (in frames) from the start of the analyzed video at which to begin tracking. Frames before this offset are skipped entirely. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress warnings. |

**Output**

Tracking results are stored under `casa["tracks"]["deepsort"][source]`. Each track ID maps to a dict of `{frame_index: [center_x, center_y]}`. Retrieve results with:

```python
deepsort_tracks = self.get_tracks(backend="deepsort")
```

**Returns**

`Casa` — the same session instance.

**Example**

```python
# Default: IoU-only association, suitable for biological cells
self.tracking.deepsort(max_age=30, n_init=3, max_iou_distance=0.7)

# Enable appearance-based matching with custom ReID features
self.tracking.deepsort(max_cosine_distance=0.3, nn_budget=100)
```

**Notes**

- On first call, requires `git` to be available on PATH for the one-time clone of `nwojke/deep_sort`.
- Track points are stored as global-frame center coordinates (not bounding boxes).
- Re-run tracking after changing the detection method to keep tracks consistent.

---

## `self.tracking.jpdaf(...)`

Run the **Joint Probabilistic Data Association Filter** tracker over the active predicted detections (and optionally groundtruth). JPDAF is a probabilistic multi-target tracker that, instead of committing to a single detection-to-track assignment per frame, weights every feasible joint association by its posterior probability — making it well-suited to dense, occluded sperm fields where IoU-based assignment becomes ambiguous.

All algorithm parameters are taken directly from Urbano et al. (2017) and scaled to pixel space using `casa["meta"]["um_per_px"]`. If calibration is missing, approximate pixel-space defaults are used and a yellow warning is emitted.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_gt` | `bool` | `False` | If `True`, skip tracking the groundtruth detection source. |
| `frame_rate` | `float \| None` | `None` | Frames per second. When `None`, read from `casa["meta"]["sampling_rate"]`. The frame period `T = 1 / frame_rate` is used in the CWNA motion model. |
| `initial_frame` | `int` | `0` | Offset (in frames) from the start of the analyzed video at which to begin tracking. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress warnings. |

**Algorithm-fixed parameters** (paper values, applied automatically)

| Symbol | Value | Description |
|---|---|---|
| σₙ | 2.0 µm | Measurement noise standard deviation. |
| q̃₀ | 20 µm/s | Velocity-uncertainty amplitude for the CWNA motion model. |
| γᵥ | 300 µm/s | Velocity-gate threshold. |
| P_D | 0.95 | Per-target detection probability. |
| λ | 1e-6 µm⁻² | Poisson clutter spatial density. |
| λₙ | 1e-5 µm⁻² | New-target spatial density. |

**Output**

Tracking results are stored under `casa["tracks"]["jpdaf"][source]`. Each track ID maps to a dict of `{frame_index: [center_x, center_y]}`. Retrieve results with:

```python
jpdaf_tracks = self.get_tracks(backend="jpdaf")
```

**Returns**

`Casa` — the same session instance.

**Raises**

- `ValueError` — if `frame_rate` cannot be determined from parameters or session metadata, or if video width/height cannot be resolved.

**Example**

```python
# Frame rate from the loaded video metadata, um_per_px from session
self.tracking.jpdaf()

# Explicit overrides
self.tracking.jpdaf(frame_rate=50.0, initial_frame=20)
```

**Notes**

- Setting `um_per_px` via `pc.io.load_video(..., um_per_px=...)` or `self.set_um_per_px(...)` before calling `jpdaf()` is strongly recommended — the paper's noise constants are micron-calibrated.
- JPDAF is more compute-intensive than SORT/DeepSORT; per-frame cost scales with cluster size.
- Track output schema matches SORT/DeepSORT, so motility and visualization steps need no changes.

---

## Citations

**SORT** in pycasa is adapted from the original work by Alex Bewley:

> Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). **Simple Online and Realtime Tracking.** *IEEE International Conference on Image Processing (ICIP)*. [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)

Source code: [https://github.com/abewley/sort](https://github.com/abewley/sort) — licensed under GPL-3.0.

**Modification note:** The original implementation does not handle frames with zero detections, which causes a crash during the IoU matrix computation step. pycasa adds a two-line early-return guard in `_iou_batch` that returns a correctly-shaped zero matrix when either input is empty. If the upstream repository is updated to include this fix, the intent is to remove the local copy and have users pull from the original repo directly.

**DeepSORT** uses the original implementation by Nicolai Wojke, auto-cloned on first use:

> Wojke, N., Bewley, A., & Paulus, D. (2017). **Simple Online and Realtime Tracking with a Deep Association Metric.** *IEEE International Conference on Image Processing (ICIP)*. [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)

Source code: [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort).

**JPDAF** implements the algorithm and parameter values from:

> Urbano, L.F., Masson, P., VerMilyea, M., & Kam, M. (2017). **Automatic Tracking and Motility Analysis of Human Sperm in Time-Lapse Images.** *IEEE Transactions on Medical Imaging*, 36(3), 792–801. [DOI:10.1109/TMI.2016.2630720](https://doi.org/10.1109/TMI.2016.2630720)
