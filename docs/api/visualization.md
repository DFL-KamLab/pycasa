# API: Visualization

Visualization provides interactive and static inspection surfaces for every stage of the session, from raw frames and overlays to track exploration and motility summaries.

All visualization methods:
- Update `casa["meta"]["last_visualization"]`.
- Return the same `Casa` instance for fluent chaining.
- Require `matplotlib` to be installed.

## Public Methods In This Section

- `self.visualization.plot_frame(...)`
- `self.visualization.timelapse(...)`
- `self.visualization.interactive_motility_calculator(...)`
- `self.visualization.motility_radar(...)`
- `self.visualization.motility_density_scatter(...)`

## Typical Use Order

1. Use `plot_frame` for quick spot-checks on a single frame.
2. Use `timelapse` for full pipeline review with overlays.
3. Use `interactive_motility_calculator` after tracking to explore individual track metrics.
4. Use `motility_radar` and `motility_density_scatter` after motility computation for aggregate summaries.

---

## `self.visualization.plot_frame(...)`

Plot one frame across one or more image representations as a static matplotlib figure.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_type` | `str \| Iterable[str]` | `"original"` | Image representation(s) to display. Accepts a single name, multiple names joined by `+` or `,`, or an iterable. Each representation is plotted as a separate subplot. Supported values: `"original"`, `"grayscale"`, `"normalized"`, `"binarized"`, `"moving_cells"`. |
| `frame_index` | `int \| None` | `None` | Zero-based local frame index to display. When `None`, the middle frame is selected automatically. |
| `show_detections` | `bool` | `True` | Overlay detections as bounding boxes on the plotted frame. Uses the active predicted detection method, or groundtruth if no predicted detections are available. |

**Returns**

`Casa` — the same session instance.

**Raises**

- `ValueError` — if a requested image type has no corresponding video array (e.g., `"binarized"` before running binarization), or if `frame_index` is out of range.
- `ImportError` — if `matplotlib` is unavailable.

**Notes**

- This method is single-frame only. For interactive timeline browsing, use `timelapse`.
- The required video array must exist before plotting (e.g., run `self.preprocessing.binarization.otsu()` before requesting `"binarized"`).

**Example**

```python
# Show original frame with detections
self.visualization.plot_frame(frame_index=10, show_detections=True)

# Compare original and binarized side-by-side
self.visualization.plot_frame(image_type="original+binarized", frame_index=5)
```

---

## `self.visualization.timelapse(...)`

Open an interactive time-lapse viewer with frame scrubbing, playback, and overlay toggles.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_type` | `str` | `"original"` | Video representation(s) to show. Accepts a single name or multiple names joined by `+` or `,`. Supported values: `"original"`, `"grayscale"` (alias `"gray"`), `"normalized"`, `"binarized"` (alias `"binary"`), `"moving_cells"`. Multiple values create side-by-side panels. |
| `image_type` | `str \| None` | `None` | Deprecated alias for `video_type`. If provided, overrides `video_type`. |
| `show_detections` | `bool` | `True` | Initial visibility of active predicted detection overlays (bounding boxes). |
| `show_tracks` | `bool` | `False` | Initial visibility of track trajectory lines. |
| `show_groundtruth` | `bool` | `True` | Initial visibility of groundtruth detection overlays (bounding boxes). |
| `show_track_ids` | `bool` | `False` | Whether to annotate each track head with its track ID. Can be slow for large track sets. |

**Controls**

| Control | Action |
|---------|--------|
| Frame slider | Scrub through frames. |
| Play / Pause button | Animate frames at the session sampling rate. |
| Left / Right arrow keys | Step one frame backward / forward. |
| Space bar | Toggle play/pause. |
| Overlay toggle buttons | Show/hide detections, tracks, and groundtruth independently. |

**Overlay Rendering**

- Predicted detections and groundtruth are drawn as bounding boxes.
- Tracks are drawn as thin trajectory lines showing the full path up to the current frame.

**Returns**

`Casa` — the same session instance.

**Raises**

- `ValueError` — if requested video layers are unavailable or frame data is invalid.
- `ImportError` — if `matplotlib` is unavailable.

**Example**

```python
self.visualization.timelapse(
    video_type="original",
    show_detections=True,
    show_tracks=True,
    show_groundtruth=True,
)
```

---

## `self.visualization.interactive_motility_calculator(...)`

Open an interactive motility-parameter explorer for active SORT tracks.

Allows selecting individual tracks and navigating sliding-window segments to inspect per-window motility metrics. The right panel shows 8 metric history tiles (VCL, VSL, VAP, LIN, ALH, WOB, STR, MAD) with a highlighted current window and, when available, global mean/std overlaid from `standard_motility_parameters` output.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_rate` | `float \| None` | `None` | FPS override for segment metric computation. If `None`, uses `casa["meta"]["sampling_rate"]` when available, otherwise `30`. |
| `smoothing_window` | `int` | `5` | Smoothing window used for VAP/ALH calculations in the segment metric preview. |

**Returns**

`Casa` — the same session instance.

**Raises**

- `ValueError` — if video width/height metadata cannot be resolved, or if `um_per_px` is present but invalid.
- `RuntimeError` — if active SORT tracks are missing or no track has enough points for interactive exploration.
- `ImportError` — if `matplotlib` is unavailable.

**Notes**

- Requires prior SORT tracking output (`self.tracking.sort()`).
- Track selector uses a scroll-style index slider and a compact track list.
- Window and step sliders control segment navigation within the selected track.

**Example**

```python
self.tracking.sort()
self.visualization.interactive_motility_calculator(smoothing_window=5)
```

---

## `self.visualization.motility_radar(...)`

Display a radar (spider) chart of aggregate motility metric means across active SORT tracks.

Requires prior motility computation (`self.motility.standard_motility_parameters()`).

**Returns**

`Casa` — the same session instance.

**Raises**

- `ImportError` — if `matplotlib` is unavailable.

**Example**

```python
self.motility.standard_motility_parameters()
self.visualization.motility_radar()
```

---

## `self.visualization.motility_density_scatter(...)`

Display a scatter plot of two motility parameters across all track windows, with density shading.

Requires prior motility computation (`self.motility.standard_motility_parameters()`).

**Returns**

`Casa` — the same session instance.

**Raises**

- `ImportError` — if `matplotlib` is unavailable.

**Example**

```python
self.motility.standard_motility_parameters()
self.visualization.motility_density_scatter()
```
