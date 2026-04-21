# API: Motility

Motility transforms tracked trajectories into biologically meaningful motion metrics using a sliding-window approach. Each track is divided into overlapping windows, and a set of standard CASA motility parameters is computed for each window.

## Public Methods In This Section

- `self.motility.standard_motility_parameters(...)`

---

## `self.motility.standard_motility_parameters(...)`

Compute legacy-standard CASA motility parameters from tracked trajectories.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_rate` | `float \| None` | `None` | Frames per second override. If `None`, uses the session's configured sampling rate (set via `self.set_sampling_rate()`) when available, otherwise falls back to `30`. |
| `window_size` | `int` | `10` | Number of trajectory points per sliding window. Minimum effective value is `2`. |
| `overlap` | `float` | `0.2` | Window overlap ratio. The step between windows is computed as `max(1, int(window_size * (1 - overlap)))`. |
| `smoothing_window` | `int \| None` | `None` | Smoothing window size used for VAP and ALH computation. When `None`, defaults to `max(2, window_size // 2)`. |
| `conversion_required` | `bool` | `True` | If `True`, requires a valid pixel calibration set via `self.set_um_per_px()`. Velocity and displacement metrics are converted from pixel units to micron units. If `False`, missing calibration leaves output in pixel units. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing tracks. |
| `verbose` | `bool` | `True` | Print per-source motility summaries after computation. Does not suppress warnings. |

**Raises**

- `ValueError` — if `conversion_required=True` and `um_per_px` is missing, `None`, or not a positive finite number.
- `RuntimeError` — if no source has tracks with enough points to fill the requested `window_size`.

**Output**

Metrics are stored in the session keyed by detection source name. Each track ID maps to per-window metric lists. Retrieve results with:

```python
motility = self.get_motility()
```

**Returns**

`Casa` — the same session instance.

---

## Motility Metric Definitions

| Metric | Full Name | Unit | Formula |
|--------|-----------|------|---------|
| **VCL** | Curvilinear Velocity | µm/s (or px/s) | Total path length divided by elapsed time. Measures the actual speed along the full curvilinear trajectory. |
| **VSL** | Straight-Line Velocity | µm/s (or px/s) | Straight-line distance from first to last point divided by elapsed time. Measures net displacement speed. |
| **VAP** | Average Path Velocity | µm/s (or px/s) | Path length of the smoothed (average path) trajectory divided by elapsed time. Uses a moving-average smoothed version of the track. |
| **LIN** | Linearity | dimensionless | `VSL / VCL`. Ratio of straight-line to curvilinear velocity. Values near 1 indicate highly linear movement. |
| **ALH** | Amplitude of Lateral Head Displacement | µm (or px) | Half-range of lateral deviations from the average path. Measures the width of the head oscillation around the mean trajectory. |
| **WOB** | Wobble | dimensionless | `VAP / VCL`. Ratio of average path to curvilinear velocity. Reflects how much the cell deviates from its mean path. |
| **STR** | Straightness | dimensionless | `VSL / VAP`. Ratio of straight-line to average path velocity. Measures how close the mean path is to a straight line. |
| **MAD** | Mean Angular Displacement | degrees | Mean absolute change in direction angle between consecutive steps. Measures the average turning per frame. |

**Unit conversion:** When `um_per_px` is set and `conversion_required=True`, VCL/VSL/VAP are multiplied by `um_per_px` (pixel → µm), and ALH is multiplied by `um_per_px` (pixel → µm). LIN, WOB, STR, and MAD are dimensionless/angular and are not converted.

---

## Notes

- A minimum of `window_size` trajectory points is needed per window. Tracks shorter than `window_size` produce no output for that window.
- If a track's frames are not consecutive (gaps due to `max_age` coasting), the elapsed time accounts for the actual frame indices.
- Run `self.set_um_per_px(value)` before this step if calibration was not set at load time.

**Example**

```python
self.tracking.sort()
self.set_um_per_px(0.24)
self.motility.standard_motility_parameters(window_size=10, overlap=0.2)

motility = self.get_motility()
```
