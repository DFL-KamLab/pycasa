# API: Tracking

Tracking links frame-level detections over time into trajectories so downstream motility and visualization steps can operate on motion continuity rather than isolated per-frame points.

pycasa implements the **SORT** (Simple Online and Realtime Tracking) algorithm, which uses Kalman filtering for state prediction and the Hungarian algorithm for IoU-based detection-to-track assignment.

## Public Methods In This Section

- `self.tracking.sort(...)`

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

Results are written to `casa["tracks"]["sort"]`, keyed by detection source name:

```python
casa["tracks"]["sort"] = {
    "groundtruth": {track_id: {frame: [x, y, ...], ...}, ...},
    "yolov5":      {track_id: {frame: [x, y, ...], ...}, ...},
}
```

Each `track_id` maps to a dict of `{frame_index: [x, y, w, h]}` coordinate data.

`casa["meta"]["last_tracking"]` is also updated.

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
