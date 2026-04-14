# API: Tracking

Purpose:
Tracking links frame-level detections over time into trajectories so downstream motility and visualization steps can operate on motion continuity rather than isolated points.

## Public Methods In This Section

- `self.tracking.sort(...)`

## `sort`

```python
self.tracking.sort(
    skip_gt=False,
    max_age=25,
    min_hits=3,
    iou_threshold=0.1,
    show_progress=True,
    verbose=True,
)
```

Key behavior:

- Can run on both groundtruth and active predicted detections.
- Writes track data to `casa["tracks"]["sort"]`.
- Updates `casa["meta"]["last_tracking"]`.

Inspect output:

```python
sort_tracks = self.get_tracks(backend="sort")
```
