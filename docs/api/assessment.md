# API: Assessment

Assessment quantifies detector performance by comparing active predicted detections against groundtruth detections and storing classification-style metrics and per-frame logs in the session.

## Public Methods In This Section

- `self.assessment.classification(...)`

---

## `self.assessment.classification(...)`

Compute detection classification metrics (TP/FP/FN) against groundtruth detections.

Detections are matched using a Hungarian algorithm (minimum-distance assignment) over Euclidean pixel distances between detection centroids. A match is counted as a true positive only if the centroid distance is below `match_min_distance_pixel`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `match_min_distance_pixel` | `float \| None` | `None` | Distance threshold in pixels for counting a matched centroid pair as a true positive. If `None`, uses `casa["meta"]["match_min_distance_pixel"]` when available, otherwise defaults to `20`. |

**Raises**

- `ValueError` — if the resolved `match_min_distance_pixel` is ≤ 0.

**Output**

Results are written under `casa["assessment"]`:

| Key | Type | Description |
|-----|------|-------------|
| `classification` | `dict` | Per-evaluation metrics: `tp`, `fp`, `fn`, `precision`, `recall`, `F1`, `evaluated_frames`. |
| `classification_log` | `str` | Human-readable per-frame breakdown of TP/FP/FN counts. |
| `last_classification` | `dict` | Metadata about this run (method used, threshold, frame count). |

`casa["meta"]["last_assessment"]` is also updated.

**Metric Definitions**

| Metric | Description |
|--------|-------------|
| `tp` | True positives: predicted detections matched to a groundtruth within threshold distance. |
| `fp` | False positives: predicted detections with no matching groundtruth. |
| `fn` | False negatives: groundtruth detections with no matching prediction. |
| `precision` | `tp / (tp + fp)` as a percentage. |
| `recall` | `tp / (tp + fn)` as a percentage. |
| `F1` | Harmonic mean of precision and recall as a percentage. |
| `evaluated_frames` | Number of frames where both predicted and groundtruth detections were present. |

**Returns**

`Casa` — the same session instance.

**Notes**

- Assessment follows legacy pyCASA Hungarian-matching logic. Metric formulas use percentage values (0–100), not fractions.
- A concise metrics summary is printed to stdout after each run regardless of the `verbose` setting.
- Assessment evaluates only the intersection of frames that have both predicted and groundtruth detections within the loaded frame range.
- If no predicted detections are present, classification runs with empty predictions (all groundtruth counts as FN).
- If no groundtruth is present, a warning is printed and no TP/FN can be computed.

**Requirement**

Assessment requires:
- Predicted detections from a detection method (e.g., `self.detection.yolo()`), and
- Groundtruth detections loaded via `load_default_data()` or `load_video(..., groundtruth_path=...)`.

**Example**

```python
self.assessment.classification(match_min_distance_pixel=20)

assessment = self.get_assessment()
print(assessment["classification"])
# {'tp': 312, 'fp': 18, 'fn': 24, 'precision': 94.5, 'recall': 92.9, 'F1': 93.7, 'evaluated_frames': 81}
```
