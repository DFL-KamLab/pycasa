# API: Assessment

Purpose:
Assessment quantifies detector performance by comparing active predicted detections against groundtruth detections and storing classification-style metrics and logs.

## Public Methods In This Section

- `self.assessment.classification(...)`

## Example

```python
self.assessment.classification(show_progress=True, verbose=True)
assessment = self.get_assessment()
```

## Output Behavior

- Writes evaluation results under `casa["assessment"]`.
- Updates `casa["meta"]["last_assessment"]`.

## Requirement

Assessment requires:

- predicted detections from a detection method, and
- groundtruth detections (for example from `load_default_data` or `load_video(..., groundtruth_path=...)`).
