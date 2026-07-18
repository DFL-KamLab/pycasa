# API: Assessment

Assessment scores predictions against groundtruth. Two evaluations are available:

- **`evaluate_detections()`** — scores predicted detections against groundtruth detections using Hungarian minimum-distance centroid matching, producing per-frame true-positive / false-positive / false-negative counts and the derived precision, recall, and F1 metrics.
- **`evaluate_tracks()`** — compares **every available track set against every other** (all ordered pairs), producing MOT-style identity metrics (**MOTA, IDF1**, ID-switches, FP/FN, fragmentations) via per-frame center-distance matching. Track sets are the imported groundtruth tracks (`casa["tracks"]["groundtruth_tracks"]`) plus each source of the active tracking backend (e.g. `sort:groundtruth`, `sort:yolov5`). Results are stored as a matrix under `casa["assessment"]["tracking"]["pairs"]` and printed as MOTA/IDF1 tables. When imported groundtruth tracks are present they are the reference row (true *accuracy*); other rows are pairwise *agreement*. Note **MOTA is role-dependent** (`MOTA(A,B) ≠ MOTA(B,A)`) while **IDF1 is symmetric**.

Requirements:
- `evaluate_detections()` — predicted detections from a detection method (e.g., `self.detection.yolo()`) and groundtruth detections loaded via `load_default_data()` or `load_video(..., groundtruth_detections_path=...)`.
- `evaluate_tracks()` — **at least two track sets** in the session. Typically a tracker run (e.g., `self.tracking.sort()`) plus imported groundtruth tracks loaded via `load_video(..., groundtruth_tracks_path=...)`; imported truth is optional but is what makes the reported numbers true accuracy rather than tracker-to-tracker agreement. Uses the optional `motmetrics` dependency, installed on demand.

!!! note "This page is generated from the code"
    The signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

::: pycasa.casa.assessment.assessment_wrapper._SessionAssessmentNamespace.evaluate_detections

---

::: pycasa.casa.assessment.assessment_wrapper._SessionAssessmentNamespace.evaluate_tracks
