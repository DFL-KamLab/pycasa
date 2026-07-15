# API: Assessment

Assessment scores predicted detections against groundtruth using Hungarian minimum-distance centroid matching, producing per-frame true-positive / false-positive / false-negative counts and the derived precision, recall, and F1 metrics.

Assessment requires:
- Predicted detections from a detection method (e.g., `self.detection.yolo()`), and
- Groundtruth detections loaded via `load_default_data()` or `load_video(..., groundtruth_path=...)`.

!!! note "This page is generated from the code"
    The signature, parameters, and description below are rendered directly from the
    `pycasa` source docstring, so they always match the installed version.

---

::: pycasa.casa.assessment.assessment_wrapper._SessionAssessmentNamespace.classification
