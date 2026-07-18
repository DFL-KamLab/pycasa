"""Assessment namespace for prediction-vs-groundtruth evaluation.

Purpose:
    Compute evaluation metrics for the active predicted detection output.

Inputs:
    A ``Casa`` session with both predicted detections and groundtruth.

Outputs:
    Assessment summaries/logs in ``casa['assessment']`` and metadata in
    ``meta['last_assessment']``.

Methods:
    - ``evaluate_detections(...)`` — predicted detections vs groundtruth detections.
    - ``evaluate_tracks(...)`` — predicted tracks vs imported groundtruth tracks (MOTA/IDF1).
"""

from ._evaluate_detections import evaluate_detections
from ._evaluate_tracks import evaluate_tracks

__all__ = ["evaluate_detections", "evaluate_tracks"]
