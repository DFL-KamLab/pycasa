"""Assessment namespace for prediction-vs-groundtruth evaluation.

Purpose:
    Compute evaluation metrics for the active predicted detection output.

Inputs:
    A ``Casa`` session with both predicted detections and groundtruth.

Outputs:
    Assessment summaries/logs in ``casa['assessment']`` and metadata in
    ``meta['last_assessment']``.

Methods:
    - ``classification(...)``
"""

from ._classification import classification

__all__ = ["classification"]
