"""Tracking namespace for trajectory generation from detections.

Purpose:
    Convert detection/groundtruth streams into per-track trajectories.

Inputs:
    A ``Casa`` session with detections and tracking parameters.

Outputs:
    Track dictionaries under ``casa['tracks']`` and ``meta['last_tracking']``
    run metadata.

Methods:
    - ``sort(...)``
    - ``jpdaf(...)``
"""

from ._sort import sort
from ._jpdaf import jpdaf

__all__ = ["sort", "jpdaf"]
