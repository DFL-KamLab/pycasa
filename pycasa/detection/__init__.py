"""Detection namespace for producing active predicted detections.

Purpose:
    Run supported detection backends and store one active predicted detection
    output (groundtruth remains separately stored when available).

Inputs:
    A ``Casa`` session with loaded video and method-specific runtime options.

Outputs:
    Frame-indexed detection rows in ``casa['detections'][<method>]`` plus
    ``meta['last_detection']`` updates.

Methods:
    - ``detect_moving_cells(...)``
    - ``digital_washing(...)``
    - ``yolo(...)``
"""

from ._detect_moving_cells import detect_moving_cells
from ._digital_washing import digital_washing
from ._yolo import yolo

__all__ = ["detect_moving_cells", "digital_washing", "yolo"]
