"""I/O namespace for creating and populating ``Casa`` sessions.

Purpose:
    Entry points for loading raw video data and optional bundled default data.

Inputs:
    File-system paths, frame-range options, sampling/calibration metadata, and
    verbosity/progress preferences.

Outputs:
    ``Casa`` session objects with populated ``meta``, ``video``, and optional
    ``detections['groundtruth']``.

Methods:
    - ``load_video(...)``
    - ``load_default_data(...)``
"""

from ._load_default_data import load_default_data
from ._load_video import load_video

__all__ = [
    "load_video",
    "load_default_data",
]
