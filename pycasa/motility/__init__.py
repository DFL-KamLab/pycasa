"""Motility namespace for quantitative trajectory analysis.

Purpose:
    Compute standard motility parameters from tracked trajectories.

Inputs:
    A ``Casa`` session with tracking output and optional calibration.

Outputs:
    Motility metrics under ``casa['motility']`` and ``meta['last_motility']``
    summary metadata.

Methods:
    - ``standard_motility_parameters(...)``
"""

from ._standard_motility_parameters import standard_motility_parameters

__all__ = ["standard_motility_parameters"]
