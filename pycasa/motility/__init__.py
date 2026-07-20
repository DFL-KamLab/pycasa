"""Motility namespace for quantitative trajectory analysis.

Purpose:
    Compute per-track kinematic parameters and population-level CASA
    parameters from tracked trajectories.

Inputs:
    A ``Casa`` session with tracking output and optional calibration.

Outputs:
    Motility metrics under ``casa['motility']`` and ``meta['last_motility']``
    summary metadata.

Methods:
    - ``kinematic_parameters(...)`` — per-track velocities (VCL, VSL, VAP, ...).
    - ``casa_parameters(...)`` — population WHO motility grades (%rapid, %slow,
      %non-progressive, %immotile) plus optional concentration/volume/count.
"""

from ._casa_parameters import casa_parameters
from ._kinematic_parameters import kinematic_parameters

__all__ = ["casa_parameters", "kinematic_parameters"]
