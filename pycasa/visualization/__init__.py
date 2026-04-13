"""Visualization namespace for session inspection and reporting.

Purpose:
    Display frames, overlays, interactive trajectory windows, and motility
    summary plots.

Inputs:
    A ``Casa`` session with loaded/processed arrays and optional detections,
    tracks, and motility results.

Outputs:
    Interactive/static matplotlib visualizations and
    ``meta['last_visualization']`` updates.

Methods:
    - ``plot_frame(...)``
    - ``timelapse(...)``
    - ``interactive_motility_calculator(...)``
    - ``motility_radar(...)``
    - ``motility_density_scatter(...)``
"""

from ._motility_density_scatter import motility_density_scatter
from ._interactive_motility_calculator import interactive_motility_calculator
from ._motility_radar import motility_radar
from ._plot_frame import plot_frame
from ._timelapse import timelapse

__all__ = [
    "plot_frame",
    "timelapse",
    "interactive_motility_calculator",
    "motility_radar",
    "motility_density_scatter",
]
