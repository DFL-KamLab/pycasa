# API: Motility

Motility transforms tracked trajectories into biologically meaningful motion metrics using a sliding-window approach. Each track is divided into overlapping windows, and a set of standard CASA motility parameters (VCL, VSL, VAP, LIN, ALH, WOB, STR, MAD) is computed for each window.

Set pixel calibration with `self.set_um_per_px(value)` before computing motility if you want velocity and displacement metrics in micron units. `load_default_data()` sets `um_per_px = 0.24` automatically for the bundled HC004 dataset.

!!! note "This page is generated from the code"
    The signature, parameters, and description below are rendered directly from the
    `pycasa` source docstring, so they always match the installed version.

---

::: pycasa.casa.motility.motility_wrapper._SessionMotilityNamespace.standard_motility_parameters

---

## Motility Metric Definitions

| Metric | Full Name | Unit | Meaning |
|--------|-----------|------|---------|
| **VCL** | Curvilinear Velocity | µm/s (or px/s) | Total path length divided by elapsed time — actual speed along the full curvilinear trajectory. |
| **VSL** | Straight-Line Velocity | µm/s (or px/s) | Straight-line distance from first to last point divided by elapsed time — net displacement speed. |
| **VAP** | Average Path Velocity | µm/s (or px/s) | Path length of the smoothed (average-path) trajectory divided by elapsed time. |
| **LIN** | Linearity | dimensionless | `VSL / VCL`. Values near 1 indicate highly linear movement. |
| **ALH** | Amplitude of Lateral Head Displacement | µm (or px) | Half-range of lateral deviations from the average path. |
| **WOB** | Wobble | dimensionless | `VAP / VCL`. How much the cell deviates from its mean path. |
| **STR** | Straightness | dimensionless | `VSL / VAP`. How close the mean path is to a straight line. |
| **MAD** | Mean Angular Displacement | degrees | Mean absolute change in direction angle between consecutive steps. |

**Unit conversion:** when `um_per_px` is set and `conversion_required=True`, VCL/VSL/VAP and ALH are converted from pixels to microns. LIN, WOB, STR, and MAD are dimensionless/angular and are not converted.
