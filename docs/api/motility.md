# API: Motility

The motility namespace has two stages:

- **`kinematic_parameters()`** — transforms tracked trajectories into per-track, per-window kinematic metrics (VCL, VSL, VAP, LIN, ALH, WOB, STR, MAD) using a sliding-window approach.
- **`casa_parameters()`** — aggregates those per-track kinematics into **population-level CASA parameters**: the four WHO motility grades (`%rapid`, `%slow`, `%non-progressive`, `%immotile`) plus optional sperm **concentration**, **volume**, and **total sperm count**.

Set pixel calibration with `self.set_um_per_px(value)` before computing motility if you want velocity and displacement metrics in micron units (and meaningful CASA grade thresholds). `load_default_data()` sets `um_per_px = 0.24` automatically for the bundled HC004 dataset.

!!! note "This page is generated from the code"
    The signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

::: pycasa.casa.motility.motility_wrapper._SessionMotilityNamespace.kinematic_parameters

---

::: pycasa.casa.motility.motility_wrapper._SessionMotilityNamespace.casa_parameters

---

## CASA Parameter Definitions

Population parameters from `casa_parameters()`. The four WHO motility grades always compute; the physical quantities are reported only when their inputs are available.

| Parameter | Needs | Meaning |
|-----------|-------|---------|
| **%rapid** | — | Progressive tracks with velocity ≥ `rapid_threshold` (default 25 µm/s) — WHO grade a. |
| **%slow** | — | Progressive tracks with velocity below `rapid_threshold` — WHO grade b. |
| **%non-progressive** | — | Motile tracks with `STR` below `progressive_str_threshold` (default 0.8) — WHO grade c. |
| **%immotile** | — | Tracks with velocity below `immotile_threshold` (default 5 µm/s) — WHO grade d. |
| **concentration** (10⁶/mL) | `um_per_px` (`chamber_depth_um` defaults to 20) | Mean cells per frame ÷ field volume (field area × chamber depth). |
| **volume** (mL) | `volume_ml` | Ejaculate volume — a manual lab measurement, passed in as metadata. |
| **total sperm count** (10⁶) | `volume_ml` + concentration | `volume_ml × concentration`. |

Set the physical inputs either as arguments to `casa_parameters(...)` or on the session via `self.set_volume_ml(...)` and `self.set_chamber_depth_um(...)`.

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
