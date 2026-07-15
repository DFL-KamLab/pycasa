# API: Visualization

Visualization provides interactive and static inspection surfaces for every stage of the session, from raw frames and overlays to track exploration and motility summaries.

All visualization methods:
- Update `casa["meta"]["last_visualization"]`.
- Return the same `Casa` instance for fluent chaining.
- Require `matplotlib` to be installed.

## Typical Use Order

1. Use `plot_frame` for quick spot-checks on a single frame.
2. Use `timelapse` for full pipeline review with overlays.
3. Use `interactive_motility_calculator` after tracking to explore individual track metrics.
4. Use `motility_radar` and `motility_density_scatter` after motility computation for aggregate summaries.

!!! note "This page is generated from the code"
    Signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

::: pycasa.casa.visualization.visualization_wrapper._SessionVisualizationNamespace.plot_frame

---

::: pycasa.casa.visualization.visualization_wrapper._SessionVisualizationNamespace.timelapse

---

::: pycasa.casa.visualization.visualization_wrapper._SessionVisualizationNamespace.interactive_motility_calculator

---

::: pycasa.casa.visualization.visualization_wrapper._SessionVisualizationNamespace.motility_radar

---

::: pycasa.casa.visualization.visualization_wrapper._SessionVisualizationNamespace.motility_density_scatter
