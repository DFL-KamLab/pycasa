from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _import_matplotlib_for_visualization
from ..utils import _resolve_visualization_source


def _is_nested_source_motility_map(candidate: dict[str, Any]) -> bool:
    """Return ``True`` for shape ``source -> track_id -> metric_dict``."""
    for source_value in candidate.values():
        if not isinstance(source_value, dict):
            continue
        if any(isinstance(track_value, dict) for track_value in source_value.values()):
            return True
    return False


def _resolve_motility_sources(
    motility_root: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve motility data into ``source -> track metrics`` maps."""
    if not isinstance(motility_root, dict) or not motility_root:
        return {}
    if not _is_nested_source_motility_map(motility_root):
        return {"groundtruth": motility_root}

    sources: dict[str, dict[str, Any]] = {}
    for source_name, source_data in motility_root.items():
        if isinstance(source_data, dict) and source_data:
            sources[str(source_name)] = source_data
    return sources


def motility_density_scatter(casa: dict[str, Any]) -> dict[str, Any]:
    """Render KDE density scatter plots for active standard motility metrics.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with metadata written to
            ``casa["meta"]["last_visualization"]``.

    Raises:
        RuntimeError:
            If active standard motility results are missing or contain no
            numeric values for plotting.
        ImportError:
            If ``matplotlib`` or ``scipy`` is unavailable.

    Notes:
        - Uses active source data resolved from
          ``casa["motility"]["standard_motility_parameters"][source]``.
        - KDE uses ``scipy.stats.gaussian_kde`` when feasible; singular/low-N
          fallbacks use uniform point density.

    Examples:
        >>> session = session.motility.standard_motility_parameters()
        >>> session = session.visualization.motility_density_scatter()
    """
    casa = _ensure_casa(casa)
    motility_root = casa.get("motility", {}).get("standard_motility_parameters", {})
    if not isinstance(motility_root, dict) or not motility_root:
        raise RuntimeError(
            "No motility parameters found under 'standard_motility_parameters'. "
            "Run `self.motility.standard_motility_parameters()` first."
        )
    source_map = _resolve_motility_sources(motility_root)
    if not source_map:
        raise RuntimeError(
            "No motility parameters were found for the active source. "
            "Run `self.motility.standard_motility_parameters()` after tracking."
        )

    plt = _import_matplotlib_for_visualization("motility-density-scatter")
    mpl_widgets = _ensure_import("matplotlib.widgets", pip_name="matplotlib")
    scipy_stats = _ensure_import("scipy.stats", pip_name="scipy", prompt_install=False)
    Button = mpl_widgets.Button
    gaussian_kde = scipy_stats.gaussian_kde

    source_order = sorted(source_map.keys(), key=lambda value: (value != "groundtruth", value))
    preferred_source = _resolve_visualization_source(casa)
    selected_source = (
        preferred_source
        if preferred_source in source_map
        else source_order[0]
    )
    state = {"source": str(selected_source)}

    pairs = [
        ("VSL", "VCL", "VSL (um/s or px/s)", "VCL (um/s or px/s)"),
        ("ALH", "LIN", "ALH (um or px)", "LIN = VSL/VCL"),
        ("VSL", "WOB", "VSL (um/s or px/s)", "WOB = VAP/VCL"),
        ("MAD", "LIN", "MAD (deg)", "LIN = VSL/VCL"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    right_margin = 0.80 if len(source_order) > 1 else 0.85
    plt.subplots_adjust(right=right_margin)
    source_buttons: dict[str, dict[str, Any]] = {}
    if len(source_order) > 1:
        base_y = 0.78
        row_gap = 0.09
        for row_idx, source_name in enumerate(source_order):
            button_ax = figure.add_axes([0.82, base_y - row_idx * row_gap, 0.16, 0.06])
            button = Button(
                button_ax,
                str(source_name),
                color="white",
                hovercolor="#efefef",
            )
            button_ax.set_xticks([])
            button_ax.set_yticks([])
            source_buttons[str(source_name)] = {"axis": button_ax, "button": button}
    colorbar_state: dict[str, Any] = {"axis": None}

    def _refresh_source_button_styles() -> None:
        """Render source selector buttons with active/inactive styles."""
        for source_name, button_data in source_buttons.items():
            axis_obj = button_data["axis"]
            button_obj = button_data["button"]
            is_active = str(source_name) == str(state["source"])
            if is_active:
                base_color = "black"
                hover_color = "#2f2f2f"
                text_color = "white"
            else:
                base_color = "white"
                hover_color = "#efefef"
                text_color = "black"
            button_obj.color = base_color
            button_obj.hovercolor = hover_color
            axis_obj.set_facecolor(base_color)
            button_obj.label.set_color(text_color)
            button_obj.label.set_fontsize(9)
            button_obj.label.set_fontweight("bold")
            for spine in axis_obj.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

    def _build_pair_data_for_source(source_name: str) -> dict[tuple[str, str], tuple[list[float], list[float]]]:
        """Collect metric pairs for density scatter plotting for one source."""
        pair_data: dict[tuple[str, str], tuple[list[float], list[float]]] = {
            (x_key, y_key): ([], [])
            for x_key, y_key, _, _ in pairs
        }
        method_data = source_map.get(str(source_name), {})
        if not isinstance(method_data, dict):
            return pair_data

        for metrics in method_data.values():
            if not isinstance(metrics, dict):
                continue
            for x_key, y_key, _, _ in pairs:
                x_values = metrics.get(x_key, [])
                y_values = metrics.get(y_key, [])
                if not isinstance(x_values, list) or not isinstance(y_values, list):
                    continue

                pair_count = min(len(x_values), len(y_values))
                xs, ys = pair_data[(x_key, y_key)]
                for index in range(pair_count):
                    try:
                        xs.append(float(x_values[index]))
                        ys.append(float(y_values[index]))
                    except (TypeError, ValueError):
                        continue
        return pair_data

    def _draw_density_for_source(source_name: str) -> None:
        """Render all scatter panels for one selected source."""
        pair_data = _build_pair_data_for_source(source_name)
        total_values = 0
        for x_key, y_key, _, _ in pairs:
            total_values += min(len(pair_data[(x_key, y_key)][0]), len(pair_data[(x_key, y_key)][1]))
        if total_values <= 0:
            raise RuntimeError("No numeric motility samples are available for density scatter plots.")

        first_scatter = None
        for axis_obj, (x_key, y_key, x_label, y_label) in zip(axes.flat, pairs):
            axis_obj.clear()

            x_values = np.asarray(pair_data[(x_key, y_key)][0], dtype=float)
            y_values = np.asarray(pair_data[(x_key, y_key)][1], dtype=float)
            if x_values.size and y_values.size:
                mask = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                x_values = np.empty((0,), dtype=float)
                y_values = np.empty((0,), dtype=float)

            if x_values.size <= 1:
                density = np.ones_like(x_values, dtype=float)
            else:
                try:
                    xy = np.vstack([x_values, y_values])
                    density = gaussian_kde(xy)(xy)
                except Exception:
                    density = np.ones_like(x_values, dtype=float)

            scatter = axis_obj.scatter(
                x_values,
                y_values,
                c=density if density.size else None,
                s=20,
                cmap="viridis",
                edgecolors="none",
                alpha=0.8,
            )
            if first_scatter is None and x_values.size:
                first_scatter = scatter

            if (x_key, y_key) == ("VSL", "VCL"):
                axis_obj.set_xlim(0, 36)
                axis_obj.set_ylim(0, 60)
            elif (x_key, y_key) == ("ALH", "LIN"):
                axis_obj.set_xlim(0, 5)
                axis_obj.set_ylim(0, 1)
            elif (x_key, y_key) == ("VSL", "WOB"):
                axis_obj.set_xlim(0, 36)
                axis_obj.set_ylim(0, 1)
            elif (x_key, y_key) == ("MAD", "LIN"):
                axis_obj.set_xlim(0, 180)
                axis_obj.set_ylim(0, 1)

            axis_obj.set_xlabel(x_label)
            axis_obj.set_ylabel(y_label)

            if x_values.size:
                mean_x = float(x_values.mean())
                std_x = float(x_values.std())
                mean_y = float(y_values.mean())
                std_y = float(y_values.std())
                summary_text = f"{x_key}: {mean_x:.1f}+/-{std_x:.1f}\n{y_key}: {mean_y:.1f}+/-{std_y:.1f}"
            else:
                summary_text = f"{x_key}: n/a\n{y_key}: n/a"

            axis_obj.text(
                0.95,
                0.95,
                summary_text,
                transform=axis_obj.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        if colorbar_state["axis"] is not None:
            try:
                colorbar_state["axis"].remove()
            except Exception:
                pass
            colorbar_state["axis"] = None

        if first_scatter is not None:
            colorbar_axis = figure.add_axes([right_margin + 0.03, 0.15, 0.02, 0.7])
            colorbar = figure.colorbar(first_scatter, cax=colorbar_axis)
            colorbar.set_label("Point density (KDE)")
            colorbar_state["axis"] = colorbar_axis

        figure.suptitle(
            f"Motility Density Scatter Plots (SORT:{source_name})",
            y=0.98,
            fontsize=14,
            fontweight="bold",
        )
        if source_buttons:
            _refresh_source_button_styles()
        figure.canvas.draw_idle()

    def _make_source_handler(source_name: str):
        """Build click handler for one source selector button."""

        def _handler(_: Any) -> None:
            state["source"] = str(source_name)
            _draw_density_for_source(str(source_name))

        return _handler

    _draw_density_for_source(str(state["source"]))
    if source_buttons:
        for source_name, button_data in source_buttons.items():
            button_data["button"].on_clicked(_make_source_handler(source_name))
        _refresh_source_button_styles()
    plt.show()

    casa["meta"]["last_visualization"] = {
        "type": "motility_density_scatter",
        "tracking_backend": "sort",
        "detection_method": str(state["source"]),
    }
    return casa
