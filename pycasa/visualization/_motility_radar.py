from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _import_matplotlib_for_visualization
from ..utils import _resolve_active_tracking_backend
from ..utils import _resolve_visualization_source

_RADAR_PARAMS = ["VCL", "VSL", "VAP", "LIN", "ALH", "WOB", "STR", "MAD"]
_RADAR_MAX_VALS = {
    "VCL": 60.0,
    "VSL": 36.0,
    "VAP": 36.0,
    "LIN": 1.0,
    "ALH": 5.0,
    "WOB": 1.0,
    "STR": 1.0,
    "MAD": 180.0,
}


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


def _collect_radar_means(method_data: dict[str, Any]) -> list[float]:
    """Compute per-metric means across all tracks/windows for radar plotting."""
    means: list[float] = []
    for param in _RADAR_PARAMS:
        values: list[float] = []
        for entry in method_data.values():
            if not isinstance(entry, dict):
                continue
            param_values = entry.get(param, [])
            if isinstance(param_values, list):
                for value in param_values:
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        continue
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        means.append(float(arr.mean()) if arr.size else 0.0)
    return means


def motility_radar(
    casa: dict[str, Any],
    axis: Any = None,
    show_legend: bool = True,
    show_text: bool = True,
) -> dict[str, Any]:
    """Render a radar-chart summary for active standard motility parameters.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        axis (Any, optional):
            Existing matplotlib polar axis. If ``None``, this function creates
            a new figure and axis.
        show_legend (bool, optional):
            Whether to display the legend.
        show_text (bool, optional):
            Whether to display numeric metric means below the chart.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with metadata written to
            ``casa["meta"]["last_visualization"]``.

    Raises:
        RuntimeError:
            If active standard motility results are missing.
        ImportError:
            If ``matplotlib`` is unavailable.

    Notes:
        - Uses active source data resolved from
          ``casa["motility"]["standard_motility_parameters"][source]``.
        - Metric values are normalized against fixed legacy reference maxima
          for radar scaling.

    Examples:
        >>> session = session.motility.standard_motility_parameters()
        >>> session = session.visualization.motility_radar()
    """
    casa = _ensure_casa(casa)
    plt = _import_matplotlib_for_visualization("motility-radar")
    mpl_widgets = _ensure_import("matplotlib.widgets", pip_name="matplotlib")
    Button = mpl_widgets.Button

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

    source_order = sorted(source_map.keys(), key=lambda value: (value != "groundtruth", value))
    preferred_source = _resolve_visualization_source(casa)
    selected_source = (
        preferred_source
        if preferred_source in source_map
        else source_order[0]
    )

    tracks_root = casa.get("tracks", {})
    active_backend = _resolve_active_tracking_backend(tracks_root) or "sort"
    backend_label = str(active_backend).upper()

    state = {"source": str(selected_source)}
    source_buttons: dict[str, dict[str, Any]] = {}

    if axis is None:
        figure = plt.figure(figsize=(7.2, 6.2))
        plot_axis = figure.add_axes([0.08, 0.12, 0.68, 0.78], polar=True)
        if len(source_order) > 1:
            base_y = 0.78
            row_gap = 0.09
            for row_idx, source_name in enumerate(source_order):
                button_ax = figure.add_axes([0.80, base_y - row_idx * row_gap, 0.17, 0.06])
                button = Button(
                    button_ax,
                    str(source_name),
                    color="white",
                    hovercolor="#efefef",
                )
                button_ax.set_xticks([])
                button_ax.set_yticks([])
                source_buttons[str(source_name)] = {"axis": button_ax, "button": button}
    else:
        plot_axis = axis

    def _refresh_source_button_styles() -> None:
        """Render source selector button styles for active/inactive states."""
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

    def _draw_radar_for_source(source_name: str) -> None:
        """Draw one radar summary for the selected source."""
        method_data = source_map.get(str(source_name), {})
        if not isinstance(method_data, dict):
            method_data = {}
        if not method_data:
            return

        means = _collect_radar_means(method_data)
        angle_count = len(_RADAR_PARAMS)
        angles = np.linspace(0, 2 * np.pi, angle_count, endpoint=False).tolist()
        angles += angles[:1]
        normalized = [means[idx] / _RADAR_MAX_VALS[_RADAR_PARAMS[idx]] for idx in range(angle_count)]
        closed_values = normalized + normalized[:1]

        plot_axis.clear()
        color = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])[0]
        label = f"{backend_label} ({source_name})"
        plot_axis.plot(angles, closed_values, color=color, linewidth=3.0, label=label)
        plot_axis.fill(angles, closed_values, color=color, alpha=0.2)

        plot_axis.set_ylim(0, 1)
        plot_axis.set_yticks([0.25, 0.5, 0.75, 1.0])
        plot_axis.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=22)
        split_labels = [f"{param}\n({int(_RADAR_MAX_VALS[param])})" for param in _RADAR_PARAMS]
        plot_axis.set_xticks(angles[:-1])
        plot_axis.set_xticklabels(split_labels, fontsize=22)
        plot_axis.set_title(
            f"Average Motility Parameters ({backend_label}:{source_name})",
            va="bottom",
            fontsize=12,
        )

        if show_legend:
            plot_axis.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=7)

        if show_text:
            text_lines = [f"{param}: {means[idx]:.1f}" for idx, param in enumerate(_RADAR_PARAMS)]
            plot_axis.text(
                0.5,
                -0.16,
                "\n".join(text_lines),
                transform=plot_axis.transAxes,
                ha="center",
                va="top",
                fontsize=24,
                family="monospace",
            )

        if source_buttons:
            _refresh_source_button_styles()
        if axis is None:
            figure.canvas.draw_idle()

    def _make_source_handler(source_name: str):
        """Build click handler for one source selector button."""

        def _handler(_: Any) -> None:
            state["source"] = str(source_name)
            _draw_radar_for_source(str(source_name))

        return _handler

    _draw_radar_for_source(str(state["source"]))
    if source_buttons:
        for source_name, button_data in source_buttons.items():
            button_data["button"].on_clicked(_make_source_handler(source_name))
        _refresh_source_button_styles()

    if axis is None:
        plt.show()

    casa["meta"]["last_visualization"] = {
        "type": "motility_radar",
        "tracking_backend": active_backend,
        "detection_method": str(state["source"]),
        "show_legend": bool(show_legend),
        "show_text": bool(show_text),
        "axis_provided": axis is not None,
    }
    return casa
