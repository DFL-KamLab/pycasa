from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..motility._standard_motility_parameters import _coerce_track_points
from ..motility._standard_motility_parameters import _compute_segment_motility
from ..motility._standard_motility_parameters import _ensure_um_per_px
from ..motility._standard_motility_parameters import _resolve_video_size
from ..utils import _ensure_import
from ..utils import _import_matplotlib_for_visualization
from ..utils import _resolve_active_sort_source_name
from ..utils import _resolve_sort_track_sources
from ..utils import _resolve_visualization_source
from ..utils import _GROUNDTRUTH_TRACKS_KEY


def _track_sort_key(track_id: str) -> tuple[int, int, str]:
    """Sort track IDs naturally when shaped like ``t1``, ``t2``, ..."""
    normalized = str(track_id).strip()
    suffix = normalized[1:] if normalized.startswith("t") else normalized
    if suffix.isdigit():
        return (0, int(suffix), normalized)
    return (1, 0, normalized)


def _is_nested_source_motility_map(candidate: dict[str, Any]) -> bool:
    """Return ``True`` for shape ``source -> track_id -> metric_dict``."""
    for source_value in candidate.values():
        if not isinstance(source_value, dict):
            continue
        if any(isinstance(track_value, dict) for track_value in source_value.values()):
            return True
    return False


def interactive_motility_calculator(
    casa: dict[str, Any],
    frame_rate: float | None = None,
    smoothing_window: int = 5,
) -> dict[str, Any]:
    """Open an interactive motility-parameter explorer for active SORT tracks.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        frame_rate (float | None, optional):
            FPS override used for segment metrics. If ``None``, uses
            ``casa["meta"]["sampling_rate"]`` when available, otherwise ``30``.
        smoothing_window (int, optional):
            Smoothing window used by VAP/ALH calculations in the segment
            metric preview.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with metadata written to
            ``casa["meta"]["last_visualization"]``.

    Raises:
        ValueError:
            If video width/height metadata cannot be resolved.
            If ``casa["meta"]["um_per_px"]`` is present but invalid.
        RuntimeError:
            If active SORT tracks are missing or no track has enough points
            for interactive exploration.
        ImportError:
            If ``matplotlib`` is unavailable.

    Notes:
        - Uses active tracking data resolved from
          ``casa["tracks"]["sort"][source]``.
        - Track selector uses a scroll-style index slider and compact track list.
        - Window and step sliders control highlighted segment navigation.
        - Right panel shows 8 metric history tiles (VCL, VSL, VAP, LIN, ALH,
          WOB, STR, MAD) with current-point highlight and global mean/std
          from ``motility.standard_motility_parameters`` when available.

    Examples:
        >>> session = session.tracking.sort()
        >>> session = session.visualization.interactive_motility_calculator()
    """
    casa = _ensure_casa(casa)
    plt = _import_matplotlib_for_visualization("interactive-motility-calculator")
    mpl_widgets = _ensure_import("matplotlib.widgets", pip_name="matplotlib")
    mpl_patches = _ensure_import("matplotlib.patches", pip_name="matplotlib")
    Button = mpl_widgets.Button
    Slider = mpl_widgets.Slider
    Rectangle = mpl_patches.Rectangle

    fps = float(frame_rate or casa.get("meta", {}).get("sampling_rate") or 30.0)
    if fps <= 0:
        fps = 30.0
    um_per_px_raw = casa.get("meta", {}).get("um_per_px")
    scale: float | None = None
    if um_per_px_raw is not None:
        scale = _ensure_um_per_px(um_per_px_raw)

    width, height = _resolve_video_size(casa)
    if width <= 0 or height <= 0:
        raise ValueError(
            "Video width/height metadata is required for "
            "`interactive_motility_calculator` visualization."
        )

    detections_root = casa.get("detections", {})
    if not isinstance(detections_root, dict):
        detections_root = {}
    last_tracking = casa.get("meta", {}).get("last_tracking")
    if not isinstance(last_tracking, dict):
        last_tracking = {}

    tracks_root = casa.get("tracks", {})
    if not isinstance(tracks_root, dict):
        tracks_root = {}
    track_sources = dict(_resolve_sort_track_sources(tracks_root))
    # Imported ground-truth tracks are not a backend, so surface them here too.
    imported_gt_tracks = tracks_root.get(_GROUNDTRUTH_TRACKS_KEY)
    if isinstance(imported_gt_tracks, dict) and imported_gt_tracks:
        track_sources[_GROUNDTRUTH_TRACKS_KEY] = imported_gt_tracks
    if not track_sources:
        raise RuntimeError(
            "No tracks found. Run tracking first, or load imported "
            "ground-truth tracks via load_video(groundtruth_tracks_path=...)."
        )

    parsed_tracks_by_source: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    eligible_ids_by_source: dict[str, list[str]] = {}
    for source_name, source_tracks in track_sources.items():
        if not isinstance(source_tracks, dict):
            continue
        parsed_tracks: dict[str, dict[int, np.ndarray]] = {}
        for track_id, track_data in source_tracks.items():
            if not isinstance(track_data, dict):
                continue
            parsed = _coerce_track_points(track_data, width=width, height=height)
            if parsed:
                parsed_tracks[str(track_id)] = parsed
        eligible_ids = [
            track_id
            for track_id, points in parsed_tracks.items()
            if len(points) >= 5
        ]
        if not eligible_ids:
            continue
        eligible_ids = sorted(eligible_ids, key=_track_sort_key)
        parsed_tracks_by_source[str(source_name)] = parsed_tracks
        eligible_ids_by_source[str(source_name)] = eligible_ids

    if not eligible_ids_by_source:
        raise RuntimeError("No eligible tracks found in active sort output (>=5 points).")

    smooth_init = max(1, min(50, int(smoothing_window)))
    source_order = sorted(
        eligible_ids_by_source.keys(),
        key=lambda value: (value != "groundtruth", value),
    )
    preferred_source = (
        _resolve_active_sort_source_name(
            tracks_root,
            detections_root=detections_root,
            meta_last_tracking=last_tracking,
        )
        or _resolve_visualization_source(casa)
    )
    selected_source = (
        preferred_source
        if preferred_source in eligible_ids_by_source
        else source_order[0]
    )

    visible_count = 24
    state = {
        "source": selected_source,
        "parsed_tracks": parsed_tracks_by_source[selected_source],
        "eligible_ids": eligible_ids_by_source[selected_source],
        "track_count": len(eligible_ids_by_source[selected_source]),
        "max_list_start": max(0, len(eligible_ids_by_source[selected_source]) - visible_count),
        "track_idx": 0,
        "track_id": eligible_ids_by_source[selected_source][0],
        "start": 0,
        "list_start": 0,
    }
    metric_names = ("VCL", "VSL", "VAP", "LIN", "ALH", "WOB", "STR", "MAD")
    history_state: dict[str, Any] = {
        "context": None,
        "series": {metric: [] for metric in metric_names},
        "last_start": None,
    }
    fallback_stats_cache: dict[tuple[str, int, int, float], dict[str, dict[str, float]]] = {}

    fig = plt.figure(figsize=(13.0, 7.2))
    ax_path = fig.add_axes([0.22, 0.20, 0.48, 0.75])
    ax_track_slider = fig.add_axes([0.02, 0.20, 0.02, 0.75])
    ax_track_list = fig.add_axes([0.05, 0.20, 0.14, 0.75], frameon=False)
    slider_y = 0.105
    slider_h = 0.03
    ax_window = fig.add_axes([0.23, slider_y, 0.17, slider_h])
    ax_step = fig.add_axes([0.52, slider_y, 0.17, slider_h])
    ax_window_prev = fig.add_axes([0.75, 0.05, 0.11, 0.04])
    ax_window_next = fig.add_axes([0.87, 0.05, 0.10, 0.04])
    ax_track_prev = fig.add_axes([0.01, 0.05, 0.10, 0.04])
    ax_track_next = fig.add_axes([0.12, 0.05, 0.10, 0.04])
    ax_metrics_container = fig.add_axes([0.73, 0.20, 0.25, 0.75], frameon=False)
    ax_metrics_container.set_xticks([])
    ax_metrics_container.set_yticks([])
    ax_metrics_container.axis("off")
    metric_axes: dict[str, Any] = {}
    tile_rows = 4
    tile_cols = 2
    tile_gap_x = 0.014
    tile_gap_y = 0.018
    panel_left = 0.73
    panel_bottom = 0.20
    panel_width = 0.25
    panel_height = 0.75
    tile_width = (panel_width - tile_gap_x * (tile_cols - 1)) / tile_cols
    tile_height = (panel_height - tile_gap_y * (tile_rows - 1)) / tile_rows
    for metric_idx, metric_name in enumerate(metric_names):
        row = metric_idx // tile_cols
        col = metric_idx % tile_cols
        tile_x = panel_left + col * (tile_width + tile_gap_x)
        tile_y = panel_bottom + panel_height - (row + 1) * tile_height - row * tile_gap_y
        metric_axes[metric_name] = fig.add_axes([tile_x, tile_y, tile_width, tile_height])

    slider_track_list = Slider(
        ax_track_slider,
        "",
        0,
        max(1, int(state["max_list_start"])),
        valinit=int(state["max_list_start"]) if int(state["max_list_start"]) > 0 else 0,
        valstep=1,
        orientation="vertical",
        color="#d9d9d9",
    )
    slider_window = Slider(ax_window, "", 1, 50, valinit=smooth_init, valstep=1)
    slider_step = Slider(ax_step, "", 1, 50, valinit=1, valstep=1)
    button_window_prev = Button(ax_window_prev, "Prev Window")
    button_window_next = Button(ax_window_next, "Next Window")
    button_track_prev = Button(ax_track_prev, "Previous Track")
    button_track_next = Button(ax_track_next, "Next Track")
    source_buttons: dict[str, dict[str, Any]] = {}
    if len(source_order) > 1:
        source_button_y = 0.955
        source_button_h = 0.035
        source_button_w = min(0.16, (0.46 - 0.01 * (len(source_order) - 1)) / len(source_order))
        source_button_x = 0.23
        for source_name in source_order:
            source_ax = fig.add_axes([source_button_x, source_button_y, source_button_w, source_button_h])
            source_button = Button(
                source_ax,
                str(source_name),
                color="white",
                hovercolor="#efefef",
            )
            source_ax.set_xticks([])
            source_ax.set_yticks([])
            source_buttons[str(source_name)] = {
                "axis": source_ax,
                "button": source_button,
            }
            source_button_x += source_button_w + 0.01
    button_track_prev.label.set_fontsize(11)
    button_track_next.label.set_fontsize(11)
    slider_window.label.set_visible(False)
    slider_step.label.set_visible(False)
    slider_window.valtext.set_visible(False)
    slider_step.valtext.set_visible(False)
    ax_window.text(
        -0.10,
        0.5,
        "Window",
        transform=ax_window.transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )
    ax_step.text(
        -0.10,
        0.5,
        "Step",
        transform=ax_step.transAxes,
        ha="right",
        va="center",
        fontsize=10,
    )
    window_value_text = ax_window.text(
        1.01,
        0.5,
        str(int(slider_window.val)),
        transform=ax_window.transAxes,
        ha="left",
        va="center",
        fontsize=10,
    )
    step_value_text = ax_step.text(
        1.01,
        0.5,
        str(int(slider_step.val)),
        transform=ax_step.transAxes,
        ha="left",
        va="center",
        fontsize=10,
    )

    ax_track_list.axis("off")

    slider_track_list.valtext.set_visible(False)
    slider_track_list.label.set_visible(False)
    ax_track_slider.set_facecolor("#efefef")
    ax_track_slider.set_xticks([])
    ax_track_slider.set_yticks([])
    if hasattr(slider_track_list, "poly"):
        slider_track_list.poly.set_alpha(1.0)
        slider_track_list.poly.set_facecolor("#d9d9d9")
        slider_track_list.poly.set_edgecolor("#d9d9d9")
    if hasattr(slider_track_list, "track"):
        slider_track_list.track.set_color("#c8c8c8")
    if hasattr(slider_track_list, "vline"):
        slider_track_list.vline.set_visible(False)
    if hasattr(slider_track_list, "hline"):
        slider_track_list.hline.set_visible(False)
    if hasattr(slider_track_list, "handle"):
        try:
            slider_track_list.handle.set_facecolor("white")
            slider_track_list.handle.set_edgecolor("#9b9b9b")
        except Exception:
            pass
    if int(state["max_list_start"]) == 0:
        slider_track_list.set_active(False)

    def _slider_value_from_list_start(list_start: int) -> int:
        """Map list-start offset to vertical slider value (top=first tracks)."""
        max_list_start = int(state["max_list_start"])
        if max_list_start <= 0:
            return 0
        return int(max(0, min(max_list_start, max_list_start - int(list_start))))

    def _list_start_from_slider_value(slider_value: float) -> int:
        """Map vertical slider value to list-start offset (top=first tracks)."""
        max_list_start = int(state["max_list_start"])
        if max_list_start <= 0:
            return 0
        bounded_value = int(max(0, min(max_list_start, int(slider_value))))
        return int(max(0, min(max_list_start, max_list_start - bounded_value)))

    def _sync_list_slider() -> None:
        """Keep list slider in sync when list start changes programmatically."""
        max_list_start = int(state["max_list_start"])
        if max_list_start == 0:
            slider_track_list.set_active(False)
            slider_track_list.eventson = False
            slider_track_list.set_val(0)
            slider_track_list.eventson = True
            return
        slider_track_list.set_active(True)
        slider_track_list.eventson = False
        slider_track_list.valmax = float(max(1, max_list_start))
        slider_track_list.set_val(_slider_value_from_list_start(state["list_start"]))
        slider_track_list.eventson = True

    def _ensure_selected_track_visible() -> None:
        """Adjust list window so selected track stays visible."""
        max_list_start = int(state["max_list_start"])
        if state["track_idx"] < state["list_start"]:
            state["list_start"] = state["track_idx"]
        elif state["track_idx"] >= state["list_start"] + visible_count:
            state["list_start"] = state["track_idx"] - visible_count + 1
        state["list_start"] = int(max(0, min(max_list_start, state["list_start"])))

    def _compute_total_motility_stats_from_session() -> dict[str, dict[str, float]] | None:
        """Compute global mean/std from ``motility.standard_motility_parameters``."""
        motility_root = casa.get("motility", {}).get("standard_motility_parameters", {})
        if not isinstance(motility_root, dict) or not motility_root:
            return None

        if not _is_nested_source_motility_map(motility_root):
            method_data = motility_root
        else:
            method_data = motility_root.get(str(state["source"]))
            if not isinstance(method_data, dict):
                method_data = motility_root.get("groundtruth")
            if not isinstance(method_data, dict):
                method_data = next(
                    (
                        value
                        for value in motility_root.values()
                        if isinstance(value, dict)
                    ),
                    {},
                )
        if not isinstance(method_data, dict) or not method_data:
            return None

        metric_values: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for track_metrics in method_data.values():
            if not isinstance(track_metrics, dict):
                continue
            for metric_name in metric_names:
                values = track_metrics.get(metric_name)
                if not isinstance(values, list):
                    continue
                for value in values:
                    try:
                        metric_values[metric_name].append(float(value))
                    except (TypeError, ValueError):
                        continue

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        has_any = False
        for metric_name in metric_names:
            arr = np.asarray(metric_values[metric_name], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                means[metric_name] = float("nan")
                stds[metric_name] = float("nan")
                continue
            has_any = True
            means[metric_name] = float(np.mean(arr))
            stds[metric_name] = float(np.std(arr, ddof=0))

        if not has_any:
            return None
        return {"mean": means, "std": stds}

    def _compute_population_stats(
        track_id: str,
        data: dict[int, np.ndarray],
        frame_ids: list[int],
        requested_window: int,
    ) -> dict[str, dict[str, float]]:
        """Compute per-metric mean/std over all possible windows for one context."""
        window_for_stats = int(max(1, min(requested_window, len(frame_ids))))
        context_key = (
            str(state["source"]),
            str(track_id),
            int(requested_window),
            int(max(1, int(smoothing_window))),
            float(round(fps, 6)),
        )
        cached = fallback_stats_cache.get(context_key)
        if cached is not None:
            return cached

        max_start = max(0, len(frame_ids) - window_for_stats)
        metric_values: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for start_idx in range(max_start + 1):
            segment_frames = frame_ids[start_idx : start_idx + window_for_stats]
            segment_data = {frame: data[frame] for frame in segment_frames}
            params = _compute_segment_motility(
                segment_data,
                fps=fps,
                smooth_w=min(window_for_stats, max(1, int(smoothing_window))),
            )
            if scale is not None:
                for key in ("VCL", "VSL", "VAP"):
                    params[key] *= scale
                params["ALH"] *= scale
            for metric_name in metric_names:
                value = float(params.get(metric_name, np.nan))
                metric_values[metric_name].append(value)

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for metric_name in metric_names:
            arr = np.asarray(metric_values[metric_name], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                means[metric_name] = float("nan")
                stds[metric_name] = float("nan")
            else:
                means[metric_name] = float(np.mean(arr))
                stds[metric_name] = float(np.std(arr, ddof=0))

        result = {"mean": means, "std": stds}
        fallback_stats_cache[context_key] = result
        return result

    def _reset_metric_history(track_id: str, requested_window: int) -> None:
        """Reset metric history for a new track/window context."""
        history_state["context"] = (
            str(state["source"]),
            str(track_id),
            int(requested_window),
        )
        history_state["series"] = {metric: [] for metric in metric_names}
        history_state["last_start"] = None

    def _append_metric_history_if_needed(window_start: int, params: dict[str, float]) -> None:
        """Append one history point per metric when window start changes."""
        if history_state["last_start"] == int(window_start):
            return
        for metric_name in metric_names:
            value = float(params.get(metric_name, np.nan))
            history_state["series"][metric_name].append(value)
        history_state["last_start"] = int(window_start)

    def _render_metric_tiles(population_stats: dict[str, dict[str, float]]) -> None:
        """Render metric history tiles with current-point highlight and mean/std."""
        blue_cmap = plt.get_cmap("Blues")
        for metric_name in metric_names:
            metric_ax = metric_axes[metric_name]
            metric_ax.clear()

            values = np.asarray(history_state["series"][metric_name], dtype=float)
            n_points = int(values.size)
            x_values = np.arange(1, n_points + 1, dtype=float) if n_points else np.asarray([], dtype=float)
            valid_mask = np.isfinite(values)

            if n_points > 1 and np.count_nonzero(valid_mask[:-1]) > 0:
                prev_valid_count = int(np.count_nonzero(valid_mask[:-1]))
                prev_colors = blue_cmap(np.linspace(0.35, 0.90, prev_valid_count))
                metric_ax.scatter(
                    x_values[:-1][valid_mask[:-1]],
                    values[:-1][valid_mask[:-1]],
                    s=26,
                    c=prev_colors,
                    edgecolors="none",
                    zorder=3,
                )

            if n_points > 1 and np.count_nonzero(valid_mask) >= 2:
                metric_ax.plot(
                    x_values[valid_mask],
                    values[valid_mask],
                    color="#94a3b8",
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=2,
                )

            if n_points > 0 and np.isfinite(values[-1]):
                metric_ax.scatter(
                    [x_values[-1]],
                    [values[-1]],
                    s=44,
                    c="red",
                    zorder=4,
                )
                metric_ax.annotate(
                    f"{values[-1]:.2f}",
                    (x_values[-1], values[-1]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                    zorder=5,
                )

            mean_value = float(population_stats["mean"].get(metric_name, np.nan))
            std_value = float(population_stats["std"].get(metric_name, np.nan))
            x_right = max(2.0, float(n_points) + 0.5)
            if np.isfinite(mean_value):
                if np.isfinite(std_value) and std_value > 0:
                    metric_ax.fill_between(
                        [0.5, x_right],
                        [mean_value - std_value, mean_value - std_value],
                        [mean_value + std_value, mean_value + std_value],
                        color="#fed7aa",
                        alpha=0.42,
                        zorder=0,
                    )
                metric_ax.axhline(
                    mean_value,
                    color="#c2410c",
                    linewidth=1.1,
                    linestyle="--",
                    zorder=1,
                )
                std_text = f"{std_value:.2f}" if np.isfinite(std_value) else "n/a"
                metric_ax.text(
                    0.02,
                    0.98,
                    f"mean(all windows)={mean_value:.2f}\nstd(all windows)={std_text}",
                    transform=metric_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#7c2d12",
                    bbox={
                        "boxstyle": "round,pad=0.24",
                        "facecolor": "#fff7ed",
                        "edgecolor": "#fdba74",
                        "linewidth": 0.8,
                        "alpha": 0.9,
                    },
                    zorder=6,
                )

            candidates: list[float] = []
            if np.any(valid_mask):
                candidates.extend(values[valid_mask].tolist())
            if np.isfinite(mean_value):
                candidates.append(mean_value)
                if np.isfinite(std_value) and std_value > 0:
                    candidates.append(mean_value - std_value)
                    candidates.append(mean_value + std_value)

            if candidates:
                y_min = float(min(candidates))
                y_max = float(max(candidates))
                if y_max <= y_min:
                    padding = max(0.2, abs(y_max) * 0.2 + 0.2)
                else:
                    padding = (y_max - y_min) * 0.22
                metric_ax.set_ylim(y_min - padding, y_max + padding)
            else:
                metric_ax.set_ylim(0.0, 1.0)

            metric_ax.set_xlim(0.5, x_right)
            metric_ax.set_title(metric_name, fontsize=12, pad=2, fontweight="bold")
            metric_ax.set_xticks([])
            metric_ax.set_yticks([])
            metric_ax.grid(alpha=0.2, linewidth=0.7)
            for spine in metric_ax.spines.values():
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.9)

    def _update(_: Any = None) -> None:
        """Redraw path and metric panel for current interactive selection."""
        ax_path.clear()
        ax_track_list.clear()
        ax_track_list.set_xlim(0, 1)
        ax_track_list.set_ylim(0, visible_count)
        ax_track_list.axis("off")

        track_id = str(state["track_id"])
        data = state["parsed_tracks"][track_id]
        track_count = int(state["track_count"])
        eligible_ids = list(state["eligible_ids"])
        max_list_start = int(state["max_list_start"])
        frame_ids = sorted(int(key) for key in data)
        points = np.array([data[frame] for frame in frame_ids], dtype=float)

        total_points = len(points)
        window = int(slider_window.val)
        start = max(0, min(state["start"], total_points - window))
        state["start"] = start
        end = start + window

        segment_frames = frame_ids[start:end]
        segment_points = points[start:end]
        segment_data = {frame: data[frame] for frame in segment_frames}

        ax_path.plot(points[:, 0], points[:, 1], "-o", c="blue", ms=3)
        ax_path.plot(segment_points[:, 0], segment_points[:, 1], "-o", c="red", ms=6)
        ax_path.grid(True)
        ax_path.set(
            xlim=(0, width),
            ylim=(0, height),
            aspect="equal",
            xlabel="X (px)",
            ylabel="Y (px)",
            title=(
                f"SORT ({state['source']}) track {track_id}: "
                f"pts {total_points}, window {segment_frames[0]}-{segment_frames[-1]}"
            ),
        )

        current_params = _compute_segment_motility(
            segment_data,
            fps=fps,
            smooth_w=min(window, max(1, int(smoothing_window))),
        )
        if scale is not None:
            for key in ("VCL", "VSL", "VAP"):
                current_params[key] *= scale
            current_params["ALH"] *= scale
        history_context = (str(state["source"]), str(track_id), int(window))
        if history_state["context"] != history_context:
            _reset_metric_history(track_id=track_id, requested_window=window)
        _append_metric_history_if_needed(window_start=start, params=current_params)
        total_stats = _compute_total_motility_stats_from_session()
        if total_stats is None:
            total_stats = _compute_population_stats(
                track_id=track_id,
                data=data,
                frame_ids=frame_ids,
                requested_window=window,
            )
        _render_metric_tiles(total_stats)

        ax_track_list.text(
            0.0,
            1.03,
            f"Track {state['track_idx'] + 1}/{track_count}",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            fontweight="bold",
            transform=ax_track_list.transAxes,
        )
        state["list_start"] = int(max(0, min(max_list_start, state["list_start"])))
        _sync_list_slider()

        visible_ids = eligible_ids[state["list_start"] : state["list_start"] + visible_count]
        for row_idx, visible_track_id in enumerate(visible_ids):
            y = visible_count - row_idx - 1
            absolute_index = state["list_start"] + row_idx
            is_selected = absolute_index == state["track_idx"]

            row_face = "#d7dce3" if is_selected else "none"
            row_edge = "#b8bec7" if is_selected else "none"
            ax_track_list.add_patch(
                Rectangle(
                    (0.0, y),
                    1.0,
                    1.0,
                    facecolor=row_face,
                    edgecolor=row_edge,
                    linewidth=0.6 if is_selected else 0.0,
                )
            )
            ax_track_list.text(
                0.05,
                y + 0.5,
                visible_track_id,
                va="center",
                ha="left",
                family="monospace",
                fontsize=9,
                color="black",
            )
        fig.canvas.draw_idle()

    def _set_track_index(new_index: int) -> None:
        track_count = int(state["track_count"])
        eligible_ids = list(state["eligible_ids"])
        bounded = int(max(0, min(track_count - 1, int(new_index))))
        state["track_idx"] = bounded
        state["track_id"] = eligible_ids[bounded]
        state["start"] = 0
        _ensure_selected_track_visible()
        _update()

    def _on_list_slider_change(value: float) -> None:
        max_list_start = int(state["max_list_start"])
        if max_list_start == 0:
            return
        state["list_start"] = _list_start_from_slider_value(value)
        _update()

    def _on_track_list_click(event: Any) -> None:
        """Select a track by clicking a visible row in the list panel."""
        if event.inaxes is not ax_track_list or event.ydata is None:
            return
        if event.ydata < 0 or event.ydata >= visible_count:
            return
        row_idx = int(visible_count - 1 - int(event.ydata))
        if row_idx < 0:
            return
        absolute_index = state["list_start"] + row_idx
        track_count = int(state["track_count"])
        if absolute_index < 0 or absolute_index >= track_count:
            return
        _set_track_index(absolute_index)

    def _on_track_list_scroll(event: Any) -> None:
        """Scroll visible track rows with mouse wheel over list panel."""
        max_list_start = int(state["max_list_start"])
        if event.inaxes is not ax_track_list or max_list_start == 0:
            return
        if event.button == "up":
            new_start = state["list_start"] - 1
        elif event.button == "down":
            new_start = state["list_start"] + 1
        else:
            return
        slider_track_list.set_val(_slider_value_from_list_start(new_start))

    def _on_track_prev(_: Any) -> None:
        _set_track_index(state["track_idx"] - 1)

    def _on_track_next(_: Any) -> None:
        _set_track_index(state["track_idx"] + 1)

    def _on_window_change(_: float) -> None:
        window_value_text.set_text(str(int(slider_window.val)))
        state["start"] = 0
        _update()

    def _on_step_change(_: float) -> None:
        step_value_text.set_text(str(int(slider_step.val)))
        _update()

    def _on_window_prev(_: Any) -> None:
        state["start"] = state["start"] - int(slider_step.val)
        _update()

    def _on_window_next(_: Any) -> None:
        state["start"] = state["start"] + int(slider_step.val)
        _update()

    def _refresh_source_button_styles() -> None:
        """Render source selector buttons with active/inactive styles."""
        for source_name, button_data in source_buttons.items():
            axis = button_data["axis"]
            button = button_data["button"]
            is_active = str(source_name) == str(state["source"])
            if is_active:
                base_color = "black"
                hover_color = "#2f2f2f"
                text_color = "white"
            else:
                base_color = "white"
                hover_color = "#efefef"
                text_color = "black"

            button.color = base_color
            button.hovercolor = hover_color
            axis.set_facecolor(base_color)
            button.label.set_color(text_color)
            button.label.set_fontsize(9)
            button.label.set_fontweight("bold")
            for spine in axis.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

    def _set_source(source_name: str) -> None:
        """Switch active source and reset selection context."""
        source_key = str(source_name)
        if source_key not in eligible_ids_by_source:
            return
        if source_key == str(state["source"]):
            return

        state["source"] = source_key
        state["parsed_tracks"] = parsed_tracks_by_source[source_key]
        state["eligible_ids"] = eligible_ids_by_source[source_key]
        state["track_count"] = len(eligible_ids_by_source[source_key])
        state["max_list_start"] = max(0, int(state["track_count"]) - visible_count)
        state["track_idx"] = 0
        state["track_id"] = state["eligible_ids"][0]
        state["start"] = 0
        state["list_start"] = 0
        history_state["context"] = None
        history_state["last_start"] = None
        _sync_list_slider()
        _refresh_source_button_styles()
        _update()

    def _make_source_handler(source_name: str):
        """Build click handler for one source selector button."""

        def _handler(_: Any) -> None:
            _set_source(source_name)

        return _handler

    slider_track_list.on_changed(_on_list_slider_change)
    slider_window.on_changed(_on_window_change)
    slider_step.on_changed(_on_step_change)
    button_window_prev.on_clicked(_on_window_prev)
    button_window_next.on_clicked(_on_window_next)
    button_track_prev.on_clicked(_on_track_prev)
    button_track_next.on_clicked(_on_track_next)
    fig.canvas.mpl_connect("button_press_event", _on_track_list_click)
    fig.canvas.mpl_connect("scroll_event", _on_track_list_scroll)
    if source_buttons:
        for source_name, button_data in source_buttons.items():
            button_data["button"].on_clicked(_make_source_handler(source_name))
        _refresh_source_button_styles()

    _update()
    plt.show()

    casa["meta"]["last_visualization"] = {
        "type": "interactive_motility_calculator",
        "tracking_backend": "sort",
        "detection_method": str(state["source"]),
        "frame_rate": frame_rate,
        "smoothing_window": int(smoothing_window),
    }
    return casa
