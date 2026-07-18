from pathlib import Path
from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _import_matplotlib_for_visualization
from ..utils import _parse_image_types
from ..utils import _prepare_frame_for_display
from ..utils import _resolve_active_predicted_detection_method
from ..utils import _resolve_active_sort_tracks
from ..utils import _resolve_active_tracking_backend
from ..utils import _resolve_sort_track_sources
from ..utils import _resolve_frame_entries
from ..utils import _GROUNDTRUTH_TRACKS_KEY


def timelapse(
    casa: dict[str, Any],
    video_type: str = "original",
    image_type: str | None = None,
    show_detections: bool = True,
    show_tracks: bool = False,
    show_groundtruth: bool = True,
    show_track_ids: bool = False,
    detection_color: str | None = None,
    groundtruth_color: str | None = None,
    track_colors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open an interactive time-lapse viewer for selected video representations.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing at least one displayable video under
            ``casa['video']``.
        video_type (str | Iterable[str], optional):
            Video representation(s) to show. Accepts single value or combined
            values separated by ``+`` or ``,``. Supported names:
            ``original``, ``grayscale``/``gray``, ``normalized``,
            ``binarized``/``binary``, ``moving_cells``.
        image_type (str | None, optional):
            Deprecated compatibility alias for ``video_type``. If provided, it
            overrides ``video_type``.
        show_detections (bool, optional):
            Initial visibility state for active detection overlays.
        show_tracks (bool, optional):
            Initial visibility state for tracks.
        show_groundtruth (bool, optional):
            Initial visibility state for groundtruth detections.
        show_track_ids (bool, optional):
            Whether to annotate each track head with track ID text. This can be
            expensive for large track sets.
        detection_color (str | None, optional):
            Matplotlib color used for predicted-detection bounding boxes.
            ``None`` keeps the default (``"#ff4d4d"``).
        groundtruth_color (str | None, optional):
            Matplotlib color used for groundtruth bounding boxes. ``None``
            keeps the default (``"lime"``).
        track_colors (dict[str, Any] | None, optional):
            Per-source color override for track lines. Keys may be the literal
            source name (e.g. ``"groundtruth"``, ``"yolo"``) or a role alias:
            ``"groundtruth"`` for the groundtruth source and ``"detection"``
            for the active predicted-detection source. Values are any valid
            matplotlib color.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with viewer metadata written to
            ``casa['meta']['last_visualization']``.

    Raises:
        ValueError:
            If requested video layers are unavailable or frame data is invalid.
        ImportError:
            If ``matplotlib`` is unavailable.

    Notes:
        Controls:
        - Slider: scrub frames.
        - Play/Pause button: animate frames at session sampling rate.
        - Right overlay toggles: show/hide detections/tracks/groundtruth.
        - Keyboard: left/right arrows step frames, space toggles play.
        Overlay rendering:
        - Detections and groundtruth are drawn as bounding boxes.
        - Tracks are drawn as trajectory lines for unobstructed viewing.
        - Track sources include any active backend sources plus imported
          ground-truth tracks (``casa["tracks"]["groundtruth_tracks"]``), each
          with its own toggle. Enable them with ``show_tracks=True`` or the
          per-source toggle button.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.Casa()
        >>> session = session.visualization.timelapse(
        ...     detection_color="blue",
        ...     groundtruth_color="red",
        ...     track_colors={"groundtruth": "yellow", "detection": "orange"},
        ... )
    """
    casa = _ensure_casa(casa)

    plt = _import_matplotlib_for_visualization("timelapse")
    mpl_collections = _ensure_import("matplotlib.collections", pip_name="matplotlib")
    mpl_colors_mod = _ensure_import("matplotlib.colors", pip_name="matplotlib")
    mpl_lines = _ensure_import("matplotlib.lines", pip_name="matplotlib")
    mpl_widgets = _ensure_import("matplotlib.widgets", pip_name="matplotlib")
    LineCollection = mpl_collections.LineCollection
    Line2D = mpl_lines.Line2D
    Button = mpl_widgets.Button
    Slider = mpl_widgets.Slider

    def _to_rgba_array(color_value: Any) -> np.ndarray:
        """Convert any matplotlib color to a ``float32`` RGBA numpy array."""
        rgba = mpl_colors_mod.to_rgba(color_value)
        return np.asarray(rgba, dtype=np.float32)

    user_track_colors: dict[str, Any] = (
        dict(track_colors) if isinstance(track_colors, dict) else {}
    )
    groundtruth_box_color = groundtruth_color if groundtruth_color is not None else "lime"
    detection_box_color = detection_color if detection_color is not None else "#ff4d4d"

    selected_video_type = image_type if image_type is not None else video_type
    image_keys = _parse_image_types(selected_video_type)
    videos = [_get_image_video(casa, key) for key in image_keys]
    frame_counts = [int(video_data.shape[0]) for video_data in videos]
    if not frame_counts or min(frame_counts) <= 0:
        raise ValueError("No frames are available for visualization.")
    frame_count = min(frame_counts)

    first_frame = videos[0][0]
    display_frame, _ = _prepare_frame_for_display(first_frame)
    if display_frame.ndim != 2 and display_frame.ndim != 3:
        raise ValueError("Unsupported frame shape in selected videos.")
    height, width = int(display_frame.shape[0]), int(display_frame.shape[1])

    initial_frame = int(casa.get("video", {}).get("initial_frame", 0))
    sampling_rate = float(casa.get("meta", {}).get("sampling_rate") or 30.0)
    if sampling_rate <= 0:
        sampling_rate = 30.0

    detections_root = casa.get("detections", {})
    if not isinstance(detections_root, dict):
        detections_root = {}
    selected_detection_method = (
        _resolve_active_predicted_detection_method(detections_root)
        or "groundtruth"
    )
    last_tracking = casa.get("meta", {}).get("last_tracking")
    if not isinstance(last_tracking, dict):
        last_tracking = {}
    selected_detections = detections_root.get(selected_detection_method, {})
    if not isinstance(selected_detections, dict):
        selected_detections = {}
    groundtruth = detections_root.get("groundtruth", {})
    if not isinstance(groundtruth, dict):
        groundtruth = {}
    tracks_root = casa.get("tracks", {})
    if not isinstance(tracks_root, dict):
        tracks_root = {}
    active_tracking_backend = _resolve_active_tracking_backend(tracks_root) or "sort"
    track_sources = dict(_resolve_sort_track_sources(tracks_root))
    # Imported ground-truth tracks live at a reserved top-level key (not a
    # backend), so surface them here as their own selectable track source.
    imported_gt_tracks = tracks_root.get(_GROUNDTRUTH_TRACKS_KEY)
    if isinstance(imported_gt_tracks, dict) and imported_gt_tracks:
        track_sources[_GROUNDTRUTH_TRACKS_KEY] = imported_gt_tracks
    ordered_track_sources = sorted(
        track_sources.keys(),
        key=lambda value: (value != "groundtruth", value),
    )
    # Backward-compatible fallback in case legacy/flat state is encountered.
    if not ordered_track_sources:
        fallback_tracks = _resolve_active_sort_tracks(
            tracks_root,
            detections_root=detections_root,
            meta_last_tracking=last_tracking,
        )
        if fallback_tracks:
            track_sources = {"groundtruth": fallback_tracks}
            ordered_track_sources = ["groundtruth"]

    track_cache_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_name in ordered_track_sources:
        source_tracks = track_sources.get(source_name, {})
        if isinstance(source_tracks, dict):
            track_cache_by_source[source_name] = _build_track_cache(
                source_tracks,
                width=width,
                height=height,
            )
        else:
            track_cache_by_source[source_name] = []

    has_detection_overlay = bool(
        selected_detection_method != "groundtruth" and _has_overlay_data(selected_detections)
    )
    has_groundtruth_overlay = bool(_has_overlay_data(groundtruth))
    available_track_sources = [
        source_name
        for source_name in ordered_track_sources
        if track_cache_by_source.get(source_name)
    ]
    has_track_overlay = bool(available_track_sources)

    detection_source_label = _format_overlay_source_label(
        str(selected_detection_method), fallback="detections"
    )
    track_source_labels = {
        source_name: (
            "groundtruth_tracks"
            if source_name == _GROUNDTRUTH_TRACKS_KEY
            else _format_overlay_source_label(
                f"{active_tracking_backend}:{source_name}",
                fallback=active_tracking_backend,
            )
        )
        for source_name in available_track_sources
    }

    fig, axes = plt.subplots(1, len(videos), figsize=(5.5 * len(videos), 5.5))
    if len(videos) == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.20, right=0.82)

    if has_track_overlay:
        source_base_colors = {
            "groundtruth": np.asarray([0.20, 0.70, 1.00, 1.00], dtype=np.float32),
        }
        fallback_palette = [
            np.asarray([1.00, 0.55, 0.20, 1.00], dtype=np.float32),
            np.asarray([0.75, 0.45, 1.00, 1.00], dtype=np.float32),
            np.asarray([1.00, 0.35, 0.55, 1.00], dtype=np.float32),
            np.asarray([0.20, 0.85, 0.70, 1.00], dtype=np.float32),
        ]
        fallback_idx = 0
        for source_name in available_track_sources:
            cache_rows = track_cache_by_source.get(source_name, [])
            if not cache_rows:
                continue

            user_color: Any = None
            if source_name in user_track_colors:
                user_color = user_track_colors[source_name]
            elif source_name == "groundtruth" and "groundtruth" in user_track_colors:
                user_color = user_track_colors["groundtruth"]
            elif source_name != "groundtruth" and "detection" in user_track_colors:
                user_color = user_track_colors["detection"]

            if user_color is not None:
                base_color = _to_rgba_array(user_color)
            elif source_name in source_base_colors:
                base_color = source_base_colors[source_name]
            else:
                base_color = fallback_palette[fallback_idx % len(fallback_palette)]
                fallback_idx += 1
            for track_data in cache_rows:
                track_data["color"] = base_color

    slider_ax = fig.add_axes([0.12, 0.10, 0.52, 0.04])
    play_ax = fig.add_axes([0.72, 0.10, 0.10, 0.04])

    slider = Slider(slider_ax, "Frame", 0, frame_count - 1, valinit=0, valstep=1)
    slider.valtext.set_visible(False)
    frame_value_text = fig.text(
        0.675,
        0.115,
        str(int(slider.val)),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
    )
    play_button = Button(play_ax, "Play")

    state = {
        "index": 0,
        "playing": False,
        "show_detections": bool(show_detections and has_detection_overlay),
        "show_groundtruth": bool(show_groundtruth and has_groundtruth_overlay),
        "show_tracks_by_source": {
            source_name: bool(show_tracks)
            for source_name in available_track_sources
        },
    }

    empty_colors = np.empty((0, 4), dtype=np.float32)

    axis_contexts: list[dict[str, Any]] = []
    for axis, video_data, label in zip(axes, videos, image_keys):
        frame0 = video_data[0]
        shown0, use_gray0 = _prepare_frame_for_display(frame0)
        if use_gray0:
            image_artist = axis.imshow(shown0, cmap="gray")
        else:
            image_artist = axis.imshow(shown0)
        axis.set_title(f"{label} | frame {initial_frame}")
        axis.axis("off")

        groundtruth_box_collection = LineCollection(
            [], linewidths=1.5, colors=groundtruth_box_color, alpha=0.85
        )
        det_box_collection = LineCollection(
            [], linewidths=1.5, colors=detection_box_color, alpha=0.85
        )
        axis.add_collection(groundtruth_box_collection)
        axis.add_collection(det_box_collection)

        track_line_collection = None
        track_text_artists: list[Any] = []
        if has_track_overlay:
            track_line_collection = LineCollection([], linewidths=1.5, alpha=0.65)
            axis.add_collection(track_line_collection)

        axis_contexts.append(
            {
                "axis": axis,
                "video_data": video_data,
                "label": label,
                "image_artist": image_artist,
                "groundtruth_box_collection": groundtruth_box_collection,
                "det_box_collection": det_box_collection,
                "track_line_collection": track_line_collection,
                "track_text_artists": track_text_artists,
            }
        )

    legend_ref: dict[str, Any] = {"obj": None}

    def _update_legend() -> None:
        """Refresh legend entries to match current overlay visibility state."""
        if legend_ref["obj"] is not None:
            legend_ref["obj"].remove()
            legend_ref["obj"] = None

        legend_handles: list[Line2D] = []
        if state["show_groundtruth"] and has_groundtruth_overlay:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor=groundtruth_box_color,
                    markeredgewidth=1.2,
                    markersize=7,
                    label="groundtruth",
                )
            )
        if state["show_detections"] and has_detection_overlay:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor=detection_box_color,
                    markeredgewidth=1.2,
                    markersize=7,
                    label=f"detections ({detection_source_label})",
                )
            )
        if has_track_overlay:
            for source_name in available_track_sources:
                if not bool(state["show_tracks_by_source"].get(source_name, False)):
                    continue
                source_cache = track_cache_by_source.get(source_name, [])
                source_color = (
                    source_cache[0].get("color")
                    if source_cache
                    else np.asarray([0.2, 0.7, 1.0, 1.0], dtype=np.float32)
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=source_color,
                        linewidth=1.0,
                        label=f"tracks ({track_source_labels.get(source_name, source_name)})",
                    )
                )
        if legend_handles:
            legend_ref["obj"] = axes[0].legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.65,
            )

    def _draw_current_frame() -> None:
        """Render all panels and overlays for the currently selected frame index."""
        local_index = int(state["index"])
        global_frame = initial_frame + local_index

        raw_detections = _resolve_frame_entries(selected_detections, global_frame, local_index)
        raw_groundtruth = _resolve_frame_entries(groundtruth, global_frame, local_index)
        detection_boxes = _parse_detection_boxes(raw_detections, width, height)
        groundtruth_boxes = _parse_detection_boxes(raw_groundtruth, width, height)
        detection_segments = _boxes_to_line_segments(detection_boxes)
        groundtruth_segments = _boxes_to_line_segments(groundtruth_boxes)
        has_method_detections = (
            bool(detection_boxes) and has_detection_overlay
        )

        track_end_indices_by_source: dict[str, list[int]] = {}
        if has_track_overlay:
            for source_name in available_track_sources:
                if not bool(state["show_tracks_by_source"].get(source_name, False)):
                    continue
                source_cache = track_cache_by_source.get(source_name, [])
                if not source_cache:
                    continue
                track_end_indices_by_source[source_name] = [
                    int(track_data["frames"].searchsorted(global_frame, side="right"))
                    for track_data in source_cache
                ]

        for context in axis_contexts:
            axis = context["axis"]
            video_data = context["video_data"]
            label = context["label"]
            frame = video_data[local_index]
            shown_frame, use_gray = _prepare_frame_for_display(frame)
            _ = use_gray  # image type is stable for each axis once initialized
            context["image_artist"].set_data(shown_frame)
            axis.set_title(f"{label} | frame {global_frame}")

            groundtruth_box_collection = context["groundtruth_box_collection"]
            show_groundtruth = bool(state["show_groundtruth"] and len(groundtruth_segments) > 0)
            groundtruth_box_collection.set_segments(groundtruth_segments if show_groundtruth else [])
            groundtruth_box_collection.set_visible(show_groundtruth)

            det_box_collection = context["det_box_collection"]
            show_det = bool(state["show_detections"] and has_method_detections and len(detection_segments) > 0)
            det_box_collection.set_segments(detection_segments if show_det else [])
            det_box_collection.set_visible(show_det)

            track_line_collection = context["track_line_collection"]
            track_text_artists = context["track_text_artists"]
            if track_line_collection is not None:
                for text_artist in list(track_text_artists):
                    try:
                        text_artist.remove()
                    except Exception:
                        pass
                track_text_artists.clear()

                if track_end_indices_by_source:
                    segments: list[np.ndarray] = []
                    segment_colors: list[Any] = []

                    for source_name, source_end_indices in track_end_indices_by_source.items():
                        source_cache = track_cache_by_source.get(source_name, [])
                        for track_idx, track_data in enumerate(source_cache):
                            end_idx = source_end_indices[track_idx]
                            if end_idx <= 0:
                                continue

                            xs = track_data["xs"][:end_idx]
                            ys = track_data["ys"][:end_idx]
                            color = track_data["color"]

                            if len(xs) > 1:
                                segments.append(np.column_stack((xs, ys)))
                                segment_colors.append(color)

                            if show_track_ids:
                                track_text_artists.append(
                                    axis.text(
                                        float(xs[-1]),
                                        float(ys[-1]),
                                        str(track_data["id"]),
                                        color=color,
                                        fontsize=5,
                                    )
                                )

                    track_line_collection.set_segments(segments)
                    if segment_colors:
                        track_line_collection.set_color(segment_colors)
                        track_line_collection.set_visible(True)
                    else:
                        track_line_collection.set_color(empty_colors)
                        track_line_collection.set_visible(False)
                else:
                    track_line_collection.set_segments([])
                    track_line_collection.set_color(empty_colors)
                    track_line_collection.set_visible(False)

        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(1, int(round(1000.0 / sampling_rate))))

    def _on_timer() -> None:
        """Advance the slider by one frame while playback is enabled."""
        if not state["playing"]:
            return
        next_index = (int(state["index"]) + 1) % frame_count
        slider.set_val(next_index)

    timer.add_callback(_on_timer)
    timer.start()

    def _on_slider_change(value: float) -> None:
        """Handle slider updates by syncing state and redrawing the frame."""
        state["index"] = int(value)
        frame_value_text.set_text(str(int(value)))
        _draw_current_frame()

    slider.on_changed(_on_slider_change)

    def _on_play_clicked(_: Any) -> None:
        """Toggle playback state and update the play/pause button label."""
        state["playing"] = not bool(state["playing"])
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    play_button.on_clicked(_on_play_clicked)

    overlay_specs: list[dict[str, str]] = []
    if has_detection_overlay:
        overlay_specs.append(
            {
                "state_key": "show_detections",
                "title": "detections",
            }
        )
    if has_track_overlay:
        if len(available_track_sources) == 1:
            source_name = available_track_sources[0]
            overlay_specs.append(
                {
                    "state_key": f"show_tracks:{source_name}",
                    "title": "tracks",
                }
            )
        else:
            for source_name in available_track_sources:
                overlay_specs.append(
                    {
                        "state_key": f"show_tracks:{source_name}",
                        "title": f"tracks ({source_name})",
                    }
                )
    if has_groundtruth_overlay:
        overlay_specs.append(
            {
                "state_key": "show_groundtruth",
                "title": "groundtruth",
            }
        )

    overlay_buttons: dict[str, dict[str, Any]] = {}

    def _is_overlay_active(state_key: str) -> bool:
        """Return toggle state for overlay buttons, including per-source tracks."""
        if state_key.startswith("show_tracks:"):
            source_name = state_key.split(":", 1)[1]
            return bool(state["show_tracks_by_source"].get(source_name, False))
        return bool(state.get(state_key, False))

    def _refresh_overlay_button_styles() -> None:
        """Render overlay toggle buttons with active/inactive modern styling."""
        for state_key, button_data in overlay_buttons.items():
            axis = button_data["axis"]
            button = button_data["button"]
            is_active = _is_overlay_active(state_key)
            if is_active:
                base_color = "black"
                hover_color = "#2f2f2f"
                text_color = "white"
            else:
                base_color = "white"
                hover_color = "#efefef"
                text_color = "black"

            # Matplotlib Button hover logic resets the axis facecolor to
            # button.color on mouse leave, so both Button colors and axis
            # facecolor must be kept in sync.
            button.color = base_color
            button.hovercolor = hover_color
            axis.set_facecolor(base_color)
            button.label.set_color(text_color)

            for spine in axis.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)
            button.label.set_fontsize(8)
            button.label.set_fontweight("bold")

    def _make_overlay_toggle_handler(state_key: str):
        """Build click handler for one overlay state key."""

        def _handler(_: Any) -> None:
            if state_key.startswith("show_tracks:"):
                source_name = state_key.split(":", 1)[1]
                current = bool(state["show_tracks_by_source"].get(source_name, False))
                state["show_tracks_by_source"][source_name] = not current
            else:
                state[state_key] = not bool(state.get(state_key, False))
            _refresh_overlay_button_styles()
            _update_legend()
            _draw_current_frame()

        return _handler

    if overlay_specs:
        base_y = 0.55
        row_gap = 0.11
        for row_idx, spec in enumerate(overlay_specs):
            button_y = base_y - (row_idx * row_gap)
            button_ax = fig.add_axes([0.845, button_y, 0.145, 0.05])
            button = Button(
                button_ax,
                spec["title"],
                color="white",
                hovercolor="#f2f2f2",
            )
            button_ax.set_xticks([])
            button_ax.set_yticks([])

            state_key = spec["state_key"]
            overlay_buttons[state_key] = {"axis": button_ax, "button": button}
            button.on_clicked(_make_overlay_toggle_handler(state_key))

        _refresh_overlay_button_styles()

    def _sync_ui_after_state_change() -> None:
        """Redraw overlay-dependent artists and legend."""
        if overlay_buttons:
            _refresh_overlay_button_styles()
        _update_legend()
        _draw_current_frame()

    def _on_key(event: Any) -> None:
        """Handle keyboard shortcuts for play/pause and frame stepping."""
        if event.key == " ":
            _on_play_clicked(event)
        elif event.key == "right":
            slider.set_val((int(state["index"]) + 1) % frame_count)
        elif event.key == "left":
            slider.set_val((int(state["index"]) - 1) % frame_count)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    _sync_ui_after_state_change()
    plt.show()

    casa["meta"]["last_visualization"] = {
        "type": "timelapse",
        "video_type": "+".join(image_keys),
        "detection_method": selected_detection_method,
        "tracking_method": active_tracking_backend,
        "tracking_sources": available_track_sources,
        "show_detections": bool(state["show_detections"]),
        "show_tracks": bool(any(state["show_tracks_by_source"].values())),
        "show_tracks_by_source": {
            source_name: bool(enabled)
            for source_name, enabled in state["show_tracks_by_source"].items()
        },
        "show_groundtruth": bool(state["show_groundtruth"]),
        "show_track_ids": bool(show_track_ids),
    }
    return casa

_IMAGE_ALIASES = {
    "original": ("original_video",),
    "grayscale": ("grayscale_video",),
    "gray": ("grayscale_video",),
    "normalized": ("normalized_video",),
    "binarized": ("binary_video",),
    "binary": ("binary_video",),
    "moving_cells": ("binarized_moving_cells_video",),
    "digital_washing": ("digital_washing_washed_video",),
}


def _get_image_video(casa: dict[str, Any], image_key: str) -> np.ndarray:
    """Fetch the first available video matching the canonical image key."""
    video_dict = casa.get("video", {})
    if image_key not in _IMAGE_ALIASES:
        raise ValueError(f"Unknown image key `{image_key}`.")

    for candidate_key in _IMAGE_ALIASES[image_key]:
        video_data = video_dict.get(candidate_key)
        if isinstance(video_data, np.ndarray):
            return video_data

    raise ValueError(
        f"`{image_key}` video is not available in casa['video']. "
        "Run the relevant preprocessing step first."
    )


def _build_track_cache(
    tracks: dict[str, dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Pre-parse and sort track points for faster per-frame viewer rendering."""
    if not isinstance(tracks, dict) or not tracks:
        return []

    cache: list[dict[str, Any]] = []
    for track_id, track_data in tracks.items():
        if not isinstance(track_data, dict):
            continue

        frames: list[int] = []
        xs: list[float] = []
        ys: list[float] = []

        for frame_key, raw_point in track_data.items():
            frame_idx = _coerce_frame_index_key(frame_key)
            if frame_idx is None:
                continue
            parsed = _coerce_track_point(raw_point, width, height)
            if parsed is None:
                continue
            frames.append(frame_idx)
            xs.append(parsed[0])
            ys.append(parsed[1])

        if not frames:
            continue

        frame_arr = np.asarray(frames, dtype=np.int32)
        order = np.argsort(frame_arr, kind="mergesort")
        cache.append(
            {
                "id": str(track_id),
                "frames": frame_arr[order],
                "xs": np.asarray(xs, dtype=np.float32)[order],
                "ys": np.asarray(ys, dtype=np.float32)[order],
            }
        )

    cache.sort(key=lambda item: item["id"])
    return cache


def _coerce_frame_index_key(value: Any) -> int | None:
    """Best-effort conversion of frame keys to integer indices."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _to_pixel_if_normalized(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Convert normalized ``[0, 1]`` coordinates to pixels."""
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def _coerce_track_point(raw_point: Any, width: int, height: int) -> tuple[float, float] | None:
    """Parse one track point to pixel coordinates."""
    if isinstance(raw_point, dict):
        raw_point = raw_point.get("coord", raw_point.get("point"))

    if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
        return None

    try:
        x_val = float(raw_point[0])
        y_val = float(raw_point[1])
    except (TypeError, ValueError):
        return None

    return _to_pixel_if_normalized(x_val, y_val, width, height)


def _has_overlay_data(frame_dict: dict[str, Any]) -> bool:
    """Return ``True`` when a frame->entries mapping contains at least one row."""
    if not isinstance(frame_dict, dict):
        return False
    for value in frame_dict.values():
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def _to_pixel_length_if_normalized(value: float, scale: int) -> float:
    """Convert normalized scalar length to pixel units when value is in ``[0, 1]``."""
    if 0.0 <= value <= 1.0:
        return float(value * float(scale))
    return float(value)


def _to_pixel_coord_if_normalized(value: float, scale: int) -> float:
    """Convert normalized coordinate to pixels when value is in ``[0, 1]``."""
    if 0.0 <= value <= 1.0:
        return float(value * float(scale))
    return float(value)


def _coerce_detection_bbox(
    raw_detection: Any,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Parse a detection row to ``(x1, y1, x2, y2)`` pixel bounds."""
    x1_val: float | None = None
    y1_val: float | None = None
    x2_val: float | None = None
    y2_val: float | None = None

    if isinstance(raw_detection, dict):
        has_corners = all(
            key in raw_detection for key in ("x1", "y1", "x2", "y2")
        )
        if has_corners:
            try:
                x1_val = _to_pixel_coord_if_normalized(float(raw_detection["x1"]), width)
                y1_val = _to_pixel_coord_if_normalized(float(raw_detection["y1"]), height)
                x2_val = _to_pixel_coord_if_normalized(float(raw_detection["x2"]), width)
                y2_val = _to_pixel_coord_if_normalized(float(raw_detection["y2"]), height)
            except (TypeError, ValueError):
                return None
        else:
            cx_raw = raw_detection.get("x", raw_detection.get("cx"))
            cy_raw = raw_detection.get("y", raw_detection.get("cy"))
            w_raw = raw_detection.get("w", raw_detection.get("width"))
            h_raw = raw_detection.get("h", raw_detection.get("height"))
            if cx_raw is None or cy_raw is None or w_raw is None or h_raw is None:
                return None
            try:
                cx_val = _to_pixel_coord_if_normalized(float(cx_raw), width)
                cy_val = _to_pixel_coord_if_normalized(float(cy_raw), height)
                w_val = abs(_to_pixel_length_if_normalized(float(w_raw), width))
                h_val = abs(_to_pixel_length_if_normalized(float(h_raw), height))
            except (TypeError, ValueError):
                return None
            if w_val <= 0.0 or h_val <= 0.0:
                return None
            x1_val = cx_val - (w_val / 2.0)
            y1_val = cy_val - (h_val / 2.0)
            x2_val = cx_val + (w_val / 2.0)
            y2_val = cy_val + (h_val / 2.0)

    elif isinstance(raw_detection, (list, tuple)):
        cx_idx = None
        cy_idx = None
        w_idx = None
        h_idx = None
        if len(raw_detection) >= 5:
            cx_idx, cy_idx, w_idx, h_idx = 1, 2, 3, 4
        elif len(raw_detection) >= 4:
            cx_idx, cy_idx, w_idx, h_idx = 0, 1, 2, 3
        if cx_idx is None or cy_idx is None or w_idx is None or h_idx is None:
            return None
        try:
            cx_val = _to_pixel_coord_if_normalized(float(raw_detection[cx_idx]), width)
            cy_val = _to_pixel_coord_if_normalized(float(raw_detection[cy_idx]), height)
            w_val = abs(_to_pixel_length_if_normalized(float(raw_detection[w_idx]), width))
            h_val = abs(_to_pixel_length_if_normalized(float(raw_detection[h_idx]), height))
        except (TypeError, ValueError, IndexError):
            return None
        if w_val <= 0.0 or h_val <= 0.0:
            return None
        x1_val = cx_val - (w_val / 2.0)
        y1_val = cy_val - (h_val / 2.0)
        x2_val = cx_val + (w_val / 2.0)
        y2_val = cy_val + (h_val / 2.0)
    else:
        return None

    if x1_val is None or y1_val is None or x2_val is None or y2_val is None:
        return None

    x1_clamped = float(max(0.0, min(float(width), min(x1_val, x2_val))))
    y1_clamped = float(max(0.0, min(float(height), min(y1_val, y2_val))))
    x2_clamped = float(max(0.0, min(float(width), max(x1_val, x2_val))))
    y2_clamped = float(max(0.0, min(float(height), max(y1_val, y2_val))))
    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        return None
    return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def _parse_detection_boxes(
    detections: list[Any],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    """Parse mixed detection rows into pixel bounding boxes."""
    boxes: list[tuple[float, float, float, float]] = []
    for raw_detection in detections:
        parsed = _coerce_detection_bbox(raw_detection, width=width, height=height)
        if parsed is not None:
            boxes.append(parsed)
    return boxes


def _boxes_to_line_segments(
    boxes: list[tuple[float, float, float, float]],
) -> list[np.ndarray]:
    """Convert ``(x1, y1, x2, y2)`` boxes to polyline segments for LineCollection."""
    segments: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        segments.append(
            np.asarray(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [x1, y1],
                ],
                dtype=np.float32,
            )
        )
    return segments


def _format_overlay_source_label(raw_source: Any, fallback: str) -> str:
    """Format a short source label for overlay controls and legends."""
    if isinstance(raw_source, str) and raw_source.strip():
        source_text = raw_source.strip()
        try:
            source_name = Path(source_text).name or source_text
        except Exception:
            source_name = source_text
        if len(source_name) > 18:
            return f"{source_name[:15]}..."
        return source_name
    return fallback
