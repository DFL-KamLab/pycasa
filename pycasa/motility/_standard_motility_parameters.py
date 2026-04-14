from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _progress_bar
from ..utils import _resolve_sort_track_sources
from ..utils import _warn_yellow

_MOTILITY_KEYS = ("VCL", "VSL", "VAP", "LIN", "ALH", "WOB", "STR", "MAD")


def _to_int(value: Any) -> int | None:
    """Return an integer when conversion succeeds, otherwise ``None``."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _format_frame_ranges(frames: list[int]) -> str:
    """Format sorted frame IDs into compact contiguous range text."""
    if not frames:
        return ""

    frames_sorted = sorted(frames)
    ranges: list[tuple[int, int]] = []
    start = frames_sorted[0]
    prev = frames_sorted[0]
    for frame in frames_sorted[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append((start, prev))
        start = frame
        prev = frame
    ranges.append((start, prev))

    return " ".join(
        f"{start_idx}" if start_idx == end_idx else f"{start_idx}-{end_idx}"
        for start_idx, end_idx in ranges
    )


def _resolve_video_size(casa: dict[str, Any]) -> tuple[int, int]:
    """Resolve video width/height from metadata, falling back to loaded array."""
    width = _to_int(casa.get("meta", {}).get("width")) or 0
    height = _to_int(casa.get("meta", {}).get("height")) or 0

    original_video = casa.get("video", {}).get("original_video")
    if isinstance(original_video, np.ndarray) and original_video.ndim >= 3:
        if height <= 0:
            height = int(original_video.shape[1])
        if width <= 0:
            width = int(original_video.shape[2])

    return width, height


def _coerce_track_points(
    track_data: dict[Any, Any],
    width: int,
    height: int,
) -> dict[int, np.ndarray]:
    """Normalize one track mapping into ``frame_int -> [x_px, y_px]`` points."""
    normalized: dict[int, np.ndarray] = {}

    for frame_key, coords in track_data.items():
        frame_int = _to_int(frame_key)
        if frame_int is None:
            continue
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue

        try:
            x_val = float(coords[0])
            y_val = float(coords[1])
        except (TypeError, ValueError):
            continue

        if width > 0 and height > 0 and 0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0:
            x_val *= width
            y_val *= height

        normalized[frame_int] = np.array([x_val, y_val], dtype=float)

    return normalized


def _compute_segment_motility(
    track_segment: dict[int, np.ndarray],
    fps: float,
    smooth_w: int,
) -> dict[str, float]:
    """Compute legacy motility metrics for one sliding-window segment."""
    frames = sorted(track_segment.keys())
    points = np.array([track_segment[frame] for frame in frames], dtype=float)
    if points.shape[0] < 2:
        return dict.fromkeys(_MOTILITY_KEYS, 0.0)

    total_time = (frames[-1] - frames[0]) / fps if fps > 0 else 0.0
    if total_time == 0:
        total_time = 1.0

    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)

    vcl = float(distances.sum() / total_time)
    vsl = float(np.linalg.norm(points[-1] - points[0]) / total_time)
    lin = float(vsl / vcl) if vcl else 0.0

    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    if angles.size <= 1:
        mad = 0.0
    else:
        mad = float(np.degrees(np.mean(np.abs(np.diff(np.unwrap(angles))))))

    if points.shape[0] >= smooth_w and smooth_w >= 2:
        kernel = np.ones(smooth_w, dtype=float) / smooth_w
        avg_points = np.stack(
            [
                np.convolve(points[:, 0], kernel, mode="valid"),
                np.convolve(points[:, 1], kernel, mode="valid"),
            ],
            axis=1,
        )
        if avg_points.shape[0] >= 2:
            avg_deltas = np.linalg.norm(np.diff(avg_points, axis=0), axis=1)
            half_window = smooth_w // 2
            effective_frames = frames[half_window : -half_window or None]
            effective_time = (
                (effective_frames[-1] - effective_frames[0]) / fps if fps > 0 else 0.0
            )
            if effective_time == 0:
                effective_time = total_time
            vap = float(avg_deltas.sum() / effective_time)

            lateral_dev: list[float] = []
            for idx, frame_idx in enumerate(range(half_window, half_window + avg_points.shape[0])):
                if idx < avg_points.shape[0] - 1:
                    tangent_vec = avg_points[idx + 1] - avg_points[idx]
                else:
                    tangent_vec = avg_points[idx] - avg_points[idx - 1]
                tangent_norm = float(np.linalg.norm(tangent_vec))
                if tangent_norm == 0:
                    continue
                tangent_unit = tangent_vec / tangent_norm
                perpendicular_unit = np.array(
                    [-tangent_unit[1], tangent_unit[0]],
                    dtype=float,
                )
                deviation = points[frame_idx] - avg_points[idx]
                lateral_dev.append(abs(float(np.dot(deviation, perpendicular_unit))))
            alh = (
                float((max(lateral_dev) - min(lateral_dev)) / 2.0)
                if lateral_dev
                else 0.0
            )
        else:
            vap = 0.0
            alh = 0.0
    else:
        vap = 0.0
        alh = 0.0

    wob = float(vap / vcl) if vcl else 0.0
    str_ = float(vsl / vap) if vap else 0.0

    return {
        "VCL": vcl,
        "VSL": vsl,
        "VAP": vap,
        "LIN": lin,
        "ALH": alh,
        "WOB": wob,
        "STR": str_,
        "MAD": mad,
    }


def _ensure_um_per_px(value: Any) -> float:
    """Validate a positive, finite microns-per-pixel value."""
    if value is None:
        raise ValueError("`um_per_px` is required when conversion is requested.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("`um_per_px` must be a positive finite number.")

    validated = float(value)
    if not np.isfinite(validated) or validated <= 0:
        raise ValueError("`um_per_px` must be a positive finite number.")
    return validated


def _build_motility_parameter_summary(
    method_metrics: dict[str, Any],
) -> dict[str, float | int | None]:
    """Aggregate track-level motility output into concise summary statistics."""
    track_count = int(len(method_metrics))
    window_counts: list[int] = []
    metric_values: dict[str, list[float]] = {key: [] for key in _MOTILITY_KEYS}

    for track_metrics in method_metrics.values():
        if not isinstance(track_metrics, dict):
            continue

        window_series = track_metrics.get("VCL")
        if isinstance(window_series, list):
            window_counts.append(len(window_series))
        else:
            window_counts.append(0)

        for key in _MOTILITY_KEYS:
            values = track_metrics.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    metric_values[key].append(float(value))
                except (TypeError, ValueError):
                    continue

    average_windows = (
        float(sum(window_counts) / len(window_counts))
        if window_counts
        else None
    )

    summary: dict[str, float | int | None] = {
        "track_count": track_count,
        "average_windows_per_track": average_windows,
    }
    for key in _MOTILITY_KEYS:
        values = metric_values[key]
        summary[f"mean_{key}"] = float(sum(values) / len(values)) if values else None
    return summary


def _format_summary_value(value: float | int | None) -> str:
    """Format summary values with stable two-decimal precision."""
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def _print_motility_parameter_summary(
    tracking_backend: str,
    source_name: str,
    summary: dict[str, float | int | None],
    converted_units: bool,
) -> None:
    """Print a concise motility summary after successful computation."""
    speed_unit = "um/s" if converted_units else "px/s"
    displacement_unit = "um" if converted_units else "px"
    print(f"Motility parameter summary ({tracking_backend}:{source_name})")
    print(
        "- "
        f"tracks={_format_summary_value(summary.get('track_count'))}, "
        "average windows per track="
        f"{_format_summary_value(summary.get('average_windows_per_track'))}"
    )
    print(
        "- "
        f"VCL={_format_summary_value(summary.get('mean_VCL'))} {speed_unit}, "
        f"VSL={_format_summary_value(summary.get('mean_VSL'))} {speed_unit}"
    )
    print(
        "- "
        f"VAP={_format_summary_value(summary.get('mean_VAP'))} {speed_unit}, "
        f"LIN={_format_summary_value(summary.get('mean_LIN'))}"
    )
    print(
        "- "
        f"ALH={_format_summary_value(summary.get('mean_ALH'))} {displacement_unit}, "
        f"WOB={_format_summary_value(summary.get('mean_WOB'))}"
    )
    print(
        "- "
        f"STR={_format_summary_value(summary.get('mean_STR'))}, "
        f"MAD={_format_summary_value(summary.get('mean_MAD'))} deg"
    )


def standard_motility_parameters(
    casa: dict[str, Any],
    frame_rate: float | None = None,
    window_size: int = 10,
    overlap: float = 0.2,
    smoothing_window: int | None = None,
    conversion_required: bool = True,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute legacy-standard motility parameters from SORT trajectories.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing tracking output under
            ``casa["tracks"]["sort"][source]``.
        frame_rate (float | None, optional):
            FPS override. If ``None``, uses ``casa["meta"]["sampling_rate"]``
            when available, otherwise ``30``.
        window_size (int, optional):
            Number of points per sliding window.
        overlap (float, optional):
            Window overlap ratio used in legacy step calculation:
            ``step = max(1, int(window_size * (1 - overlap)))``.
        smoothing_window (int | None, optional):
            Smoothing window for VAP/ALH. Defaults to
            ``max(2, window_size // 2)`` when ``None``.
        conversion_required (bool, optional):
            If ``True``, requires a valid positive finite
            ``casa["meta"]["um_per_px"]`` for micron conversion.
            If ``False``, missing calibration keeps output in pixel units.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            tracks.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for motility
            computation. If ``False``, suppress those summaries. Warnings are
            not affected by this flag.

    Returns:
        dict[str, Any]:
            Updated ``casa`` dictionary with motility output written to
            ``casa["motility"]["standard_motility_parameters"][source]``.

    Raises:
        ValueError:
            If ``conversion_required=True`` and ``um_per_px`` is missing or
            invalid.
        RuntimeError:
            If no source has tracks with enough points for the requested
            ``window_size``.

    Notes:
        - Metrics per track are ``VCL``, ``VSL``, ``VAP``, ``LIN``, ``ALH``,
          ``WOB``, ``STR``, ``MAD``, and ``frame_ranges``.
        - VCL/VSL/VAP are converted from px/s to um/s when calibration is
          available and used.
        - ALH is converted from px to um when calibration is available and used.
        - ``casa["meta"]["last_motility"]`` is updated for both skipped and
          successful runs.
        - A concise motility-parameter summary is printed for each processed
          source.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data()
        >>> session = session.tracking.sort(skip_gt=False)
        >>> session = session.set_um_per_px(0.24)
        >>> session = session.motility.standard_motility_parameters()
    """
    casa = _ensure_casa(casa)
    result_key = "standard_motility_parameters"
    tracking_backend = "sort"

    meta = casa.setdefault("meta", {})
    um_per_px_raw = meta.get("um_per_px")

    scale: float | None = None
    if conversion_required:
        scale = _ensure_um_per_px(um_per_px_raw)
    elif um_per_px_raw is not None:
        scale = _ensure_um_per_px(um_per_px_raw)

    fps = float(frame_rate or meta.get("sampling_rate") or 30.0)
    if fps <= 0:
        fps = 30.0

    effective_window_size = max(2, int(window_size))
    effective_smoothing_window = (
        int(smoothing_window)
        if smoothing_window is not None
        else max(2, effective_window_size // 2)
    )
    effective_smoothing_window = max(2, effective_smoothing_window)
    overlap_value = float(overlap)

    tracks_root = casa.get("tracks", {})
    if not isinstance(tracks_root, dict):
        tracks_root = {}
    tracks_by_source = _resolve_sort_track_sources(tracks_root)

    motility_root = casa.setdefault("motility", {})
    existing_motility_methods = [
        str(key) for key, value in motility_root.items() if isinstance(value, dict)
    ]
    if existing_motility_methods:
        _warn_yellow(
            "Previous motility result overwritten "
            f"({', '.join(existing_motility_methods)} -> {result_key})."
        )
    motility_root.clear()
    motility_root[result_key] = {}

    if not tracks_by_source:
        if verbose:
            print(
                "No tracks found under 'sort'. "
                "Run tracking first."
            )
        meta["last_motility"] = {
            "backend": result_key,
            "tracking_backend": tracking_backend,
            "detection_method": None,
            "sources_processed": [],
            "per_source": {},
            "frame_rate": fps,
            "window_size": effective_window_size,
            "overlap": overlap_value,
            "smoothing_window": effective_smoothing_window,
            "um_per_px": scale,
            "conversion_required": bool(conversion_required),
            "tracks_used": 0,
            "skipped": True,
            "reason": "missing_tracks",
        }
        return casa

    width, height = _resolve_video_size(casa)
    step = max(1, int(effective_window_size * (1 - overlap_value)))
    per_source: dict[str, dict[str, Any]] = {}
    processed_sources: list[str] = []
    any_source_succeeded = False
    tracks_used_total = 0

    # Keep deterministic source order and prioritize groundtruth first.
    ordered_sources = sorted(tracks_by_source.keys(), key=lambda value: (value != "groundtruth", value))
    if verbose:
        print(
            "Running standard motility parameters on tracks "
            f"(sources={ordered_sources})..."
        )

    for source_name in ordered_sources:
        tracks_for_source = tracks_by_source.get(source_name, {})
        if not isinstance(tracks_for_source, dict) or not tracks_for_source:
            motility_root[result_key][source_name] = {}
            per_source[source_name] = {
                "tracks_used": 0,
                "track_count": 0,
                "skipped": True,
                "reason": "missing_tracks",
            }
            continue

        parsed_tracks: dict[str, dict[int, np.ndarray]] = {}
        for track_id, raw_track in tracks_for_source.items():
            if isinstance(raw_track, dict):
                parsed_tracks[str(track_id)] = _coerce_track_points(
                    raw_track,
                    width=width,
                    height=height,
                )

        valid_track_ids = [
            track_id
            for track_id, track_data in parsed_tracks.items()
            if len(track_data) >= effective_window_size
        ]
        if not valid_track_ids:
            motility_root[result_key][source_name] = {}
            per_source[source_name] = {
                "tracks_used": 0,
                "track_count": int(len(parsed_tracks)),
                "skipped": True,
                "reason": "insufficient_track_points",
            }
            continue

        processed_sources.append(source_name)
        tracks_used_total += int(len(valid_track_ids))
        any_source_succeeded = True
        motility_root[result_key][source_name] = {}

        for track_id in _progress_bar(
            valid_track_ids,
            total=len(valid_track_ids),
            desc=f"Motility ({tracking_backend}:{source_name})",
            unit="track",
            leave=True,
            enabled=show_progress,
        ):
            track_data = parsed_tracks[track_id]
            frame_list = sorted(track_data.keys())
            number_points = len(frame_list)

            metrics: dict[str, list[float] | list[str]] = {
                key: [] for key in _MOTILITY_KEYS
            }
            metrics["frame_ranges"] = []

            for start in range(0, number_points - effective_window_size + 1, step):
                window_frames = frame_list[start : start + effective_window_size]
                segment = {frame: track_data[frame] for frame in window_frames}
                values = _compute_segment_motility(
                    segment,
                    fps=fps,
                    smooth_w=effective_smoothing_window,
                )

                if scale is not None:
                    for key in ("VCL", "VSL", "VAP"):
                        values[key] *= scale
                    values["ALH"] *= scale

                metrics["frame_ranges"].append(_format_frame_ranges(window_frames))
                for key in _MOTILITY_KEYS:
                    metrics[key].append(float(values[key]))

            motility_root[result_key][source_name][track_id] = metrics

        source_summary = _build_motility_parameter_summary(motility_root[result_key][source_name])
        per_source[source_name] = {
            "tracks_used": int(len(valid_track_ids)),
            "track_count": int(source_summary.get("track_count") or 0),
            "average_windows_per_track": source_summary.get("average_windows_per_track"),
            "skipped": False,
        }
        if verbose:
            _print_motility_parameter_summary(
                tracking_backend=tracking_backend,
                source_name=source_name,
                summary=source_summary,
                converted_units=(scale is not None),
            )

    if not any_source_succeeded:
        raise RuntimeError(
            "No tracks with enough points were found in any tracked source. "
            f"Required points per track: {effective_window_size}."
        )

    primary_source: str | None = None
    if isinstance(meta.get("last_tracking"), dict):
        candidate = meta.get("last_tracking", {}).get("detection_method")
        if isinstance(candidate, str) and candidate in processed_sources:
            primary_source = candidate
    if primary_source is None:
        if "groundtruth" in processed_sources:
            primary_source = "groundtruth"
        elif processed_sources:
            primary_source = processed_sources[0]

    meta["last_motility"] = {
        "backend": result_key,
        "tracking_backend": tracking_backend,
        "detection_method": primary_source,
        "sources_processed": processed_sources,
        "per_source": per_source,
        "frame_rate": fps,
        "window_size": effective_window_size,
        "overlap": overlap_value,
        "smoothing_window": effective_smoothing_window,
        "um_per_px": scale,
        "conversion_required": bool(conversion_required),
        "tracks_used": int(tracks_used_total),
        "skipped": False,
    }
    return casa
