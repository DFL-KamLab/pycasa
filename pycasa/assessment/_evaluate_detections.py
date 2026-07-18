from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _resolve_active_predicted_detection_method


def _to_int_frame(value: Any) -> int | None:
    """Best-effort conversion of frame-like keys to integers."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _frame_entries(frame_dict: dict[Any, Any], frame: int) -> list[Any]:
    """Return list-like frame entries for integer/string key variants."""
    for key in (str(frame), frame):
        entries = frame_dict.get(key)
        if isinstance(entries, list):
            return entries
    return []


def _to_pixel_if_normalized(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Convert normalized [0, 1] coordinates to pixel coordinates."""
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def _extract_xy_points(entries: list[Any], width: int, height: int) -> np.ndarray:
    """Extract detection centroids from mixed legacy/new detection row formats."""
    points: list[tuple[float, float]] = []
    for entry in entries:
        x_val: float
        y_val: float
        if isinstance(entry, dict):
            x_raw = entry.get("x", entry.get("cx"))
            y_raw = entry.get("y", entry.get("cy"))
            if x_raw is None or y_raw is None:
                continue
            try:
                x_val = float(x_raw)
                y_val = float(y_raw)
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, (list, tuple)):
            if len(entry) >= 3:
                # Legacy format: [label, x, y, w, h]
                x_raw = entry[1]
                y_raw = entry[2]
            elif len(entry) >= 2:
                # Point-like format: [x, y]
                x_raw = entry[0]
                y_raw = entry[1]
            else:
                continue
            try:
                x_val = float(x_raw)
                y_val = float(y_raw)
            except (TypeError, ValueError):
                continue
        else:
            continue

        px, py = _to_pixel_if_normalized(x_val, y_val, width, height)
        points.append((px, py))

    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _format_frame_ranges(frames: list[int]) -> str:
    """Build compact frame-range text (legacy helper parity)."""
    if not frames:
        return "No frames available for detection assessment."

    sorted_frames = sorted(frames)
    ranges: list[str] = []
    start = sorted_frames[0]
    prev = sorted_frames[0]

    for frame in sorted_frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = frame
        prev = frame
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"Detection assessment is performed on frames: {', '.join(ranges)}"


def _resolve_eval_frames(
    casa: dict[str, Any],
    detections: dict[Any, Any],
    groundtruth: dict[Any, Any],
) -> list[int]:
    """Resolve frame IDs using legacy intersection semantics."""
    detection_frames = {_to_int_frame(key) for key in detections.keys()}
    groundtruth_frames = {_to_int_frame(key) for key in groundtruth.keys()}
    detection_frames.discard(None)
    groundtruth_frames.discard(None)

    if not detection_frames or not groundtruth_frames:
        return []

    initial_frame_raw = casa.get("video", {}).get("initial_frame")
    final_frame_raw = casa.get("video", {}).get("final_frame")
    initial_frame = _to_int_frame(initial_frame_raw)
    final_frame = _to_int_frame(final_frame_raw)

    if initial_frame is None or final_frame is None or final_frame < initial_frame:
        min_frame = min(detection_frames | groundtruth_frames)
        max_frame = max(detection_frames | groundtruth_frames)
        range_frames = set(range(min_frame, max_frame + 1))
    else:
        range_frames = set(range(initial_frame, final_frame + 1))

    return sorted(range_frames & detection_frames & groundtruth_frames)


def _compute_fscore(precision: float, recall: float, beta: float) -> float:
    """Compute generalized F-score for the given precision/recall pair."""
    denom = (beta**2 * precision) + recall
    if denom <= 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / denom


def _print_detection_assessment_results(
    detection_method: str | None,
    metrics: dict[str, Any],
    frame_summary: str,
) -> None:
    """Print a concise human-readable detection-assessment summary."""
    method_label = detection_method or "none"
    print(
        "Detection assessment results "
        f"(detection_method={method_label}): "
        f"tp={metrics['tp']}, "
        f"fp={metrics['fp']}, "
        f"fn={metrics['fn']}, "
        f"precision={metrics['precision']}%, "
        f"recall={metrics['recall']}%, "
        f"F1={metrics['F1']}%, "
        f"evaluated_frames={metrics['evaluated_frames']}"
    )
    print(frame_summary)


def evaluate_detections(
    casa: dict[str, Any],
    match_min_distance_pixel: float | None = None,
) -> dict[str, Any]:
    """Compute detection assessment metrics against groundtruth detections.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        match_min_distance_pixel (float | None, optional):
            Distance threshold (in pixels) used to count a matched pair as true
            positive. If ``None``, uses ``casa['meta']['match_min_distance_pixel']``
            when available, otherwise defaults to ``20``.

    Returns:
        dict[str, Any]:
            Input ``casa`` with detection metrics stored under:
            ``casa['assessment']['detection']`` and
            per-frame logs under ``casa['assessment']['detection_log']``.

    Notes:
        This follows legacy pyCASA Hungarian-matching logic and metric formulas.
        A concise metrics summary is printed after each run.
    """
    casa = _ensure_casa(casa)

    detections_root = casa.get("detections", {})
    if not isinstance(detections_root, dict):
        detections_root = {}

    active_detection_method = _resolve_active_predicted_detection_method(detections_root)
    detections = (
        detections_root.get(active_detection_method, {})
        if active_detection_method is not None
        else {}
    )
    groundtruth = detections_root.get("groundtruth", {})
    if not isinstance(detections, dict):
        detections = {}
    if not isinstance(groundtruth, dict):
        groundtruth = {}

    if not detections:
        if active_detection_method is None:
            print(
                "Warning: no active predicted detection method was found in "
                "casa['detections']. Detection assessment will run with empty predictions."
            )
        else:
            print(
                "Warning: no detections were found for active detection method "
                f"'{active_detection_method}'."
            )
    if not groundtruth:
        print(
            "Warning: no groundtruth detections were found in "
            "casa['detections']['groundtruth']."
        )

    match_threshold = (
        float(match_min_distance_pixel)
        if match_min_distance_pixel is not None
        else float(casa.get("meta", {}).get("match_min_distance_pixel", 20.0))
    )
    if match_threshold <= 0:
        raise ValueError("`match_min_distance_pixel` must be > 0.")

    width = _to_int_frame(casa.get("meta", {}).get("width")) or 0
    height = _to_int_frame(casa.get("meta", {}).get("height")) or 0
    original_video = casa.get("video", {}).get("original_video")
    if isinstance(original_video, np.ndarray) and original_video.ndim >= 3:
        if width <= 0:
            width = int(original_video.shape[2])
        if height <= 0:
            height = int(original_video.shape[1])
    if width <= 0:
        width = 1
    if height <= 0:
        height = 1

    all_frames = _resolve_eval_frames(casa, detections, groundtruth)

    tp = 0
    fp = 0
    fn = 0
    prev_tp = 0
    prev_fp = 0
    prev_fn = 0
    performance_metrics_log: list[list[str]] = []

    linear_sum_assignment: Any = None
    scipy_checked = False

    for frame in all_frames:
        y_points = _extract_xy_points(_frame_entries(detections, frame), width=width, height=height)
        x_points = _extract_xy_points(_frame_entries(groundtruth, frame), width=width, height=height)

        n = x_points.shape[0]
        m = y_points.shape[0]

        if n == 0:
            fp += m
            continue
        if m == 0:
            fn += n
            continue

        if n > m:
            fn += n - m
        elif n < m:
            fp += m - n

        if not scipy_checked:
            try:
                scipy_optimize = _ensure_import(
                    "scipy.optimize",
                    pip_name="scipy",
                    prompt_install=False,
                )
            except ImportError as exc:
                raise ImportError(
                    "SciPy is required for detection assessment matching. "
                    "Install with `python -m pip install -e .[detection]`."
                ) from exc
            linear_sum_assignment = scipy_optimize.linear_sum_assignment
            scipy_checked = True

        distances = np.linalg.norm(x_points[:, None, :] - y_points[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(distances)
        matches = distances[row_ind, col_ind]

        for match in matches:
            if float(match) < match_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1

        performance_metrics_log.append(
            [f"Frame:{frame}, tp:{tp - prev_tp}, fp:{fp - prev_fp}, fn:{fn - prev_fn}"]
        )
        prev_tp, prev_fp, prev_fn = tp, fp, fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": np.round(precision * 100, 2),
        "recall": np.round(recall * 100, 2),
        "F0.5": np.round(_compute_fscore(precision, recall, 0.5) * 100, 2),
        "F1": np.round(_compute_fscore(precision, recall, 1.0) * 100, 2),
        "F2": np.round(_compute_fscore(precision, recall, 2.0) * 100, 2),
        "evaluated_frames": len(all_frames),
    }

    assessment = casa.setdefault("assessment", {})
    assessment["detection"] = metrics
    assessment["detection_log"] = performance_metrics_log
    frame_summary = _format_frame_ranges(all_frames)
    assessment["last_detection"] = {
        "detection_method": active_detection_method,
        "match_min_distance_pixel": match_threshold,
        "evaluated_frames": len(all_frames),
        "frame_summary": frame_summary,
    }
    casa["meta"]["last_assessment"] = {
        "backend": "detection",
        "detection_method": active_detection_method,
        "match_min_distance_pixel": match_threshold,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "F1": metrics["F1"],
        "evaluated_frames": metrics["evaluated_frames"],
    }
    _print_detection_assessment_results(active_detection_method, metrics, frame_summary)
    return casa
