from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _progress_bar
from ..utils import _resolve_active_predicted_detection_method
from ..utils import _ensure_video_dimensions
from ..utils import _msg_yellow, _warn_yellow


def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve linear assignment, preferring LAP when available."""
    try:
        import lap  # type: ignore

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0], dtype=int)
    except ImportError:
        scipy_optimize = _ensure_import(
            "scipy.optimize",
            pip_name="scipy",
            prompt_install=False,
        )
        x, y = scipy_optimize.linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)), dtype=int)


def _iou_batch(bb_test: np.ndarray, bb_groundtruth: np.ndarray) -> np.ndarray:
    """Compute IoU matrix for arrays of ``[x1, y1, x2, y2]`` boxes."""
    if bb_test.size == 0 or bb_groundtruth.size == 0:
        return np.zeros((len(bb_test), len(bb_groundtruth)), dtype=float)

    bb_groundtruth = np.expand_dims(bb_groundtruth, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_groundtruth[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_groundtruth[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_groundtruth[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_groundtruth[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    denom = (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_groundtruth[..., 2] - bb_groundtruth[..., 0])
        * (bb_groundtruth[..., 3] - bb_groundtruth[..., 1])
        - wh
    )
    denom = np.where(denom <= 0, 1e-12, denom)
    return wh / denom


def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to Kalman measurement ``[x, y, s, r]``."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h if h != 0 else 1e-12)
    return np.array([x, y, s, r], dtype=float).reshape((4, 1))


def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state ``[x, y, s, r]`` to ``[x1, y1, x2, y2]`` box."""
    s = float(x[2, 0])
    r = float(x[3, 0])
    if s <= 0 or r <= 0:
        w = 0.0
        h = 0.0
    else:
        w = float(np.sqrt(s * r))
        h = float(s / (w if w != 0 else 1e-12))

    cx = float(x[0, 0])
    cy = float(x[1, 0])
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=float,
    ).reshape((1, 4))


class _KalmanBoxTracker:
    """Internal SORT track state for one object."""

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        self.x = np.zeros((7, 1), dtype=float)
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.P = np.eye(7, dtype=float)
        self.R = np.eye(4, dtype=float)
        self.Q = np.eye(7, dtype=float)

        self.R[2:, 2:] *= 10.0
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x[:4] = _convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = _KalmanBoxTracker.count
        _KalmanBoxTracker.count += 1
        self.history: list[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, bbox: np.ndarray) -> None:
        z = _convert_bbox_to_z(bbox)
        y = z - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + (k @ y)
        i = np.eye(self.P.shape[0], dtype=float)
        self.P = (i - (k @ self.H)) @ self.P

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self._update(bbox)

    def predict(self) -> np.ndarray:
        if float(self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self._predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_convert_x_to_bbox(self.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return _convert_x_to_bbox(self.x)


def _associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign detections to trackers using IoU and Hungarian matching."""
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty((0,), dtype=int),
        )

    iou_matrix = _iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        association = (iou_matrix > iou_threshold).astype(np.int32)
        if association.sum(1).max() == 1 and association.sum(0).max() == 1:
            matched_indices = np.stack(np.where(association), axis=1)
        else:
            matched_indices = _linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    matched_det_ids = set(matched_indices[:, 0].tolist()) if matched_indices.size else set()
    matched_trk_ids = set(matched_indices[:, 1].tolist()) if matched_indices.size else set()
    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matched_det_ids]
    unmatched_trackers = [idx for idx in range(len(trackers)) if idx not in matched_trk_ids]

    matches: list[np.ndarray] = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_detections.append(int(det_idx))
            unmatched_trackers.append(int(trk_idx))
        else:
            matches.append(np.array([[det_idx, trk_idx]], dtype=int))

    matched = (
        np.empty((0, 2), dtype=int)
        if len(matches) == 0
        else np.concatenate(matches, axis=0)
    )
    return (
        matched,
        np.array(unmatched_detections, dtype=int),
        np.array(unmatched_trackers, dtype=int),
    )


class _SortTracker:
    """SORT tracker implementation used by ``pycasa.tracking.sort``."""

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray | None = None) -> np.ndarray:
        dets = np.empty((0, 4), dtype=float) if detections is None else np.asarray(detections, dtype=float)
        if dets.ndim != 2:
            dets = np.empty((0, 4), dtype=float)
        if dets.size == 0:
            dets = np.empty((0, 4), dtype=float)
        elif dets.shape[1] >= 4:
            dets = dets[:, :4]
        else:
            dets = np.empty((0, 4), dtype=float)

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5), dtype=float)
        to_delete: list[int] = []
        ret: list[np.ndarray] = []

        for idx in range(len(trks)):
            pos = self.trackers[idx].predict()[0]
            trks[idx, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_delete.append(idx)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for idx in reversed(to_delete):
            self.trackers.pop(idx)

        matched, unmatched_dets, _ = _associate_detections_to_trackers(
            dets,
            trks[:, :4] if trks.size else np.empty((0, 4), dtype=float),
            self.iou_threshold,
        )

        for det_idx, trk_idx in matched:
            self.trackers[int(trk_idx)].update(dets[int(det_idx), :4])

        for det_idx in unmatched_dets:
            self.trackers.append(_KalmanBoxTracker(dets[int(det_idx), :4]))

        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            state = tracker.get_state()[0]
            if tracker.time_since_update < 1 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((state, [tracker.id + 1]), axis=0).reshape(1, -1))
            i -= 1
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret, axis=0)
        return np.empty((0, 5), dtype=float)


def _detection_to_bbox(det: Any, width: int, height: int) -> list[float] | None:
    """Convert one detection row to pixel-space ``[x1, y1, x2, y2]``."""
    cx: float
    cy: float
    w: float
    h: float

    if isinstance(det, dict):
        if all(key in det for key in ("x1", "y1", "x2", "y2")):
            try:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
            except (TypeError, ValueError):
                return None
            return [x1, y1, x2, y2]

        x_raw = det.get("x", det.get("cx"))
        y_raw = det.get("y", det.get("cy"))
        w_raw = det.get("w", det.get("width"))
        h_raw = det.get("h", det.get("height"))
        if x_raw is None or y_raw is None or w_raw is None or h_raw is None:
            return None
        try:
            cx = float(x_raw)
            cy = float(y_raw)
            w = float(w_raw)
            h = float(h_raw)
        except (TypeError, ValueError):
            return None
    elif isinstance(det, (list, tuple)):
        if len(det) >= 5:
            try:
                cx = float(det[1])
                cy = float(det[2])
                w = float(det[3])
                h = float(det[4])
            except (TypeError, ValueError):
                return None
        elif len(det) == 4:
            try:
                x1 = float(det[0])
                y1 = float(det[1])
                x2 = float(det[2])
                y2 = float(det[3])
            except (TypeError, ValueError):
                return None
            return [x1, y1, x2, y2]
        else:
            return None
    else:
        return None

    if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
        cx *= width
        cy *= height
        w *= width
        h *= height

    if w <= 0 or h <= 0:
        return None

    x_min = cx - w / 2.0
    y_min = cy - h / 2.0
    x_max = cx + w / 2.0
    y_max = cy + h / 2.0
    return [x_min, y_min, x_max, y_max]


def _get_frame_detections(raw_detections: dict[Any, Any], frame_key: int) -> list[Any]:
    """Fetch frame detections using string/int key compatibility."""
    for key in (str(frame_key), frame_key):
        data = raw_detections.get(key)
        if isinstance(data, list):
            return data
    return []


def _sort_track_key(track_key: str) -> tuple[int, str]:
    """Build numeric-friendly sort key for track IDs shaped like ``t1``."""
    suffix = str(track_key)[1:]
    if suffix.isdigit():
        return (int(suffix), str(track_key))
    return (10**9, str(track_key))


def _sorted_tracks(tracks: dict[str, dict[int, list[float]]]) -> dict[str, dict[int, list[float]]]:
    """Sort track dictionary by track id suffix (``t1``, ``t2``, ...)."""
    return {key: tracks[key] for key in sorted(tracks.keys(), key=_sort_track_key)}


def _resolve_tracking_frame_range(
    video_info: dict[str, Any],
    video_array: np.ndarray | None,
    raw_detections: dict[Any, Any],
) -> tuple[int, int] | None:
    """Resolve ``(initial_frame, number_frame_used)`` for tracking."""
    initial_frame = int(video_info.get("initial_frame", 0) or 0)
    number_frame_used = int(video_info.get("number_frame_used", 0) or 0)

    if number_frame_used <= 0 and isinstance(video_array, np.ndarray):
        number_frame_used = int(video_array.shape[0])

    if number_frame_used > 0:
        return initial_frame, number_frame_used

    frame_ids: list[int] = []
    for key in raw_detections.keys():
        try:
            frame_ids.append(int(str(key)))
        except (TypeError, ValueError):
            continue
    if not frame_ids:
        return None

    initial_frame = min(frame_ids)
    number_frame_used = max(frame_ids) - initial_frame + 1
    return initial_frame, number_frame_used


def sort(
    casa: dict[str, Any],
    skip_gt: bool = False,
    delete_temp: bool = True,
    max_age: int = 25,
    min_hits: int = 3,
    iou_threshold: float = 0.1,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Track detections using SORT and store per-source results.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        skip_gt (bool, optional):
            If ``False`` (default), run tracking on both available sources:
            ``groundtruth`` and active predicted detections. If ``True``,
            skip ``groundtruth`` and run detections-only.
        delete_temp (bool, optional):
            Kept for legacy API compatibility. No temporary files are created
            in this in-process implementation.
        max_age (int, optional):
            Maximum number of missed frames before dropping a track.
        min_hits (int, optional):
            Minimum associated detections before a track is emitted.
        iou_threshold (float, optional):
            Minimum IoU threshold for assignment.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames during tracking.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for SORT.
            If ``False``, suppress those summaries. Warnings are not affected
            by this flag.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with per-source tracks stored in
            ``casa['tracks']['sort'][source]``.

    Raises:
        ValueError:
            If tracking cannot resolve frame geometry from ``meta`` or video
            array shape.
        ImportError:
            If SciPy is unavailable when Hungarian assignment is needed.

    Notes:
        - Supports both legacy list/tuple detections and dict-based detection
          rows.
        - Track points are stored as global-frame center coordinates:
          ``{track_id: {frame: [center_x, center_y]}}``.
        - Tracking output is reset on each call before recomputation.
        - ``casa['meta']['last_tracking']`` is always updated.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.Casa()
        >>> session = session.tracking.sort(skip_gt=False)
    """
    casa = _ensure_casa(casa)
    _msg_yellow(
        "This SORT implementation is adopted from https://github.com/abewley/sort, authored by Alex Bewley under the GPL-3.0 License, with a small modification to handle frames with zero detections (empty _iou_batch guard)."
    )
    detections_root = casa.get("detections", {})
    if not isinstance(detections_root, dict):
        detections_root = {}

    active_predicted_method = _resolve_active_predicted_detection_method(detections_root)
    gt_detections = detections_root.get("groundtruth", {})
    has_groundtruth = isinstance(gt_detections, dict) and bool(gt_detections)
    predicted_detections: dict[str, Any] = {}
    if active_predicted_method is not None:
        candidate = detections_root.get(active_predicted_method, {})
        if isinstance(candidate, dict):
            predicted_detections = candidate
    has_detections = bool(predicted_detections)

    tracks_root = casa.setdefault("tracks", {})
    tracks_root.clear()
    tracks_root["sort"] = {}
    sort_root = tracks_root["sort"]

    source_order: list[str] = []
    if skip_gt:
        if has_detections and active_predicted_method is not None:
            source_order.append(active_predicted_method)
        else:
            if has_groundtruth:
                _warn_yellow(
                    "No detections found, tracking skipped because skip_gt=True"
                )
                reason = "missing_detections_skip_gt"
            else:
                _warn_yellow("No detections/GT found, either import GT or run detection")
                reason = "missing_detections_and_groundtruth"
            casa["meta"]["last_tracking"] = {
                "backend": "sort",
                "detection_method": None,
                "sources_processed": [],
                "per_source": {},
                "skip_gt": True,
                "delete_temp": bool(delete_temp),
                "max_age": int(max_age),
                "min_hits": int(min_hits),
                "iou_threshold": float(iou_threshold),
                "skipped": True,
                "reason": reason,
            }
            return casa
    else:
        if has_groundtruth and has_detections and active_predicted_method is not None:
            print("GT and detections found, tracking will run on both")
            source_order.extend(["groundtruth", active_predicted_method])
        elif has_groundtruth:
            _warn_yellow("No detections found, tracking will only run on GT")
            source_order.append("groundtruth")
        elif has_detections and active_predicted_method is not None:
            _warn_yellow("No GT found, tracking will only run on detections")
            source_order.append(active_predicted_method)
        else:
            _warn_yellow("No detections/GT found, either import GT or run detection")
            casa["meta"]["last_tracking"] = {
                "backend": "sort",
                "detection_method": None,
                "sources_processed": [],
                "per_source": {},
                "skip_gt": False,
                "delete_temp": bool(delete_temp),
                "max_age": int(max_age),
                "min_hits": int(min_hits),
                "iou_threshold": float(iou_threshold),
                "skipped": True,
                "reason": "missing_detections_and_groundtruth",
            }
            return casa

    video_info = casa.get("video", {})
    meta_info = casa.get("meta", {})
    video_array = video_info.get("array")
    width = int(meta_info.get("width") or 0)
    height = int(meta_info.get("height") or 0)
    if isinstance(video_array, np.ndarray):
        _, video_height, video_width = _ensure_video_dimensions(video_array)
        if width <= 0:
            width = int(video_width)
        if height <= 0:
            height = int(video_height)
    if width <= 0 or height <= 0:
        raise ValueError(
            "Tracking requires video width/height in casa['meta'] or a valid video array."
        )

    if verbose:
        if source_order:
            print(
                "Running SORT tracking on frames "
                f"(sources={', '.join(source_order)})..."
            )

    per_source: dict[str, dict[str, Any]] = {}
    total_output_tracks = 0
    total_input_frames = 0
    processed_sources: list[str] = []

    for source_name in source_order:
        raw_detections = detections_root.get(source_name, {})
        if not isinstance(raw_detections, dict) or not raw_detections:
            sort_root[source_name] = {}
            per_source[source_name] = {
                "input_frames": 0,
                "output_tracks": 0,
                "average_track_length": None,
                "skipped": True,
                "reason": "missing_detections",
            }
            continue

        frame_range = _resolve_tracking_frame_range(
            video_info=video_info if isinstance(video_info, dict) else {},
            video_array=video_array if isinstance(video_array, np.ndarray) else None,
            raw_detections=raw_detections,
        )
        if frame_range is None:
            sort_root[source_name] = {}
            per_source[source_name] = {
                "input_frames": 0,
                "output_tracks": 0,
                "average_track_length": None,
                "skipped": True,
                "reason": "invalid_frame_range",
            }
            continue

        initial_frame, number_frame_used = frame_range
        local_detections: dict[str, list[Any]] = {}
        for local_idx in range(number_frame_used):
            global_frame = initial_frame + local_idx
            local_detections[str(local_idx)] = _get_frame_detections(
                raw_detections,
                global_frame,
            )

        _KalmanBoxTracker.count = 0
        tracker = _SortTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )
        tracks: dict[str, dict[int, list[float]]] = {}

        for frame_idx in _progress_bar(
            range(number_frame_used),
            total=number_frame_used,
            desc=f"Tracking sort ({source_name})",
            unit="frame",
            leave=True,
            enabled=show_progress,
        ):
            frame_data = local_detections.get(str(frame_idx), [])
            frame_boxes: list[list[float]] = []
            for detection in frame_data:
                box = _detection_to_bbox(detection, width=width, height=height)
                if box is not None:
                    frame_boxes.append(box)

            detections_array = (
                np.asarray(frame_boxes, dtype=float)
                if frame_boxes
                else np.empty((0, 4), dtype=float)
            )

            tracked_objects = tracker.update(detections_array)
            for tracked_object in tracked_objects:
                track_id = int(tracked_object[4])
                center_x = float((tracked_object[0] + tracked_object[2]) / 2.0)
                center_y = float((tracked_object[1] + tracked_object[3]) / 2.0)
                track_key = f"t{track_id}"
                tracks.setdefault(track_key, {})[frame_idx] = [center_x, center_y]

        sorted_local_tracks = _sorted_tracks(tracks)
        globalized: dict[str, dict[str, list[float]]] = {}
        for track_id, local_track in sorted_local_tracks.items():
            global_track: dict[str, list[float]] = {}
            for local_idx, coords in local_track.items():
                global_frame = initial_frame + int(local_idx)
                global_track[str(global_frame)] = coords
            globalized[track_id] = global_track

        sort_root[source_name] = globalized
        processed_sources.append(source_name)
        output_tracks = int(len(globalized))
        total_output_tracks += output_tracks
        total_input_frames += int(number_frame_used)
        average_track_length = (
            float(
                np.mean(
                    [
                        len(track_points)
                        for track_points in globalized.values()
                        if isinstance(track_points, dict)
                    ]
                )
            )
            if globalized
            else 0.0
        )
        per_source[source_name] = {
            "input_frames": int(number_frame_used),
            "output_tracks": output_tracks,
            "average_track_length": average_track_length,
            "skipped": False,
        }

    primary_source: str | None = None
    if active_predicted_method is not None and active_predicted_method in sort_root:
        primary_source = active_predicted_method
    elif "groundtruth" in sort_root:
        primary_source = "groundtruth"
    elif sort_root:
        primary_source = sorted(sort_root.keys())[0]

    all_skipped = all(
        bool(source_info.get("skipped", True))
        for source_info in per_source.values()
    ) if per_source else True
    if verbose:
        print(
            "SORT summary: "
            f"sources_processed={processed_sources}, "
            f"input_frames={total_input_frames}, "
            f"output_tracks={total_output_tracks}"
        )
        for source_name in processed_sources:
            source_info = per_source.get(source_name, {})
            avg_len = source_info.get("average_track_length")
            avg_text = f"{float(avg_len):.2f}" if isinstance(avg_len, (int, float)) else "None"
            print(
                "- "
                f"{source_name}: "
                f"tracks={source_info.get('output_tracks')}, "
                f"average_track_length={avg_text}"
            )

    reason: str | None = None
    if all_skipped:
        reason = "all_sources_skipped"
    casa["meta"]["last_tracking"] = {
        "backend": "sort",
        "detection_method": primary_source,
        "sources_requested": list(source_order),
        "sources_processed": processed_sources,
        "per_source": per_source,
        "skip_gt": bool(skip_gt),
        "delete_temp": bool(delete_temp),
        "max_age": int(max_age),
        "min_hits": int(min_hits),
        "iou_threshold": float(iou_threshold),
        "input_frames": int(total_input_frames),
        "output_tracks": int(total_output_tracks),
        "skipped": bool(all_skipped),
        "reason": reason,
    }
    return casa
