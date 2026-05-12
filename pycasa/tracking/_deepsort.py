from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _progress_bar
from ..utils import _resolve_active_predicted_detection_method
from ..utils import _ensure_video_dimensions
from ..utils import _warn_yellow
from ._sort import _detection_to_bbox, _get_frame_detections, _resolve_tracking_frame_range

_DEEPSORT_REPO_DIR = Path.home() / ".pycasa" / "deepsort"
_DUMMY_FEATURE = np.ones(128, dtype=np.float32) / np.sqrt(128.0)


@contextmanager
def _temporary_sys_path(directory: str):
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        if directory in sys.path:
            sys.path.remove(directory)


def _ensure_deepsort_repo() -> Path:
    """Clone nwojke/deep_sort to ~/.pycasa/deepsort/ if not already present."""
    marker = _DEEPSORT_REPO_DIR / "deep_sort" / "__init__.py"
    if marker.is_file():
        return _DEEPSORT_REPO_DIR
    _DEEPSORT_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning nwojke/deep_sort to ~/.pycasa/deepsort/ ...")
    subprocess.run(
        [
            "git", "clone", "--depth=1",
            "https://github.com/nwojke/deep_sort.git",
            str(_DEEPSORT_REPO_DIR),
        ],
        check=True,
    )
    return _DEEPSORT_REPO_DIR


def _sorted_tracks(
    tracks: dict[str, dict[int, list[float]]],
) -> dict[str, dict[int, list[float]]]:
    def _key(k: str) -> tuple[int, str]:
        suffix = str(k)[1:]
        return (int(suffix), str(k)) if suffix.isdigit() else (10**9, str(k))

    return {k: tracks[k] for k in sorted(tracks.keys(), key=_key)}


def deepsort(
    casa: dict[str, Any],
    skip_gt: bool = False,
    max_age: int = 30,
    n_init: int = 3,
    max_iou_distance: float = 0.7,
    max_cosine_distance: float = 1.0,
    nn_budget: int | None = None,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Track detections using DeepSORT and store per-source results.

    Uses the original nwojke/deep_sort implementation, auto-cloned to
    ``~/.pycasa/deepsort/`` on first use.

    By default (``max_cosine_distance=1.0``) appearance-based matching is
    disabled and association relies purely on Kalman-filter predictions and
    IoU gating — the recommended setting for biological cells whose visual
    appearance is near-identical across tracks.  Set a lower
    ``max_cosine_distance`` (e.g. 0.3) if you supply meaningful ReID features
    via a custom workflow.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        skip_gt (bool, optional):
            If ``False`` (default), run tracking on both available sources:
            ``groundtruth`` and active predicted detections. If ``True``,
            skip ``groundtruth`` and run detections-only.
        max_age (int, optional):
            Maximum number of missed frames before dropping a track.
        n_init (int, optional):
            Minimum consecutive detections needed before a track is confirmed
            (equivalent to ``min_hits`` in SORT).
        max_iou_distance (float, optional):
            Maximum IoU *distance* for the gate (distance = 1 − IoU).
            Default 0.7 corresponds to a minimum IoU of 0.3.
        max_cosine_distance (float, optional):
            Appearance-feature gate threshold.  ``1.0`` (default) effectively
            disables appearance-based matching and uses IoU-only association.
        nn_budget (int | None, optional):
            Maximum number of appearance features stored per track for the
            nearest-neighbour appearance metric.  ``None`` = unlimited.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with per-source tracks stored in
            ``casa['tracks']['deepsort'][source]``.

    Raises:
        ValueError:
            If tracking cannot resolve frame geometry from ``meta`` or video
            array shape.

    Notes:
        - Track points are stored as global-frame center coordinates:
          ``{track_id: {frame: [center_x, center_y]}}``.
        - Tracking output is reset on each call before recomputation.
        - ``casa['meta']['last_tracking']`` is always updated.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.Casa()
        >>> session = session.tracking.deepsort(skip_gt=False)
    """
    deepsort_dir = str(_ensure_deepsort_repo())
    with _temporary_sys_path(deepsort_dir):
        from deep_sort.tracker import Tracker
        from deep_sort import nn_matching
        from deep_sort.detection import Detection

    casa = _ensure_casa(casa)

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
    existing_backends = [k for k, v in tracks_root.items() if isinstance(v, dict)]
    if existing_backends:
        _warn_yellow(
            f"Previous tracking result overwritten "
            f"({', '.join(existing_backends)} -> deepsort)."
        )
    tracks_root.clear()
    tracks_root["deepsort"] = {}
    deepsort_root = tracks_root["deepsort"]

    _skipped_meta = {
        "backend": "deepsort",
        "detection_method": None,
        "sources_processed": [],
        "per_source": {},
        "skip_gt": bool(skip_gt),
        "max_age": int(max_age),
        "n_init": int(n_init),
        "max_iou_distance": float(max_iou_distance),
        "max_cosine_distance": float(max_cosine_distance),
        "nn_budget": nn_budget,
        "skipped": True,
    }

    source_order: list[str] = []
    if skip_gt:
        if has_detections and active_predicted_method is not None:
            source_order.append(active_predicted_method)
        else:
            if has_groundtruth:
                _warn_yellow("No detections found, tracking skipped because skip_gt=True")
                reason = "missing_detections_skip_gt"
            else:
                _warn_yellow("No detections/GT found, either import GT or run detection")
                reason = "missing_detections_and_groundtruth"
            casa["meta"]["last_tracking"] = {**_skipped_meta, "reason": reason}
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
                **_skipped_meta,
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

    if verbose and source_order:
        print(
            "Running DeepSORT tracking on frames "
            f"(sources={', '.join(source_order)})..."
        )

    per_source: dict[str, dict[str, Any]] = {}
    total_output_tracks = 0
    total_input_frames = 0
    processed_sources: list[str] = []

    for source_name in source_order:
        raw_detections = detections_root.get(source_name, {})
        if not isinstance(raw_detections, dict) or not raw_detections:
            deepsort_root[source_name] = {}
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
            deepsort_root[source_name] = {}
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
                raw_detections, global_frame,
            )

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget,
        )
        tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
        )
        tracks: dict[str, dict[int, list[float]]] = {}

        for frame_idx in _progress_bar(
            range(number_frame_used),
            total=number_frame_used,
            desc=f"Tracking deepsort ({source_name})",
            unit="frame",
            leave=True,
            enabled=show_progress,
        ):
            frame_data = local_detections.get(str(frame_idx), [])
            ds_dets: list[Any] = []
            for det in frame_data:
                bbox = _detection_to_bbox(det, width=width, height=height)
                if bbox is None:
                    continue
                conf = det.get("conf", 1.0) if isinstance(det, dict) else 1.0
                x1, y1, x2, y2 = bbox
                ds_dets.append(
                    Detection([x1, y1, x2 - x1, y2 - y1], float(conf), _DUMMY_FEATURE)
                )

            tracker.predict()
            tracker.update(ds_dets)

            for t in tracker.tracks:
                if not t.is_confirmed():
                    continue
                ltrb = t.to_tlbr()
                center_x = float((ltrb[0] + ltrb[2]) / 2.0)
                center_y = float((ltrb[1] + ltrb[3]) / 2.0)
                track_key = f"t{t.track_id}"
                tracks.setdefault(track_key, {})[frame_idx] = [center_x, center_y]

        sorted_local_tracks = _sorted_tracks(tracks)
        globalized: dict[str, dict[str, list[float]]] = {}
        for track_id, local_track in sorted_local_tracks.items():
            global_track: dict[str, list[float]] = {}
            for local_idx, coords in local_track.items():
                global_frame = initial_frame + int(local_idx)
                global_track[str(global_frame)] = coords
            globalized[track_id] = global_track

        deepsort_root[source_name] = globalized
        processed_sources.append(source_name)
        output_tracks = int(len(globalized))
        total_output_tracks += output_tracks
        total_input_frames += int(number_frame_used)
        average_track_length = (
            float(
                np.mean([
                    len(track_points)
                    for track_points in globalized.values()
                    if isinstance(track_points, dict)
                ])
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
    if active_predicted_method is not None and active_predicted_method in deepsort_root:
        primary_source = active_predicted_method
    elif "groundtruth" in deepsort_root:
        primary_source = "groundtruth"
    elif deepsort_root:
        primary_source = sorted(deepsort_root.keys())[0]

    all_skipped = (
        all(bool(s.get("skipped", True)) for s in per_source.values())
        if per_source
        else True
    )

    if verbose:
        print(
            "DeepSORT summary: "
            f"sources_processed={processed_sources}, "
            f"input_frames={total_input_frames}, "
            f"output_tracks={total_output_tracks}"
        )
        for source_name in processed_sources:
            source_info = per_source.get(source_name, {})
            avg_len = source_info.get("average_track_length")
            avg_text = f"{float(avg_len):.2f}" if isinstance(avg_len, (int, float)) else "None"
            print(
                f"- {source_name}: "
                f"tracks={source_info.get('output_tracks')}, "
                f"average_track_length={avg_text}"
            )

    casa["meta"]["last_tracking"] = {
        "backend": "deepsort",
        "detection_method": primary_source,
        "sources_requested": list(source_order),
        "sources_processed": processed_sources,
        "per_source": per_source,
        "skip_gt": bool(skip_gt),
        "max_age": int(max_age),
        "n_init": int(n_init),
        "max_iou_distance": float(max_iou_distance),
        "max_cosine_distance": float(max_cosine_distance),
        "nn_budget": nn_budget,
        "input_frames": int(total_input_frames),
        "output_tracks": int(total_output_tracks),
        "skipped": bool(all_skipped),
        "reason": "all_sources_skipped" if all_skipped else None,
    }
    return casa
