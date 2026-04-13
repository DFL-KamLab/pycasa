from typing import Any

from ._detection_helpers import _resolve_active_predicted_detection_method


def _is_track_map(candidate: Any, *, allow_empty: bool = False) -> bool:
    """Return ``True`` when data looks like ``track_id -> frame -> coord``."""
    if not isinstance(candidate, dict):
        return False
    if not candidate:
        return bool(allow_empty)

    first_track = next((value for value in candidate.values() if isinstance(value, dict)), None)
    if first_track is None:
        return False

    first_point = next(iter(first_track.values()), None)
    return first_point is None or not isinstance(first_point, dict)


def _resolve_sort_track_sources(tracks_root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return SORT track maps grouped by source method key."""
    if not isinstance(tracks_root, dict):
        return {}

    sort_tracks = tracks_root.get("sort")
    if not isinstance(sort_tracks, dict):
        return {}
    if not sort_tracks:
        return {}

    # Backward-compatibility fallback for older single-result shape.
    if _is_track_map(sort_tracks):
        return {"groundtruth": sort_tracks}

    sources: dict[str, dict[str, Any]] = {}
    for source_name, source_tracks in sort_tracks.items():
        if not isinstance(source_tracks, dict):
            continue
        if _is_track_map(source_tracks, allow_empty=True):
            sources[str(source_name)] = source_tracks
    return sources


def _resolve_active_sort_source_name(
    tracks_root: dict[str, Any],
    detections_root: dict[str, Any] | None = None,
    meta_last_tracking: dict[str, Any] | None = None,
) -> str | None:
    """Resolve active SORT source name with detection-first fallback semantics."""
    sources = _resolve_sort_track_sources(tracks_root)
    if not sources:
        return None

    if isinstance(detections_root, dict):
        active_detection = _resolve_active_predicted_detection_method(detections_root)
        if active_detection is not None and active_detection in sources:
            return active_detection

    if isinstance(meta_last_tracking, dict):
        from_meta = meta_last_tracking.get("detection_method")
        if isinstance(from_meta, str) and from_meta in sources:
            return from_meta

    if "groundtruth" in sources:
        return "groundtruth"

    return sorted(sources.keys())[0]


def _resolve_active_sort_tracks(
    tracks_root: dict[str, Any],
    detections_root: dict[str, Any] | None = None,
    meta_last_tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return active SORT tracks from nested ``tracks['sort'][source]`` storage."""
    sources = _resolve_sort_track_sources(tracks_root)
    if not sources:
        return {}

    active_source = _resolve_active_sort_source_name(
        tracks_root,
        detections_root=detections_root,
        meta_last_tracking=meta_last_tracking,
    )
    if active_source is None:
        return {}
    return sources.get(active_source, {})
