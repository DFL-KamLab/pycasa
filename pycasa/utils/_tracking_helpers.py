from typing import Any

from ._detection_helpers import _resolve_active_predicted_detection_method

_KNOWN_TRACKING_BACKENDS = ("sort", "deepsort", "jpdaf")


def _resolve_active_tracking_backend(tracks_root: dict[str, Any]) -> str | None:
    """Return the name of the active tracking backend stored in ``tracks_root``.

    Checks known backends in declaration order and returns the first one that
    exists as a dict key.  Returns ``None`` when no recognized backend is found.
    """
    if not isinstance(tracks_root, dict):
        return None
    for backend in _KNOWN_TRACKING_BACKENDS:
        if isinstance(tracks_root.get(backend), dict):
            return backend
    return None


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
    """Return track maps grouped by source for the active tracking backend.

    Checks all known backends (``sort``, ``jpdaf``) and returns sources for the
    first one found.  Retains the original ``sort``-centric name for backward
    compatibility with existing call sites.
    """
    if not isinstance(tracks_root, dict):
        return {}

    backend = _resolve_active_tracking_backend(tracks_root)
    if backend is None:
        return {}

    backend_tracks = tracks_root.get(backend)
    if not isinstance(backend_tracks, dict) or not backend_tracks:
        return {}

    # Backward-compatibility fallback for older single-result shape.
    if _is_track_map(backend_tracks):
        return {"groundtruth": backend_tracks}

    sources: dict[str, dict[str, Any]] = {}
    for source_name, source_tracks in backend_tracks.items():
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
