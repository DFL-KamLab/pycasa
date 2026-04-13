from copy import deepcopy
from typing import Any

from .._core._casa import _ensure_casa
from ._detection_helpers import _resolve_active_predicted_detection_method


def get_casa(casa: dict[str, Any]) -> dict[str, Any]:
    """Return the validated CASA session dictionary."""
    return _ensure_casa(casa)


def get_meta(casa: dict[str, Any]) -> dict[str, Any]:
    """Return ``casa['meta']``."""
    return _ensure_casa(casa)["meta"]


def get_video(casa: dict[str, Any]) -> dict[str, Any]:
    """Return ``casa['video']``."""
    return _ensure_casa(casa)["video"]


def get_groundtruth(casa: dict[str, Any]) -> dict[str, Any]:
    """Return ``casa['detections']['groundtruth']`` when present."""
    detections = _ensure_casa(casa)["detections"]
    groundtruth = detections.get("groundtruth", {})
    return groundtruth if isinstance(groundtruth, dict) else {}


def get_detections(
    casa: dict[str, Any],
    *,
    include_groundtruth: bool = False,
) -> dict[str, Any]:
    """Return active predicted detections, or full detections when requested."""
    detections = _ensure_casa(casa)["detections"]
    if include_groundtruth:
        return detections

    active_method = _resolve_active_predicted_detection_method(detections)
    if active_method is None:
        return {}
    active = detections.get(active_method, {})
    return active if isinstance(active, dict) else {}


def get_tracks(casa: dict[str, Any], *, backend: str | None = None) -> dict[str, Any]:
    """Return all tracks or a single backend bucket such as ``sort``."""
    tracks = _ensure_casa(casa)["tracks"]
    if backend is None:
        return tracks
    selected = tracks.get(str(backend), {})
    return selected if isinstance(selected, dict) else {}


def get_motility(casa: dict[str, Any]) -> dict[str, Any]:
    """Return ``casa['motility']``."""
    return _ensure_casa(casa)["motility"]


def get_assessment(casa: dict[str, Any]) -> dict[str, Any]:
    """Return ``casa['assessment']``."""
    return _ensure_casa(casa)["assessment"]


def get_assesment(casa: dict[str, Any]) -> dict[str, Any]:
    """Compatibility spelling alias for :func:`get_assessment`."""
    return get_assessment(casa)


def copy_casa(casa: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied CASA dictionary snapshot."""
    return deepcopy(_ensure_casa(casa))
