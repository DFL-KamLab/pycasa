from typing import Any


_GROUNDTRUTH_KEYS = {"groundtruth", "groundtruth_path"}


def _predicted_detection_keys(detections_root: dict[str, Any]) -> list[str]:
    """Return non-groundtruth detection method keys."""
    keys: list[str] = []
    for key, value in detections_root.items():
        if key in _GROUNDTRUTH_KEYS:
            continue
        if isinstance(value, dict):
            keys.append(str(key))
    return sorted(keys)


def _clear_predicted_detections(
    detections_root: dict[str, Any],
    keep_method: str | None = None,
) -> list[str]:
    """Remove all predicted detection entries, optionally keeping one key."""
    removed: list[str] = []
    for key in list(detections_root.keys()):
        if key in _GROUNDTRUTH_KEYS:
            continue
        if keep_method is not None and str(key) == keep_method:
            continue
        if isinstance(detections_root.get(key), dict):
            removed.append(str(key))
            detections_root.pop(key, None)
    return removed


def _resolve_active_predicted_detection_method(
    detections_root: dict[str, Any],
) -> str | None:
    """Resolve the active predicted detection method under single-result policy."""
    methods = _predicted_detection_keys(detections_root)
    if not methods:
        return None
    if len(methods) > 1:
        raise ValueError(
            "Multiple predicted detection methods were found in "
            "casa['detections']. This session now supports one active predicted "
            "detection. Re-run one detection method to overwrite previous results."
        )
    return methods[0]
