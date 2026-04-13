_REQUIRED_TOP_LEVEL_KEYS = (
    "meta",
    "video",
    "detections",
    "tracks",
    "motility",
    "assessment",
)


def _new_casa():
    """Return a new empty CASA session dictionary with required top-level keys."""
    return {k: {} for k in _REQUIRED_TOP_LEVEL_KEYS}


def _ensure_casa(casa=None):
    """Validate/normalize a CASA session dictionary and fill missing sections."""
    if casa is None:
        casa = _new_casa()
    if not isinstance(casa, dict):
        raise TypeError("`casa` must be a dictionary.")
    for k in _REQUIRED_TOP_LEVEL_KEYS:
        if k not in casa or not isinstance(casa[k], dict):
            casa[k] = {}
    return casa
