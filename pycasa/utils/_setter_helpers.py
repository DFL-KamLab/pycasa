import math
from typing import Any

from .._core._casa import _ensure_casa


def _ensure_um_per_px_value(um_per_px: float) -> float:
    """Validate and normalize a positive finite microns-per-pixel value."""
    if isinstance(um_per_px, bool) or not isinstance(um_per_px, (int, float)):
        raise TypeError("`um_per_px` must be a numeric value.")

    validated = float(um_per_px)
    if not math.isfinite(validated):
        raise ValueError("`um_per_px` must be finite.")
    if validated <= 0:
        raise ValueError("`um_per_px` must be > 0.")
    return validated


def set_um_per_px(casa: dict[str, Any], um_per_px: float) -> dict[str, Any]:
    """Set ``casa['meta']['um_per_px']`` after strict value validation."""
    casa = _ensure_casa(casa)
    casa["meta"]["um_per_px"] = _ensure_um_per_px_value(um_per_px)
    return casa
