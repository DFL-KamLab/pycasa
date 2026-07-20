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


def _ensure_positive_value(value: float, name: str) -> float:
    """Validate and normalize a positive finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a numeric value.")
    validated = float(value)
    if not math.isfinite(validated):
        raise ValueError(f"`{name}` must be finite.")
    if validated <= 0:
        raise ValueError(f"`{name}` must be > 0.")
    return validated


def set_volume_ml(casa: dict[str, Any], volume_ml: float) -> dict[str, Any]:
    """Set ``casa['meta']['volume_ml']`` (ejaculate volume, mL) after validation."""
    casa = _ensure_casa(casa)
    casa["meta"]["volume_ml"] = _ensure_positive_value(volume_ml, "volume_ml")
    return casa


def set_chamber_depth_um(casa: dict[str, Any], chamber_depth_um: float) -> dict[str, Any]:
    """Set ``casa['meta']['chamber_depth_um']`` (counting-chamber depth, um)."""
    casa = _ensure_casa(casa)
    casa["meta"]["chamber_depth_um"] = _ensure_positive_value(
        chamber_depth_um, "chamber_depth_um"
    )
    return casa
