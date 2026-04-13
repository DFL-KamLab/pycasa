from typing import Any

import numpy as np

from .._core._casa import _ensure_casa


def _store_normalization_results(
    casa: dict[str, Any],
    *,
    method: str,
    normalized: np.ndarray,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Store normalization output videos and method metadata into ``casa``."""
    casa = _ensure_casa(casa)
    casa["video"]["normalized_video"] = normalized
    casa["video"]["normalized_type"] = method
    if overwrite:
        casa["video"]["original_video"] = normalized
    casa["meta"]["last_preprocessing"] = {
        "operation": "normalize",
        "method": method,
        "overwrite": overwrite,
    }
    if verbose:
        print(
            "Normalization summary: "
            f"method={method}, "
            f"frames={int(normalized.shape[0])}, "
            f"shape={tuple(normalized.shape)}, "
            f"dtype={normalized.dtype}, "
            f"overwrite={overwrite}"
        )
    return casa


def _framewise_minmax(image: np.ndarray) -> np.ndarray:
    """Scale one image/frame to ``[0, 255]`` using min-max normalization."""
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _hist_equalize_uint8(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to one ``uint8`` image."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum().astype(np.float64)
    non_zero = cdf[cdf > 0]
    if non_zero.size == 0:
        return image.copy()
    cdf_min = non_zero[0]
    denom = cdf[-1] - cdf_min
    if denom <= 0:
        return image.copy()
    lut = np.clip((cdf - cdf_min) * 255.0 / denom, 0, 255).astype(np.uint8)
    return lut[image]
