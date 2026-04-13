from typing import Any

import numpy as np

from .._core._casa import _ensure_casa


def _store_binarization_results(
    casa: dict[str, Any],
    *,
    method: str,
    binary_video: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Store binarization output videos and method metadata into ``casa``."""
    casa = _ensure_casa(casa)
    casa["video"]["binary_video"] = binary_video
    casa["video"]["binary_type"] = method
    casa["meta"]["last_preprocessing"] = {"operation": "binarize", "method": method}
    if verbose:
        print(
            "Binarization summary: "
            f"method={method}, "
            f"frames={int(binary_video.shape[0])}, "
            f"shape={tuple(binary_video.shape)}, "
            f"dtype={binary_video.dtype}"
        )
    return casa
