from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import (
    _convert_video_to_grayscale,
    _progress_bar,
    _ensure_original_video,
    _store_binarization_results,
)


def _otsu_threshold(image: np.ndarray) -> int:
    """Return the integer Otsu threshold for one grayscale frame."""
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = image.size
    if total == 0:
        return 0

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2 = np.nan_to_num(sigma_b2, nan=-1.0)
    return int(np.argmax(sigma_b2))


def otsu(
    casa: dict[str, Any],
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Binarize frames using per-frame Otsu thresholding.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            preprocessing step. If ``False``, suppress those summaries.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with binary results stored.

    Raises:
        ValueError:
            If ``casa["video"]["original_video"]`` is missing or has invalid shape.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.

    Notes:
        - Input frames are converted to grayscale internally.
        - Output is ``uint8`` with values ``0`` and ``255``.
        - Writes ``casa["video"]["binary_video"]`` and
          ``casa["video"]["binary_type"] = "otsu"``.
        - Updates ``casa["meta"]["last_preprocessing"]``.

    Examples:
        >>> import pycasa_as as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.binarization.otsu()
    """
    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if verbose:
        print("Running binarization otsu on frames...")
    grayscale_frames = _convert_video_to_grayscale(original_video)

    frame_count = int(grayscale_frames.shape[0])
    binary_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        grayscale_frames,
        total=frame_count,
        desc="Binarization otsu",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        binary_frames.append(np.where(frame > _otsu_threshold(frame), 255, 0).astype(np.uint8))
    binary_video = np.stack(binary_frames, axis=0)
    return _store_binarization_results(
        casa,
        method="otsu",
        binary_video=binary_video,
        verbose=verbose,
    )
