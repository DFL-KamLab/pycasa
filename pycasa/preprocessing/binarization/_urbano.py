from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import (
    _convert_video_to_grayscale,
    _progress_bar,
    _ensure_original_video,
    _store_binarization_results,
)


def urbano(
    casa: dict[str, Any],
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the placeholder Urbano-style binarization implementation.

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
        - This is currently a placeholder implementation that writes zero-valued
          binary frames.
        - Input frames are converted to grayscale internally for shape
          normalization.
        - Writes ``casa["video"]["binary_video"]`` and
          ``casa["video"]["binary_type"] = "urbano"``.
        - Updates ``casa["meta"]["last_preprocessing"]``.

    Examples:
        >>> import pycasa_as as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.binarization.urbano()
    """
    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if verbose:
        print("Running binarization urbano on frames...")
    grayscale_frames = _convert_video_to_grayscale(original_video)
    frame_count = int(grayscale_frames.shape[0])
    binary_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        grayscale_frames,
        total=frame_count,
        desc="Binarization urbano",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        binary_frames.append(np.zeros_like(frame, dtype=np.uint8))
    binary_video = np.stack(binary_frames, axis=0)
    return _store_binarization_results(
        casa,
        method="urbano",
        binary_video=binary_video,
        verbose=verbose,
    )
