from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import (
    _ensure_import,
    _convert_video_to_grayscale,
    _progress_bar,
    _ensure_original_video,
    _store_binarization_results,
)


def adaptive_gaussian(
    casa: dict[str, Any],
    *,
    block_size: int = 11,
    c: float = 2.0,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Binarize frames with OpenCV adaptive Gaussian thresholding.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        block_size (int, optional):
            Odd neighborhood size (>= 3) used by adaptive thresholding.
        c (float, optional):
            Constant subtracted from neighborhood-weighted mean.
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
            If ``block_size`` is not an odd integer >= 3, or if the video
            is missing/invalid.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.
        ImportError:
            If ``opencv-python`` is unavailable.

    Notes:
        - Input frames are converted to grayscale internally.
        - Output is ``uint8`` with values ``0`` and ``255``.
        - Writes ``casa["video"]["binary_video"]`` and
          ``casa["video"]["binary_type"] = "adaptive-gaussian"``.
        - Updates ``casa["meta"]["last_preprocessing"]``.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.binarization.adaptive_gaussian(block_size=11, c=2.0)
    """
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("`block_size` must be an odd integer >= 3.")

    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if verbose:
        print(
            "Running binarization adaptive-gaussian on frames "
            f"(block_size={block_size}, c={float(c):.3f})..."
        )
    grayscale_frames = _convert_video_to_grayscale(original_video)
    cv2 = _ensure_import("cv2", pip_name="opencv-python")

    frame_count = int(grayscale_frames.shape[0])
    binary_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        grayscale_frames,
        total=frame_count,
        desc="Binarization adaptive-gaussian",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        binary_frames.append(
            cv2.adaptiveThreshold(
                frame.astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                int(block_size),
                float(c),
            )
        )
    binary_video = np.stack(binary_frames, axis=0)
    return _store_binarization_results(
        casa,
        method="adaptive-gaussian",
        binary_video=binary_video,
        verbose=verbose,
    )
