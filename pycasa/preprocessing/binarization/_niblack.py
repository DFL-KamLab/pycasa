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


def niblack(
    casa: dict[str, Any],
    *,
    window_size: int = 25,
    k: float = -0.2,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Binarize frames with Niblack local thresholding.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        window_size (int, optional):
            Local window size (>= 3) used to estimate local mean and standard
            deviation.
        k (float, optional):
            Niblack scaling factor applied to local standard deviation.
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
            If ``window_size`` is < 3, or if the video is missing/invalid.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.
        ImportError:
            If ``scipy`` is unavailable.

    Notes:
        - Input frames are converted to grayscale internally.
        - Output is ``uint8`` with values ``0`` and ``255``.
        - Writes ``casa["video"]["binary_video"]`` and
          ``casa["video"]["binary_type"] = "niblack"``.
        - Updates ``casa["meta"]["last_preprocessing"]``.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.binarization.niblack(window_size=25, k=-0.2)
    """
    if window_size < 3:
        raise ValueError("`window_size` must be >= 3.")

    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if verbose:
        print(
            "Running binarization niblack on frames "
            f"(window_size={window_size}, k={float(k):.3f})..."
        )
    grayscale_frames = _convert_video_to_grayscale(original_video)
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    uniform_filter = scipy_ndimage.uniform_filter

    frame_count = int(grayscale_frames.shape[0])
    binary_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        grayscale_frames,
        total=frame_count,
        desc="Binarization niblack",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        frame_f = frame.astype(np.float32)
        mean = uniform_filter(frame_f, size=int(window_size))
        mean_sq = uniform_filter(frame_f * frame_f, size=int(window_size))
        var = np.maximum(mean_sq - mean * mean, 0.0)
        std = np.sqrt(var)
        threshold_map = mean + float(k) * std
        binary_frames.append(np.where(frame_f > threshold_map, 255, 0).astype(np.uint8))

    binary_video = np.stack(binary_frames, axis=0)
    return _store_binarization_results(
        casa,
        method="niblack",
        binary_video=binary_video,
        verbose=verbose,
    )
