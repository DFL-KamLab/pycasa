from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import _progress_bar, _ensure_original_video, _store_normalization_results


def z_score(
    casa: dict[str, Any],
    *,
    overwrite: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Normalize frames using per-frame z-score standardization.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        overwrite (bool, optional):
            If ``True``, replace ``casa["video"]["original_video"]`` with normalized
            output. If ``False``, keep the original array unchanged.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            preprocessing step. If ``False``, suppress those summaries.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with normalization results stored.

    Raises:
        ValueError:
            If the video is missing or does not have 3D/4D shape.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.

    Notes:
        - Per frame, output is ``(frame - mean) / std`` in ``float32``.
        - Zero-variance frames become zeros.
        - Output is stored in ``casa["video"]["normalized_video"]``.
        - Method tag is stored in ``casa["video"]["normalized_type"] = "z-score"``.
        - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
        - Updates ``casa["meta"]["last_preprocessing"]`` with method metadata.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.normalization.z_score(overwrite=False)
    """
    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if original_video.ndim not in (3, 4):
        raise ValueError(f"Unsupported video shape: {original_video.shape}")
    if verbose:
        print(f"Running normalization z-score on frames (overwrite={overwrite})...")

    frame_count = int(original_video.shape[0])
    normalized_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        original_video,
        total=frame_count,
        desc="Normalization z-score",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        frame_f = frame.astype(np.float32)
        std = float(np.std(frame_f))
        if std == 0:
            normalized_frames.append(np.zeros_like(frame_f, dtype=np.float32))
        else:
            normalized_frames.append(((frame_f - np.mean(frame_f)) / std).astype(np.float32))

    normalized = np.stack(normalized_frames, axis=0)
    return _store_normalization_results(
        casa,
        method="z-score",
        normalized=normalized,
        overwrite=overwrite,
        verbose=verbose,
    )
