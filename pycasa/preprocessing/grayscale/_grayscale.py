from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import _convert_video_to_grayscale, _progress_bar, _ensure_original_video


def grayscale(
    casa: dict[str, Any],
    overwrite: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert loaded video frames to grayscale.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        overwrite (bool, optional):
            If ``True``, replace ``casa["video"]["original_video"]`` with grayscale
            output. If ``False``, keep the original array unchanged.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            preprocessing step. If ``False``, suppress those summaries.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with grayscale results stored.

    Raises:
        ValueError:
            If ``casa["video"]["original_video"]`` is missing or has invalid shape.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.

    Notes:
        - Always writes ``casa["video"]["grayscale_video"]``.
        - When ``overwrite=True``, also writes ``casa["video"]["original_video"]``.
        - Updates ``casa["meta"]["last_preprocessing"]`` with operation metadata.

    Examples:
        >>> import pycasa_as as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.grayscale(overwrite=False)
    """
    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if verbose:
        print("Running preprocessing grayscale on frames...")

    frame_count = int(original_video.shape[0])
    grayscale_frames: list[np.ndarray] = []

    for frame_idx in _progress_bar(
        range(frame_count),
        total=frame_count,
        desc="Preprocessing grayscale",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        grayscale_frame = _convert_video_to_grayscale(original_video[frame_idx : frame_idx + 1])[0]
        grayscale_frames.append(grayscale_frame)

    grayscale_video = np.stack(grayscale_frames, axis=0)
    casa["video"]["grayscale_video"] = grayscale_video

    if overwrite:
        casa["video"]["original_video"] = grayscale_video

    casa["meta"]["last_preprocessing"] = {
        "operation": "grayscale",
        "overwrite": overwrite,
    }
    if verbose:
        print(
            "Grayscale summary: "
            f"frames={frame_count}, "
            f"shape={tuple(grayscale_video.shape)}, "
            f"dtype={grayscale_video.dtype}, "
            f"overwrite={overwrite}"
        )
    return casa
