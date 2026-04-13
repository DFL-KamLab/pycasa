from typing import Any

import numpy as np

from ..._core._casa import _ensure_casa
from ...utils import _ensure_import, _progress_bar, _ensure_original_video, _store_normalization_results


def clahe(
    casa: dict[str, Any],
    *,
    overwrite: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Normalize frames using CLAHE (contrast-limited adaptive histogram equalization).

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        overwrite (bool, optional):
            If ``True``, replace ``casa["video"]["original_video"]`` with normalized
            output. If ``False``, keep the original array unchanged.
        clip_limit (float, optional):
            CLAHE clip limit passed to OpenCV.
        tile_grid_size (tuple[int, int], optional):
            Tile grid size passed to OpenCV CLAHE.
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
        ImportError:
            If ``opencv-python`` is unavailable.

    Notes:
        - CLAHE is applied per frame.
        - For color frames, CLAHE is applied channel-wise.
        - Output is stored in ``casa["video"]["normalized_video"]``.
        - Method tag is stored in ``casa["video"]["normalized_type"] = "clahe"``.
        - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
        - Updates ``casa["meta"]["last_preprocessing"]`` with method metadata.

    Examples:
        >>> import pycasa_as as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.preprocessing.normalization.clahe(overwrite=False)
    """
    casa = _ensure_casa(casa)
    original_video = _ensure_original_video(casa)
    if original_video.ndim not in (3, 4):
        raise ValueError(f"Unsupported video shape: {original_video.shape}")
    if verbose:
        print(
            "Running normalization clahe on frames "
            f"(overwrite={overwrite}, clip_limit={float(clip_limit):.3f}, "
            f"tile_grid_size={tuple(tile_grid_size)})..."
        )

    cv2 = _ensure_import("cv2", pip_name="opencv-python")
    clahe_engine = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )

    frame_count = int(original_video.shape[0])
    normalized_frames: list[np.ndarray] = []
    for frame in _progress_bar(
        original_video,
        total=frame_count,
        desc="Normalization clahe",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        if frame.ndim == 2:
            normalized_frames.append(clahe_engine.apply(frame.astype(np.uint8)))
        else:
            channels = [
                clahe_engine.apply(frame[..., idx].astype(np.uint8))
                for idx in range(frame.shape[-1])
            ]
            normalized_frames.append(np.stack(channels, axis=-1))

    normalized = np.stack(normalized_frames, axis=0)
    return _store_normalization_results(
        casa,
        method="clahe",
        normalized=normalized,
        overwrite=overwrite,
        verbose=verbose,
    )
