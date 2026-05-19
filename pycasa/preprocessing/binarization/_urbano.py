from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, diamond

from ..._core._casa import _ensure_casa
from ...utils import (
    _convert_video_to_grayscale,
    _ensure_original_video,
    _progress_bar,
    _store_binarization_results,
)


def _urbano_binarize_frame(
    gray_frame: np.ndarray,
    gaussian_size: int = 11,
    gaussian_iters: int = 5,
    log_size: int = 9,
    weight: float = 1.0,
) -> np.ndarray:
    """Apply Urbano et al. (2017) steps 1–5 to a single grayscale frame."""
    # Step 1: Gaussian filter applied N times.
    # truncate=3.0 -> kernel radius = round(3*sigma), giving gaussian_size x gaussian_size kernel.
    sigma_g = (gaussian_size - 1) / 6.0
    img = gray_frame.astype(np.float64)
    for _ in range(gaussian_iters):
        img = gaussian_filter(img, sigma=sigma_g, truncate=3.0)

    # Step 2: LoG (Mexican-hat) filter.
    # gaussian_laplace computes the true Laplacian-of-Gaussian in one step at the correct scale.
    # Cells are bright on dark background -> gaussian_laplace is negative at cell centers -> negate.
    sigma_log = (log_size - 1) / 6.0
    img = -gaussian_laplace(img, sigma=sigma_log, truncate=3.0)

    # Step 3: per-frame Otsu threshold x user weight factor (positive half only)
    img_pos = np.clip(img, 0, None)
    mx = float(img_pos.max())
    img_u8 = ((img_pos / mx) * 255).astype(np.uint8) if mx > 0 else img_pos.astype(np.uint8)
    thr = float(threshold_otsu(img_u8)) * weight if int(img_u8.max()) > 0 else 0.0
    binary = img_u8 > thr

    # Step 4: morphological erosion with 5x5 diamond (radius 2)
    binary = binary_erosion(binary, diamond(2))

    # Step 5: morphological dilation with 3x3 diamond (radius 1)
    binary = binary_dilation(binary, diamond(1))

    return binary.astype(np.uint8)


def urbano(
    casa: dict[str, Any],
    weight: float = 1.0,
    gaussian_size: int = 11,
    gaussian_iters: int = 5,
    log_size: int = 9,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Urbano et al. (2017) binarization pipeline on the loaded video.

    Implements the five-step segmentation preprocessing described in:
    Urbano et al., *Automatic Sperm Motility Analysis*, IEEE TMI 36(3), 2017.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        weight (float, optional):
            Multiplier applied to Otsu's threshold (per frame). Values > 1
            raise the threshold (fewer detections); values < 1 lower it.
        gaussian_size (int, optional):
            Side length in pixels of the Gaussian kernel (paper: 11).
        gaussian_iters (int, optional):
            Number of times the Gaussian filter is applied (paper: 5).
        log_size (int, optional):
            Side length in pixels of the LoG kernel (paper: 9).
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            preprocessing step.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with binary results stored under
            ``casa["video"]["binary_video"]`` and
            ``casa["video"]["binary_type"] = "urbano"``.

    Raises:
        ValueError:
            If ``casa["video"]["original_video"]`` is missing or has invalid shape.
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.

    Notes:
        - Pipeline: Gaussian (×N) → LoG → Otsu×weight → erosion 5×5 → dilation 3×3.
        - Structuring elements are diamond-shaped, as specified in the paper.
        - Updates ``casa["meta"]["last_preprocessing"]``.

    Examples:
        >>> import pycasa as pc
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
        binary_frames.append(
            _urbano_binarize_frame(
                frame,
                gaussian_size=gaussian_size,
                gaussian_iters=gaussian_iters,
                log_size=log_size,
                weight=weight,
            )
        )
    binary_video = np.stack(binary_frames, axis=0)
    return _store_binarization_results(
        casa,
        method="urbano",
        binary_video=binary_video,
        verbose=verbose,
    )
