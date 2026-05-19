from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.ndimage import label as ndimage_label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, diamond

from .._core._casa import _ensure_casa
from ..utils import (
    _clear_predicted_detections,
    _convert_video_to_grayscale,
    _ensure_original_video,
    _ensure_video_dimensions,
    _predicted_detection_keys,
    _progress_bar,
    _resolve_sort_track_sources,
    _warn_yellow,
)

# 8-connectivity structure for connected-component labeling
_STRUCT_8 = np.ones((3, 3), dtype=int)


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


def urbano_detection(
    casa: dict[str, Any],
    weight: float = 1.0,
    gaussian_size: int = 11,
    gaussian_iters: int = 5,
    log_size: int = 9,
    min_pixels: int = 5,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Detect sperm cells using the Urbano et al. (2017) LoG pipeline.

    Implements the full seven-step segmentation pipeline from:
    Urbano et al., *Automatic Sperm Motility Analysis*, IEEE TMI 36(3), 2017.

    Steps: Gaussian (×N) → LoG → Otsu×weight → erosion → dilation →
    connected-component labeling (8-connectivity) → centroid extraction.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        weight (float, optional):
            Multiplier applied to Otsu's per-frame threshold. Values > 1
            raise the threshold (fewer detections); values < 1 lower it.
        gaussian_size (int, optional):
            Side length in pixels of the Gaussian kernel (paper: 11).
        gaussian_iters (int, optional):
            Number of times the Gaussian filter is applied (paper: 5).
        log_size (int, optional):
            Side length in pixels of the LoG kernel (paper: 9).
        min_pixels (int, optional):
            Minimum connected-component area in pixels to keep as a detection
            (paper: 5). Smaller groups are discarded.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while processing
            frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with detections stored in
            ``casa['detections']['urbano_detection']``.

    Raises:
        ValueError:
            If ``casa["video"]["original_video"]`` is missing or has invalid shape.

    Notes:
        - Detection positions are the centroids of connected pixel groups.
        - Bounding boxes are the actual pixel extents of each group.
        - Coordinates are stored normalized (0–1) by frame width/height.
        - ``casa['meta']['last_detection']`` is always updated.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> session = session.detection.urbano_detection()
    """
    casa = _ensure_casa(casa)
    detections_root = casa.setdefault("detections", {})
    existing_predicted_methods = _predicted_detection_keys(detections_root)
    if existing_predicted_methods:
        _warn_yellow(
            "Previous detection result overwritten "
            f"({', '.join(existing_predicted_methods)} -> urbano_detection)."
        )
    _clear_predicted_detections(detections_root)

    _skipped_meta = {
        "backend": "urbano_detection",
        "weight": float(weight),
        "gaussian_size": int(gaussian_size),
        "gaussian_iters": int(gaussian_iters),
        "log_size": int(log_size),
        "min_pixels": int(min_pixels),
        "skipped": True,
    }

    if casa.get("video", {}).get("original_video") is None:
        if verbose:
            print("Skipping Urbano detection: no original video is loaded.")
        detections_root["urbano_detection"] = {}
        casa["meta"]["last_detection"] = {**_skipped_meta, "reason": "missing_original_video"}
        return casa

    original_video = _ensure_original_video(casa)
    num_frames, frame_height, frame_width = _ensure_video_dimensions(original_video)
    if num_frames == 0:
        detections_root["urbano_detection"] = {}
        casa["meta"]["last_detection"] = {**_skipped_meta, "reason": "empty_original_video"}
        return casa

    if verbose:
        print("Running Urbano detection on frames...")

    grayscale_video = _convert_video_to_grayscale(original_video)
    initial_frame = int(casa.get("video", {}).get("initial_frame", 0) or 0)

    detections: dict[str, list[list[str]]] = {}
    for local_idx in _progress_bar(
        range(num_frames),
        total=num_frames,
        desc="Detection urbano",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        global_frame = initial_frame + local_idx
        gray = grayscale_video[local_idx]
        binary = _urbano_binarize_frame(
            gray,
            gaussian_size=gaussian_size,
            gaussian_iters=gaussian_iters,
            log_size=log_size,
            weight=weight,
        )

        # Steps 6–7: connected-component labeling then centroid extraction
        labeled, _ = ndimage_label(binary, structure=_STRUCT_8)
        frame_rows: list[list[str]] = []
        for prop in regionprops(labeled):
            if prop.area < min_pixels:
                continue
            cy, cx = prop.centroid          # row=y, col=x
            min_r, min_c, max_r, max_c = prop.bbox
            w = float(max_c - min_c)
            h = float(max_r - min_r)
            frame_rows.append([
                "0",
                str(round(cx / frame_width, 6)),
                str(round(cy / frame_height, 6)),
                str(round(min(w / frame_width, 1.0), 6)),
                str(round(min(h / frame_height, 1.0), 6)),
            ])
        detections[str(global_frame)] = frame_rows

    detections_root["urbano_detection"] = detections

    tracked_sources = _resolve_sort_track_sources(casa.get("tracks", {}))
    if tracked_sources and "urbano_detection" not in tracked_sources:
        _warn_yellow(
            "Detections were updated to 'urbano_detection'. "
            "Re-run tracking to generate detection tracks."
        )

    detections_found = int(sum(len(rows) for rows in detections.values()))
    frames_with_detections = int(sum(1 for rows in detections.values() if rows))
    casa["meta"]["last_detection"] = {
        "backend": "urbano_detection",
        "weight": float(weight),
        "gaussian_size": int(gaussian_size),
        "gaussian_iters": int(gaussian_iters),
        "log_size": int(log_size),
        "min_pixels": int(min_pixels),
        "frames_processed": int(num_frames),
        "frames_with_detections": frames_with_detections,
        "detections_found": detections_found,
        "skipped": False,
    }

    if verbose:
        print(
            "Urbano detection summary: "
            f"frames_processed={int(num_frames)}, "
            f"frames_with_detections={frames_with_detections}, "
            f"detections_found={detections_found}"
        )

    return casa
