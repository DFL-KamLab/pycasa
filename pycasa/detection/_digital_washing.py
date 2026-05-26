from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import (
    _clear_predicted_detections,
    _convert_video_to_grayscale,
    _ensure_import,
    _ensure_original_video,
    _ensure_video_dimensions,
    _predicted_detection_keys,
    _progress_bar,
    _resolve_sort_track_sources,
    _warn_yellow,
)
from ._detect_moving_cells import _gaussian_mixture_motion_filter


def _safe_log_hu(values: np.ndarray) -> np.ndarray:
    """Return log-scaled Hu moments with zero-safe epsilon handling."""
    eps = 1e-30
    x = np.asarray(values, dtype=float)
    x = np.where(np.abs(x) < eps, eps, x)
    return -np.sign(x) * np.log10(np.abs(x))


def _cent_moment(p: int, q: int, image: np.ndarray) -> float:
    """Compute normalized central image moment used by Hu features."""
    image_array = np.asarray(image, dtype=float)
    if image_array.ndim != 2:
        raise ValueError("Central moments require a 2D array.")

    m, n = image_array.shape
    moo = float(np.sum(image_array))
    if moo == 0:
        return 0.0

    yy, xx = np.meshgrid(np.arange(n, dtype=float), np.arange(m, dtype=float))
    m1o = float(np.sum(xx * image_array))
    mo1 = float(np.sum(yy * image_array))
    x_bar = m1o / moo
    y_bar = mo1 / moo

    mu_pq = float(np.sum(((xx - x_bar) ** p) * ((yy - y_bar) ** q) * image_array))
    gamma = 0.5 * (p + q) + 1.0
    return float(mu_pq / (moo**gamma))


def _feature_vector(image: np.ndarray) -> np.ndarray:
    """Build seven Hu-like invariant moments from one image patch."""
    n20 = _cent_moment(2, 0, image)
    n02 = _cent_moment(0, 2, image)
    n30 = _cent_moment(3, 0, image)
    n12 = _cent_moment(1, 2, image)
    n21 = _cent_moment(2, 1, image)
    n03 = _cent_moment(0, 3, image)
    n11 = _cent_moment(1, 1, image)

    m1 = n20 + n02
    m2 = (n20 - n02) ** 2 + 4 * (n11**2)
    m3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    m4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    m5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (
        3 * n21 - n03
    ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    m6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (
        n21 + n03
    )
    m7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - (
        n30 - 3 * n12
    ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    return np.asarray([m1, m2, m3, m4, m5, m6, m7], dtype=float)


def _sauvola_threshold(image: np.ndarray, window: tuple[int, int], k: float) -> np.ndarray:
    """Apply Sauvola thresholding and return a boolean foreground map."""
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    uniform_filter = scipy_ndimage.uniform_filter

    image_array = np.asarray(image, dtype=float)
    mean = uniform_filter(image_array, size=window, mode="nearest")
    mean_square = uniform_filter(image_array * image_array, size=window, mode="nearest")
    deviation = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    dynamic_range = float(np.max(deviation))
    if dynamic_range <= 0:
        return np.zeros(image_array.shape, dtype=bool)
    threshold = mean * (1 + k * (deviation / dynamic_range - 1))
    return image_array >= threshold


def _log_based_binarization(frame: np.ndarray) -> np.ndarray:
    """Background binarization via LoG + Otsu (Section 3.1.2 of paper)."""
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    skimage_filters = _ensure_import("skimage.filters", pip_name="scikit-image")
    skimage_morphology = _ensure_import("skimage.morphology", pip_name="scikit-image")
    skimage_segmentation = _ensure_import(
        "skimage.segmentation", pip_name="scikit-image"
    )
    skimage_util = _ensure_import("skimage.util", pip_name="scikit-image")

    binary_fill_holes = scipy_ndimage.binary_fill_holes
    gaussian_laplace = scipy_ndimage.gaussian_laplace
    threshold_otsu = skimage_filters.threshold_otsu
    binary_dilation = skimage_morphology.binary_dilation
    binary_erosion = skimage_morphology.binary_erosion
    disk = skimage_morphology.disk
    clear_border = skimage_segmentation.clear_border
    img_as_float = skimage_util.img_as_float

    image = np.asarray(frame, dtype=np.uint8)
    image_float = img_as_float(image)
    # Paper: LoG filter (10×10 kernel, σ²=0.4 → σ≈0.632).
    # gaussian_laplace gives negative response at bright blob centers → negate.
    sigma_log = float(np.sqrt(0.4))
    image_log = -gaussian_laplace(image_float, sigma=sigma_log)
    image_log_pos = np.clip(image_log, 0.0, None)
    mx = float(image_log_pos.max())
    image_log_u8 = ((image_log_pos / mx) * 255).astype(np.uint8) if mx > 0 else image_log_pos.astype(np.uint8)
    # Paper: Otsu threshold first, then fill holes, then erosion + dilation with 2×2 disk.
    thr = float(threshold_otsu(image_log_u8)) if int(image_log_u8.max()) > 0 else 0.0
    binary = image_log_u8 > thr
    binary_filled = binary_fill_holes(binary)
    eroded = binary_erosion(binary_filled, disk(2))
    dilated = binary_dilation(eroded, disk(2))
    return clear_border(dilated)


def _remove_motion_regions(motion: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Remove connected components in ``frame`` that overlap motion mask."""
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    skimage_measure = _ensure_import("skimage.measure", pip_name="scikit-image")
    label = scipy_ndimage.label
    regionprops = skimage_measure.regionprops

    output = frame.astype(bool).copy()
    structure = np.ones((3, 3), dtype=int)
    motion_labels, _ = label(motion.astype(np.uint8), structure=structure)
    frame_labels, _ = label(frame.astype(np.uint8), structure=structure)

    motion_props = regionprops(motion_labels)
    frame_props = regionprops(frame_labels)
    if not motion_props or not frame_props:
        return output

    frame_x_sets = [set(int(coord[1]) for coord in prop.coords) for prop in frame_props]
    frame_y_sets = [set(int(coord[0]) for coord in prop.coords) for prop in frame_props]
    for motion_prop in motion_props:
        motion_x = set(int(coord[1]) for coord in motion_prop.coords)
        motion_y = set(int(coord[0]) for coord in motion_prop.coords)
        for index, frame_prop in enumerate(frame_props):
            if motion_x.intersection(frame_x_sets[index]) and motion_y.intersection(frame_y_sets[index]):
                output[frame_prop.coords[:, 0], frame_prop.coords[:, 1]] = False
    return output


def _motion_background_separation(
    motion: np.ndarray,
    entire: np.ndarray,
    number_training_frames: int,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """Separate background candidates by removing motion-overlapping blobs."""
    num_frames = int(entire.shape[0])
    output = np.zeros(entire.shape, dtype=bool)
    start_index = int(max(0, min(number_training_frames, num_frames)))
    for frame_index in _progress_bar(
        range(start_index, num_frames),
        total=max(0, num_frames - start_index),
        desc="Digital Washing motion/background separation",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        output[frame_index] = _remove_motion_regions(motion[frame_index], entire[frame_index])
    return output


def _create_washed_video(
    original_video: np.ndarray,
    motion_mask: np.ndarray,
    decisions: np.ndarray,
    initial_frame: int,
    number_training_frames: int,
) -> np.ndarray:
    """Return original video with non-sperm regions blacked out (Fig. 12c of paper).

    SM class (motion pixels) and SD class (accepted background detections) are
    preserved; class O pixels are set to zero.
    """
    from skimage.draw import ellipse as draw_ellipse

    num_frames = int(original_video.shape[0])
    H = int(original_video.shape[1])
    W = int(original_video.shape[2])
    washed = np.zeros_like(original_video)
    start_local = int(number_training_frames)

    for local_idx in range(start_local, num_frames):
        global_frame = initial_frame + local_idx
        sm_mask = motion_mask[local_idx].astype(bool)

        sd_mask = np.zeros((H, W), dtype=bool)
        if decisions.size > 0:
            frame_dets = decisions[decisions[:, 2] == float(global_frame)]
            for det in frame_dets:
                x_col, y_row = float(det[0]), float(det[1])
                minor_r, major_r = float(det[3]), float(det[4])
                r_radius = max(int(round(minor_r * 1.5)), 5)
                c_radius = max(int(round(major_r * 1.5)), 5)
                rr, cc = draw_ellipse(
                    int(round(y_row)) - 1,
                    int(round(x_col)) - 1,
                    r_radius,
                    c_radius,
                    shape=(H, W),
                )
                sd_mask[rr, cc] = True

        combined = sm_mask | sd_mask
        if original_video.ndim == 4:
            washed[local_idx] = original_video[local_idx] * combined[:, :, np.newaxis]
        else:
            washed[local_idx] = original_video[local_idx] * combined

    return washed


def _crop_around_detected_cells(gray_image: np.ndarray, x_center: float, y_center: float) -> np.ndarray:
    """Crop a fixed 60x60 patch around the detected centroid."""
    rows, cols = gray_image.shape
    y_start = max(int(np.floor(x_center - 30.0)), 1)
    y_end = min(int(np.ceil(x_center + 30.0)), cols)
    x_start = max(int(np.floor(y_center - 30.0)), 1)
    x_end = min(int(np.ceil(y_center + 30.0)), rows)
    return gray_image[x_start - 1 : x_end, y_start - 1 : y_end]


def _features_and_detections(
    gray_image: np.ndarray,
    bin_image: np.ndarray,
    frame_idx: int,
    blob_min_pixel_area: int,
) -> np.ndarray:
    """Extract feature rows ``[x, y, half_w, half_h, aspect_ratio, hu_sum, frame]``."""
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    skimage_filters = _ensure_import("skimage.filters", pip_name="scikit-image")
    skimage_measure = _ensure_import("skimage.measure", pip_name="scikit-image")
    skimage_morphology = _ensure_import("skimage.morphology", pip_name="scikit-image")
    skimage_transform = _ensure_import("skimage.transform", pip_name="scikit-image")

    binary_fill_holes = scipy_ndimage.binary_fill_holes
    label = scipy_ndimage.label
    unsharp_mask = skimage_filters.unsharp_mask
    regionprops = skimage_measure.regionprops
    binary_dilation = skimage_morphology.binary_dilation
    binary_erosion = skimage_morphology.binary_erosion
    disk = skimage_morphology.disk
    remove_small_objects = skimage_morphology.remove_small_objects
    resize = skimage_transform.resize

    structure = np.ones((3, 3), dtype=int)
    label_matrix, _ = label(bin_image.astype(np.uint8), structure=structure)
    properties = regionprops(label_matrix)

    rows: list[list[float]] = []
    for region in properties:
        if int(region.area) < int(blob_min_pixel_area):
            continue

        centroid_row, centroid_col = region.centroid
        x_center = float(centroid_col + 1.0)
        y_center = float(centroid_row + 1.0)

        cropped = _crop_around_detected_cells(gray_image, x_center, y_center)
        if cropped.size == 0:
            continue

        resized = resize(
            cropped.astype(float),
            (int(cropped.shape[0] * 3), int(cropped.shape[1] * 3)),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        )
        sharpened = unsharp_mask(resized / 255.0, radius=30.0, amount=2.0, preserve_range=True) * 255.0

        binary = np.logical_not(_sauvola_threshold(sharpened, window=(20, 20), k=0.15))
        binary = binary_dilation(binary, disk(3))
        binary = binary_fill_holes(binary)
        binary = binary_erosion(binary, disk(9))
        binary = remove_small_objects(binary.astype(bool), min_size=50)
        if not np.any(binary):
            continue

        local_labels, _ = label(binary.astype(np.uint8), structure=structure)
        local_props = regionprops(local_labels)
        if not local_props:
            continue

        if len(local_props) > 1:
            distances = []
            for prop in local_props:
                row, col = prop.centroid
                distances.append((col + 1.0 - 75.0) ** 2 + (row + 1.0 - 75.0) ** 2)
            chosen_index = int(np.argmin(np.asarray(distances)))
        else:
            chosen_index = 0
        chosen = local_props[chosen_index]

        crop_min_r, crop_min_c, crop_max_r, crop_max_c = chosen.bbox
        half_w = float((crop_max_c - crop_min_c) / 2.0 / 3.0)
        half_h = float((crop_max_r - crop_min_r) / 2.0 / 3.0)
        if half_w <= 0 or half_h <= 0:
            continue

        hu_raw = _feature_vector(sharpened.astype(float))
        hu_vector = _safe_log_hu(hu_raw)
        hu_features = float(np.sum(np.abs(hu_vector)))

        rows.append(
            [
                x_center,
                y_center,
                half_w,
                half_h,
                float(half_w / half_h),
                hu_features,
                float(frame_idx),
            ]
        )

    if rows:
        return np.asarray(rows, dtype=float)
    return np.empty((0, 7), dtype=float)


def _features_calculation(
    video_orig: np.ndarray,
    video_bin: np.ndarray,
    number_training_frames: int,
    blob_min_pixel_area: int,
    initial_frame: int,
    *,
    progress_desc: str,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract Digital Washing feature rows across selected frame range."""
    num_frames = int(video_orig.shape[0])
    start_index = int(max(0, min(number_training_frames, num_frames)))
    output = np.empty((0, 7), dtype=float)
    for frame_index in _progress_bar(
        range(start_index, num_frames),
        total=max(0, num_frames - start_index),
        desc=progress_desc,
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        global_frame = initial_frame + frame_index
        rows = _features_and_detections(
            gray_image=video_orig[frame_index],
            bin_image=video_bin[frame_index],
            frame_idx=global_frame,
            blob_min_pixel_area=blob_min_pixel_area,
        )
        if rows.size > 0:
            output = np.vstack((output, rows))
    return output


def _local_detectors_decisions(
    motion_features: np.ndarray,
    background_features: np.ndarray,
    number_training_frames: int,
    initial_frame: int,
    final_frame: int,
    std_val: float,
) -> np.ndarray:
    """Fuse motion/background features into final candidate detections."""
    if motion_features.size == 0:
        return np.empty((0, 5), dtype=float)

    var1 = np.abs(motion_features[:, 2])
    var2 = np.abs(motion_features[:, 4])
    var3 = np.abs(motion_features[:, 5])

    mu1, sd1 = float(np.mean(var1)), float(np.std(var1))
    mu2, sd2 = float(np.mean(var2)), float(np.std(var2))
    mu3, sd3 = float(np.mean(var3)), float(np.std(var3))

    range1 = (mu1 - std_val * sd1, mu1 + std_val * sd1)
    range2 = (mu2 - std_val * sd2, mu2 + std_val * sd2)
    range3 = (mu3 - std_val * sd3, mu3 + std_val * sd3)

    rows: list[list[float]] = []
    start_frame = int(initial_frame + number_training_frames)
    for frame in range(start_frame, int(final_frame) + 1):
        if background_features.size > 0:
            background_subset = background_features[background_features[:, -1] == frame, :6]
            if background_subset.size > 0:
                test1 = (background_subset[:, 2] > range1[0]) & (background_subset[:, 2] < range1[1])
                test2 = (background_subset[:, 4] > range2[0]) & (background_subset[:, 4] < range2[1])
                test3 = (background_subset[:, 5] > range3[0]) & (background_subset[:, 5] < range3[1])
                keep = test1 & test2 & test3
                for item in background_subset[keep]:
                    rows.append([item[0], item[1], float(frame), item[2], item[3]])

        motion_subset = motion_features[motion_features[:, -1] == frame, :]
        if motion_subset.size > 0:
            for item in motion_subset:
                rows.append([item[0], item[1], float(frame), item[2], item[3]])

    if rows:
        return np.asarray(rows, dtype=float)
    return np.empty((0, 5), dtype=float)


def _binarize_video_log_method(video: np.ndarray, *, show_progress: bool = True) -> np.ndarray:
    """Run log-based binarization frame-by-frame over a grayscale video."""
    output = np.zeros(video.shape, dtype=bool)
    for frame_index in _progress_bar(
        range(video.shape[0]),
        total=int(video.shape[0]),
        desc="Digital Washing binarization",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        output[frame_index] = _log_based_binarization(video[frame_index])
    return output


def digital_washing(
    casa: dict[str, Any],
    motion_threshold: float = 3.0,
    number_training_frames: int = 20,
    blob_min_pixel_area: int = 20,
    k_val: float = 1.7,
    border_margin_px: int = 20,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Digital Washing detection and store one active predicted result.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary containing ``casa["video"]["original_video"]``.
        motion_threshold (float, optional):
            Sigma threshold for Gaussian-mixture motion extraction.
        number_training_frames (int, optional):
            Warm-up frame count used by motion/background separation.
        blob_min_pixel_area (int, optional):
            Minimum connected-component area kept during feature extraction.
        k_val (float, optional):
            Standard-deviation multiplier used in local detector rules.
        border_margin_px (int, optional):
            Border exclusion margin in pixels.
        show_progress (bool, optional):
            If ``True``, show shared pycasa progress bars for iterative stages.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries. If
            ``False``, suppress those summaries. Overwrite warnings are not
            affected by this flag.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with detections under
            ``casa["detections"]["digital_washing"]``.

    Raises:
        TypeError:
            If ``casa["video"]["original_video"]`` is not a numpy array.
        ValueError:
            If ``casa["video"]["original_video"]`` exists but has unsupported
            dimensions.

    Notes:
        - Previous predicted detection outputs are overwritten using the
          single-result policy.
        - Detection rows follow normalized legacy format
          ``[label, norm_x, norm_y, norm_w, norm_h]``.
        - Intermediate videos are stored as:
          ``digital_washing_motion_video``,
          ``digital_washing_binarized_video``,
          ``digital_washing_background_video``.
        - ``casa["meta"]["last_detection"]`` is always updated.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data()
        >>> session = session.detection.digital_washing(show_progress=False)
    """
    casa = _ensure_casa(casa)
    detections_root = casa.setdefault("detections", {})
    existing_predicted_methods = _predicted_detection_keys(detections_root)
    if existing_predicted_methods:
        _warn_yellow(
            "Previous detection result overwritten "
            f"({', '.join(existing_predicted_methods)} -> digital_washing)."
        )
    _clear_predicted_detections(detections_root)

    if verbose:
        print("Running digital-washing detection on frames...")

    if casa.get("video", {}).get("original_video") is None:
        if verbose:
            print("Skipping digital-washing detection: no original video is loaded.")
        detections_root["digital_washing"] = {}
        casa["meta"]["last_detection"] = {
            "backend": "digital_washing",
            "motion_threshold": float(motion_threshold),
            "number_training_frames": int(number_training_frames),
            "blob_min_pixel_area": int(blob_min_pixel_area),
            "k_val": float(k_val),
            "border_margin_px": int(border_margin_px),
            "skipped": True,
            "reason": "missing_original_video",
        }
        return casa

    original_video = _ensure_original_video(casa)
    num_frames, frame_height, frame_width = _ensure_video_dimensions(original_video)
    if num_frames == 0:
        detections_root["digital_washing"] = {}
        casa["meta"]["last_detection"] = {
            "backend": "digital_washing",
            "motion_threshold": float(motion_threshold),
            "number_training_frames": int(number_training_frames),
            "blob_min_pixel_area": int(blob_min_pixel_area),
            "k_val": float(k_val),
            "border_margin_px": int(border_margin_px),
            "skipped": True,
            "reason": "empty_original_video",
        }
        return casa

    grayscale_video = _convert_video_to_grayscale(original_video)
    motion_video, _ = _gaussian_mixture_motion_filter(
        original_video,
        num_train=int(number_training_frames),
        threshold=float(motion_threshold),
        show_progress=bool(show_progress),
    )
    motion_mask = motion_video > 0

    binarized_video = _binarize_video_log_method(
        grayscale_video,
        show_progress=bool(show_progress),
    )
    background_video = _motion_background_separation(
        motion=motion_mask,
        entire=binarized_video,
        number_training_frames=int(number_training_frames),
        show_progress=bool(show_progress),
    )

    initial_frame = int(casa.get("video", {}).get("initial_frame", 0) or 0)
    final_frame = int(initial_frame + num_frames - 1)

    motion_features = _features_calculation(
        video_orig=grayscale_video,
        video_bin=motion_mask,
        number_training_frames=int(number_training_frames),
        blob_min_pixel_area=int(blob_min_pixel_area),
        initial_frame=initial_frame,
        progress_desc="Digital Washing motion features extraction",
        show_progress=bool(show_progress),
    )
    background_features = _features_calculation(
        video_orig=grayscale_video,
        video_bin=background_video,
        number_training_frames=int(number_training_frames),
        blob_min_pixel_area=int(blob_min_pixel_area),
        initial_frame=initial_frame,
        progress_desc="Digital Washing background features extraction",
        show_progress=bool(show_progress),
    )

    decisions = _local_detectors_decisions(
        motion_features=motion_features,
        background_features=background_features,
        number_training_frames=int(number_training_frames),
        initial_frame=initial_frame,
        final_frame=final_frame,
        std_val=float(k_val),
    )

    detections: dict[str, list[list[str]]] = {}
    start_frame = initial_frame + int(number_training_frames)
    for frame in range(start_frame, final_frame + 1):
        frame_rows: list[list[str]] = []
        if decisions.size > 0:
            frame_decisions = decisions[decisions[:, 2] == float(frame)]
            if frame_decisions.size > 0:
                x_vals = frame_decisions[:, 0]
                y_vals = frame_decisions[:, 1]
                keep = (
                    (x_vals >= (int(border_margin_px) + 1))
                    & (x_vals <= (frame_width - int(border_margin_px)))
                    & (y_vals >= (int(border_margin_px) + 1))
                    & (y_vals <= (frame_height - int(border_margin_px)))
                )
                for item in frame_decisions[keep]:
                    bbox_width = max(float(item[3] * 2.0), 1.0)
                    bbox_height = max(float(item[4] * 2.0), 1.0)
                    frame_rows.append(
                        [
                            "0",
                            str(round(float(item[0]) / frame_width, 6)),
                            str(round(float(item[1]) / frame_height, 6)),
                            str(round(min(bbox_width / frame_width, 1.0), 6)),
                            str(round(min(bbox_height / frame_height, 1.0), 6)),
                        ]
                    )
        detections[str(frame)] = frame_rows

    washed_video = _create_washed_video(
        original_video=original_video,
        motion_mask=motion_mask,
        decisions=decisions,
        initial_frame=initial_frame,
        number_training_frames=int(number_training_frames),
    )

    motion_uint8 = motion_mask.astype(np.uint8) * 255
    binarized_uint8 = binarized_video.astype(np.uint8) * 255
    background_uint8 = background_video.astype(np.uint8) * 255
    casa["video"]["digital_washing_motion_video"] = motion_uint8
    casa["video"]["digital_washing_binarized_video"] = binarized_uint8
    casa["video"]["digital_washing_background_video"] = background_uint8
    casa["video"]["digital_washing_washed_video"] = washed_video
    detections_root["digital_washing"] = detections
    tracked_sources = _resolve_sort_track_sources(casa.get("tracks", {}))
    if tracked_sources and "digital_washing" not in tracked_sources:
        _warn_yellow(
            "Detections were updated to 'digital_washing'. "
            "Re-run tracking to generate detection tracks."
        )

    detections_found = int(sum(len(rows) for rows in detections.values()))
    frames_with_detections = int(sum(1 for rows in detections.values() if rows))
    casa["meta"]["last_detection"] = {
        "backend": "digital_washing",
        "motion_threshold": float(motion_threshold),
        "number_training_frames": int(number_training_frames),
        "blob_min_pixel_area": int(blob_min_pixel_area),
        "k_val": float(k_val),
        "border_margin_px": int(border_margin_px),
        "frames_processed": int(num_frames),
        "motion_feature_rows": int(motion_features.shape[0]),
        "background_feature_rows": int(background_features.shape[0]),
        "decision_rows": int(decisions.shape[0]),
        "frames_with_detections": frames_with_detections,
        "detections_found": detections_found,
        "skipped": False,
    }

    if verbose:
        print(
            "Digital-washing summary: "
            f"frames_processed={int(num_frames)}, "
            f"frames_with_detections={frames_with_detections}, "
            f"detections_found={detections_found}, "
            f"motion_feature_rows={int(motion_features.shape[0])}, "
            f"background_feature_rows={int(background_features.shape[0])}"
        )
    return casa
