from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import (
    _clear_predicted_detections,
    _convert_video_to_grayscale,
    _ensure_import,
    _ensure_original_video,
    _predicted_detection_keys,
    _ensure_video_dimensions,
    _progress_bar,
    _resolve_sort_track_sources,
    _warn_yellow,
)


def _update_mean_variance(
    curr_mean: np.ndarray,
    curr_variance: np.ndarray,
    data_point_to_drop: np.ndarray,
    data_point_to_add: np.ndarray,
    nbr_training_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy running mean/variance update used by the gm method."""
    diff_old_curr_data_point = data_point_to_drop - curr_mean
    curr_mean = curr_mean - diff_old_curr_data_point / nbr_training_frames
    curr_variance = (
        curr_variance
        - (diff_old_curr_data_point**2) / nbr_training_frames
        + curr_variance / (nbr_training_frames - 1)
    )

    diff_new_curr_data_point = data_point_to_add - curr_mean
    new_mean = curr_mean + diff_new_curr_data_point / nbr_training_frames
    new_variance = (
        curr_variance
        + (diff_new_curr_data_point**2) / nbr_training_frames
        - curr_variance / (nbr_training_frames - 1)
    )
    return new_mean, new_variance


def _matlab_disk(radius: int) -> np.ndarray:
    """Replicate MATLAB-style disk structuring element from legacy code."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (x**2 + y**2) <= (radius + 1.5) ** 2
    return disk.astype(np.uint8)


def _gaussian_mixture_motion_filter(
    original_video: np.ndarray,
    num_train: int = 20,
    threshold: float = 3,
    med_filter_size: int = 3,
    disk_size: int = 6,
    *,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of legacy gaussian_mixture_motion_filter helper."""
    scipy_signal = _ensure_import("scipy.signal", pip_name="scipy")
    medfilt2d = scipy_signal.medfilt2d
    skimage_morphology = _ensure_import("skimage.morphology", pip_name="scikit-image")
    skimage_segmentation = _ensure_import(
        "skimage.segmentation", pip_name="scikit-image"
    )
    dilation = skimage_morphology.dilation
    erosion = skimage_morphology.erosion
    clear_border = skimage_segmentation.clear_border

    grayscale_video = _convert_video_to_grayscale(original_video).astype(np.float64) / 255.0
    num_frames, height, width = grayscale_video.shape

    video_float = np.zeros((num_frames, height, width), dtype=np.float64)
    med_filtered_original_video = np.zeros((num_frames, height, width), dtype=np.float64)
    binarized_moving_cells_video = np.zeros((num_frames, height, width), dtype=np.float64)

    for curr_frame in _progress_bar(
        range(num_frames),
        total=num_frames,
        desc="Gaussian mixture preparation",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        med_filtered_original_video[curr_frame, :, :] = medfilt2d(
            grayscale_video[curr_frame, :, :], kernel_size=med_filter_size
        )
        video_float[curr_frame, :, :] = grayscale_video[curr_frame, :, :]

    if num_frames == 0:
        return binarized_moving_cells_video, med_filtered_original_video

    train_frames = int(max(1, min(num_train, num_frames)))
    mean_original_video = np.mean(video_float[:train_frames, :, :], axis=0)
    mean_med_filtered_original_video = np.mean(
        med_filtered_original_video[:train_frames, :, :], axis=0
    )

    if train_frames > 1:
        var_original_video = np.var(video_float[:train_frames, :, :], axis=0, ddof=1)
        var_med_filtered_original_video = np.var(
            med_filtered_original_video[:train_frames, :, :], axis=0, ddof=1
        )
    else:
        var_original_video = np.zeros_like(mean_original_video)
        var_med_filtered_original_video = np.zeros_like(mean_med_filtered_original_video)

    for j in _progress_bar(
        range(train_frames, num_frames),
        total=max(0, num_frames - train_frames),
        desc="Gaussian mixture: Moving cells detection",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        if np.any(var_original_video > (1 / 255)):
            var_original_video[var_original_video < (1 / 255)] = 1 / (255 * 3)

        if np.any(var_med_filtered_original_video > (1 / 255)):
            var_med_filtered_original_video[var_med_filtered_original_video < (1 / 255)] = (
                1 / (255 * 3)
            )

        max_original_video_threshold = video_float[j, :, :] > (
            mean_original_video + threshold * np.sqrt(var_original_video)
        )
        min_original_video_threshold = video_float[j, :, :] < (
            mean_original_video - threshold * np.sqrt(var_original_video)
        )
        detections_original_video = (
            max_original_video_threshold + min_original_video_threshold
        ).astype(int)

        max_med_filtered_original_video_threshold = med_filtered_original_video[j, :, :] > (
            mean_med_filtered_original_video + threshold * np.sqrt(var_med_filtered_original_video)
        )
        min_med_filtered_original_video_threshold = med_filtered_original_video[j, :, :] < (
            mean_med_filtered_original_video - threshold * np.sqrt(var_med_filtered_original_video)
        )
        detections_med_filtered_original_video = (
            max_med_filtered_original_video_threshold + min_med_filtered_original_video_threshold
        ).astype(int)

        detected = detections_original_video & detections_med_filtered_original_video
        detected = detected.astype(np.uint8) * 255

        seg_out = dilation(detected, _matlab_disk(disk_size))
        seg_out = erosion(seg_out, _matlab_disk(disk_size))
        morph = clear_border(seg_out)
        binarized_moving_cells_video[j] = morph

        if train_frames > 1:
            mean_original_video, var_original_video = _update_mean_variance(
                mean_original_video,
                var_original_video,
                video_float[j - train_frames, :, :],
                video_float[j, :, :],
                train_frames,
            )
            (
                mean_med_filtered_original_video,
                var_med_filtered_original_video,
            ) = _update_mean_variance(
                mean_med_filtered_original_video,
                var_med_filtered_original_video,
                med_filtered_original_video[j - train_frames, :, :],
                med_filtered_original_video[j, :, :],
                train_frames,
            )

    return binarized_moving_cells_video, med_filtered_original_video


def _find_blobs(
    original_video: np.ndarray,
    width: int,
    height: int,
    initial_frame: int,
    final_frame: int,
    detection_label_name: str,
    nbr_skipped_frames: int = 0,
    blob_min_pixel_area: int = 20,
    *,
    show_progress: bool = True,
) -> tuple[dict[str, list[list[str]]], np.ndarray]:
    """Port of legacy blob extraction and normalization logic."""
    scipy_ndimage = _ensure_import("scipy.ndimage", pip_name="scipy")
    label = scipy_ndimage.label
    skimage_measure = _ensure_import("skimage.measure", pip_name="scikit-image")
    regionprops = skimage_measure.regionprops

    blob_detections: dict[str, list[list[str]]] = {}
    modified_original_video = np.copy(original_video)

    start_frame = int(initial_frame + max(0, nbr_skipped_frames))
    total_frames = max(0, final_frame - start_frame + 1)
    for curr_frame in _progress_bar(
        range(start_frame, final_frame + 1),
        total=total_frames,
        desc="Detection moving-cells blobs",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        video_index = curr_frame - initial_frame
        if video_index < 0 or video_index >= len(original_video):
            continue

        structure = np.ones((3, 3), dtype=int)
        label_matrix, _ = label(original_video[video_index], structure=structure)
        regions = regionprops(label_matrix)

        current_frame_detections: list[list[str]] = []
        for region in regions:
            if region.area < blob_min_pixel_area:
                modified_original_video[video_index][
                    region.coords[:, 0], region.coords[:, 1]
                ] = 0
            else:
                centroid_y, centroid_x = region.centroid
                min_row, min_col, max_row, max_col = region.bbox
                bbox_width = max_col - min_col
                bbox_height = max_row - min_row

                norm_centroid_x = str(round((centroid_x + 1) / width, 6))
                norm_centroid_y = str(round((centroid_y + 1) / height, 6))
                norm_bbox_width = str(round(bbox_width / width, 6))
                norm_bbox_height = str(round(bbox_height / height, 6))

                current_frame_detections.append(
                    [
                        detection_label_name,
                        norm_centroid_x,
                        norm_centroid_y,
                        norm_bbox_width,
                        norm_bbox_height,
                    ]
                )

        blob_detections[str(curr_frame)] = current_frame_detections

    return blob_detections, modified_original_video


def _create_bg_subtractor(method: str) -> Any:
    """Create OpenCV background subtractor matching legacy method names."""
    cv2 = _ensure_import("cv2", pip_name="opencv-python")

    if method == "cv-gmg":
        bgsegm = getattr(cv2, "bgsegm", None)
        if bgsegm is None or not hasattr(bgsegm, "createBackgroundSubtractorGMG"):
            raise ImportError(
                "method='cv-gmg' requires cv2.bgsegm. Install `opencv-contrib-python`."
            )
        return bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=20, decisionThreshold=0.7
        )

    if method == "cv-mog":
        bgsegm = getattr(cv2, "bgsegm", None)
        if bgsegm is None or not hasattr(bgsegm, "createBackgroundSubtractorMOG"):
            raise ImportError(
                "method='cv-mog' requires cv2.bgsegm. Install `opencv-contrib-python`."
            )
        return bgsegm.createBackgroundSubtractorMOG()

    if method == "cv-mog2":
        return cv2.createBackgroundSubtractorMOG2()

    raise ValueError(f"method {method} is invalid.")


def detect_moving_cells(
    casa: dict[str, Any],
    method: str = "cv-gmg",
    *,
    show_progress: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Legacy-parity moving-cells detection.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        method (str, optional):
            Moving-cells extraction backend identifier:
            ``cv-gmg``, ``cv-mog``, ``cv-mog2``, or ``gm``.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while
            processing frames.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            detector. If ``False``, suppress those summaries. Warnings are not
            affected by this flag.
        **kwargs (Any):
            Optional controls. Legacy defaults are preserved:
            - ``number_training_frames`` (default ``20``)
            - ``blob_min_pixel_area`` (default ``20``)
            For method ``gm``:
            - ``med_filter_size`` (default ``3``)
            - ``disk_size`` (default ``6``)
            - ``threshold`` (default ``3``)

    Returns:
        dict[str, Any]:
            Updated session with:
            - ``casa['video']['binarized_moving_cells_video']``
            - ``casa['detections']['moving_cells']``
            - ``casa['meta']['last_detection']``

    Notes:
        Detection rows follow legacy normalized format:
        ``[label, norm_centroid_x, norm_centroid_y, norm_bbox_w, norm_bbox_h]``.
    """
    casa = _ensure_casa(casa)
    detections_root = casa.setdefault("detections", {})
    existing_predicted_methods = _predicted_detection_keys(detections_root)
    if existing_predicted_methods:
        _warn_yellow(
            "Previous detection result overwritten "
            f"({', '.join(existing_predicted_methods)} -> moving_cells)."
        )
    _clear_predicted_detections(detections_root)
    if verbose:
        print(f"Running moving-cells detection on frames (method={method})...")

    if casa.get("video", {}).get("original_video") is None:
        if verbose:
            print("Skipping moving-cells detection: no original video is loaded.")
        detections_root["moving_cells"] = {}
        casa["meta"]["last_detection"] = {
            "backend": "moving_cells",
            "method": method,
            "kwargs": kwargs,
            "skipped": True,
            "reason": "missing_original_video",
        }
        return casa
    original_video = _ensure_original_video(casa)

    num_frames, height, width = _ensure_video_dimensions(original_video)
    initial_frame = int(casa.get("video", {}).get("initial_frame", 0))
    final_frame = int(casa.get("video", {}).get("final_frame", initial_frame + num_frames - 1))
    if final_frame < initial_frame:
        final_frame = initial_frame + num_frames - 1

    number_training_frames = int(
        kwargs.get(
            "number_training_frames",
            casa.get("meta", {}).get("number_training_frames", 20),
        )
    )
    blob_min_pixel_area = int(
        kwargs.get(
            "blob_min_pixel_area",
            casa.get("meta", {}).get("blob_min_pixel_area", 20),
        )
    )

    method_key = method.lower()
    if method_key in ("cv-gmg", "cv-mog", "cv-mog2"):
        bg_subtractor = _create_bg_subtractor(method_key)
        moving_cells_video_list = []
        frame_count = int(len(original_video))
        for frame in _progress_bar(
            range(frame_count),
            total=frame_count,
            desc=f"Detection moving-cells ({method_key})",
            unit="frame",
            leave=True,
            enabled=show_progress,
        ):
            moving_cells_video_list.append(bg_subtractor.apply(original_video[frame]))
        binarized_moving_cells_video = np.asarray(moving_cells_video_list)
    elif method_key == "gm":
        binarized_moving_cells_video, med_filtered_original_video = _gaussian_mixture_motion_filter(
            original_video,
            num_train=number_training_frames,
            threshold=float(kwargs.get("threshold", 3)),
            med_filter_size=int(kwargs.get("med_filter_size", 3)),
            disk_size=int(kwargs.get("disk_size", 6)),
            show_progress=show_progress,
        )
        casa["video"]["med_filtered_grayscale_video"] = med_filtered_original_video
    else:
        raise ValueError(f"method {method} is invalid.")

    detections, binarized_moving_cells_video = _find_blobs(
        original_video=binarized_moving_cells_video,
        width=width,
        height=height,
        initial_frame=initial_frame,
        final_frame=final_frame,
        detection_label_name="moving_cells",
        nbr_skipped_frames=number_training_frames,
        blob_min_pixel_area=blob_min_pixel_area,
        show_progress=show_progress,
    )

    casa["video"]["binarized_moving_cells_video"] = binarized_moving_cells_video
    detections_root["moving_cells"] = detections
    tracked_sources = _resolve_sort_track_sources(casa.get("tracks", {}))
    if tracked_sources and "moving_cells" not in tracked_sources:
        _warn_yellow(
            "Detections were updated to 'moving_cells'. "
            "Re-run tracking to generate detection tracks."
        )
    casa["meta"]["last_detection"] = {
        "backend": "moving_cells",
        "method": method,
        "number_training_frames": number_training_frames,
        "blob_min_pixel_area": blob_min_pixel_area,
        "kwargs": kwargs,
        "skipped": False,
    }
    if verbose:
        total_detections = sum(
            len(rows)
            for rows in detections.values()
            if isinstance(rows, list)
        )
        frames_with_detections = sum(
            1
            for rows in detections.values()
            if isinstance(rows, list) and len(rows) > 0
        )
        settings_summary = (
            f"method={method}, "
            f"training_frames={number_training_frames}, "
            f"blob_min_pixel_area={blob_min_pixel_area}"
        )
        print(
            "Moving-cells summary: "
            f"frames_processed={num_frames}, "
            f"frames_with_detections={frames_with_detections}, "
            f"detections_found={total_detections}, "
            f"{settings_summary}"
        )
    return casa
