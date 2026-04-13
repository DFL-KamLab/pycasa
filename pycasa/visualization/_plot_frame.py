from typing import Any, Iterable

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _import_matplotlib_for_visualization
from ..utils import _parse_detection_entries
from ..utils import _parse_image_types
from ..utils import _prepare_frame_for_display
from ..utils import _resolve_active_predicted_detection_method
from ..utils import _resolve_frame_entries

_IMAGE_KEY_TO_VIDEO_KEY = {
    "original": "original_video",
    "grayscale": "grayscale_video",
    "normalized": "normalized_video",
    "binarized": "binary_video",
    "moving_cells": "binarized_moving_cells_video",
}

_MISSING_IMAGE_ERROR = {
    "original": "No original video found. Load a video first.",
    "grayscale": "No grayscale video found. Run `self.preprocessing.grayscale()` first.",
    "normalized": (
        "No normalized video found. Run `self.preprocessing.normalization.min_max()` first."
    ),
    "binarized": (
        "No binarized video found. Run `self.preprocessing.binarization.otsu()` first."
    ),
    "moving_cells": (
        "No moving_cells video found. Run `self.detection.detect_moving_cells()` first."
    ),
}


def _ensure_image_video(casa: dict[str, Any], image_key: str) -> np.ndarray:
    """Return the requested image video from ``casa['video']`` or raise ``ValueError``."""
    video = casa.get("video", {})
    if not isinstance(video, dict):
        raise ValueError(_MISSING_IMAGE_ERROR[image_key])

    video_key = _IMAGE_KEY_TO_VIDEO_KEY[image_key]
    video_data = video.get(video_key)
    if not isinstance(video_data, np.ndarray):
        raise ValueError(_MISSING_IMAGE_ERROR[image_key])
    return video_data


def _resolve_frame_index(frame_count: int, frame_index: int | None) -> int:
    """Resolve and validate frame index; default to the middle frame when unset."""
    if frame_count <= 0:
        raise ValueError("No frames are available for visualization.")

    if frame_index is None:
        return frame_count // 2

    try:
        resolved = int(frame_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("`frame_index` must be an integer.") from exc

    if resolved < 0 or resolved >= frame_count:
        raise ValueError(
            f"`frame_index`={resolved} is out of range for {frame_count} frame(s)."
        )
    return resolved


def plot_frame(
    casa: dict[str, Any],
    image_type: str | Iterable[str] = "original",
    frame_index: int | None = None,
    show_detections: bool = True,
) -> dict[str, Any]:
    """Plot one frame across one or more image representations.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary with videos and optional detections.
        image_type (str | Iterable[str], optional):
            One or more image types (single value, ``+``/``,`` separated string,
            or iterable) among ``original``, ``grayscale``, ``normalized``,
            ``binarized``, and ``moving_cells``.
        frame_index (int | None, optional):
            Zero-based local index to display. If ``None``, the middle frame is
            selected from available videos.
        show_detections (bool, optional):
            Whether to overlay detections on the plotted frame.

    Returns:
        dict[str, Any]:
            The same ``casa`` dictionary with visualization metadata in
            ``casa['meta']['last_visualization']``.

    Raises:
        ValueError:
            If a requested image video is missing or frame selection is invalid.
        ImportError:
            If ``matplotlib`` is unavailable.

    Notes:
        This function is intentionally single-frame only. For interactive
        timeline browsing, use ``timelapse``.
    """
    casa = _ensure_casa(casa)
    image_keys = _parse_image_types(image_type)
    videos = [_ensure_image_video(casa, key) for key in image_keys]

    frame_counts = [int(video_data.shape[0]) for video_data in videos]
    frame_count = min(frame_counts) if frame_counts else 0
    local_frame = _resolve_frame_index(frame_count, frame_index)

    initial_frame = int(casa.get("video", {}).get("initial_frame", 0) or 0)
    global_frame = initial_frame + local_frame

    resolved_detection_method: str | None = None
    selected_detections: dict[str, list[Any]] = {}
    if show_detections:
        detections_root = casa.get("detections", {})
        if not isinstance(detections_root, dict):
            detections_root = {}
        resolved_detection_method = (
            _resolve_active_predicted_detection_method(detections_root)
            or "groundtruth"
        )
        raw = (
            detections_root.get(resolved_detection_method, {})
        )
        if isinstance(raw, dict):
            selected_detections = raw
        else:
            raw = {}
        if not selected_detections:
            print(
                f"Warning: no detections were found for '{resolved_detection_method}'. "
                "Plotting without detections."
            )

    plt = _import_matplotlib_for_visualization("plot-frame")
    _, axes = plt.subplots(1, len(videos), figsize=(5.5 * len(videos), 5.5))
    if len(videos) == 1:
        axes = [axes]

    for axis, image_key, video_data in zip(axes, image_keys, videos):
        frame = video_data[local_frame]
        shown_frame, is_grayscale = _prepare_frame_for_display(frame)
        if is_grayscale:
            axis.imshow(shown_frame, cmap="gray")
        else:
            axis.imshow(shown_frame)
        axis.set_title(f"{image_key} | frame {global_frame}")
        axis.axis("off")

        if show_detections and selected_detections:
            raw_entries = _resolve_frame_entries(
                selected_detections,
                global_frame=global_frame,
                local_index=local_frame,
            )
            if raw_entries:
                height = int(shown_frame.shape[0])
                width = int(shown_frame.shape[1])
                points = _parse_detection_entries(raw_entries, width=width, height=height)
                if points:
                    offsets = np.asarray(points, dtype=np.float32)
                    axis.scatter(
                        offsets[:, 0],
                        offsets[:, 1],
                        s=18,
                        c="red",
                        marker="x",
                        alpha=0.9,
                    )

    plt.tight_layout()
    plt.show()

    casa["meta"]["last_visualization"] = {
        "type": "plot_frame",
        "image_type": "+".join(image_keys),
        "local_frame_index": int(local_frame),
        "global_frame_index": int(global_frame),
        "detection_method": resolved_detection_method,
        "show_detections": bool(show_detections),
    }
    return casa
