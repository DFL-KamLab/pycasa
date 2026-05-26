from collections.abc import Iterable
import numpy as np
from typing import Any

from ._dependency_helpers import _ensure_import

_IMAGE_ALIASES = {
    "original": ("original_video",),
    "grayscale": ("grayscale_video",),
    "gray": ("grayscale_video",),
    "normalized": ("normalized_video",),
    "binarized": ("binary_video",),
    "binary": ("binary_video",),
    "moving_cells": ("binarized_moving_cells_video",),
    "digital_washing": ("digital_washing_washed_video",),
}


def _is_noninteractive_backend(backend_name: str) -> bool:
    """Return ``True`` when backend name is non-interactive."""
    normalized = backend_name.strip().lower()
    return normalized in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"} or (
        "inline" in normalized
    )


def _import_matplotlib_for_visualization(kind: str):
    """Import matplotlib pyplot and prefer interactive backends."""
    matplotlib = _ensure_import("matplotlib", pip_name="matplotlib")
    plt = _ensure_import("matplotlib.pyplot", pip_name="matplotlib")

    backend = str(matplotlib.get_backend())
    _ = kind
    if _is_noninteractive_backend(backend):
        for candidate in ("TkAgg", "QtAgg"):
            try:
                plt.switch_backend(candidate)
                # Probe backend usability immediately; some environments accept
                # backend switch but fail on first figure creation.
                probe_figure = plt.figure()
                plt.close(probe_figure)
                backend = str(matplotlib.get_backend())
                break
            except Exception:
                continue

    if _is_noninteractive_backend(backend):
        print(
            f"Warning: matplotlib backend '{matplotlib.get_backend()}' is non-interactive. "
            "No plot window may appear. Set `MPLBACKEND=tkagg` before running, "
            "or switch backend in code."
        )
    return plt


def _parse_image_types(image_type: str | Iterable[str]) -> list[str]:
    """Parse and normalize requested image layers for viewer display."""
    if isinstance(image_type, str):
        raw = image_type.replace(",", "+").split("+")
        tokens = [token.strip().lower() for token in raw if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in image_type if str(token).strip()]

    if not tokens:
        raise ValueError("`image_type` must contain at least one image key.")

    normalized: list[str] = []
    for token in tokens:
        if token not in _IMAGE_ALIASES:
            allowed = ", ".join(sorted(_IMAGE_ALIASES))
            raise ValueError(f"Unknown image type `{token}`. Allowed: {allowed}")
        canonical = "grayscale" if token == "gray" else token
        canonical = "binarized" if token == "binary" else canonical
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _prepare_frame_for_display(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert frame to a matplotlib-ready array and return grayscale flag."""
    if frame.ndim == 2:
        return frame, True
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return frame[..., 0], True
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[..., :3][..., ::-1], False
    raise ValueError(f"Unsupported frame shape for display: {frame.shape}")


def _resolve_frame_entries(
    frame_dict: dict[str, list[Any]],
    global_frame: int,
    local_index: int,
) -> list[Any]:
    """Resolve frame data by global index first, then local index fallback."""
    for key in (str(global_frame), str(local_index)):
        value = frame_dict.get(key)
        if isinstance(value, list):
            return value
    return []


def _to_pixel_if_normalized(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Convert normalized ``[0, 1]`` coordinates to pixels."""
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def _parse_detection_entries(
    detections: list[Any],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Parse mixed-format detection entries into ``(x, y)`` pixel points."""
    points: list[tuple[float, float]] = []
    for det in detections:
        if isinstance(det, dict):
            x_raw = det.get("x", det.get("cx"))
            y_raw = det.get("y", det.get("cy"))
            if x_raw is None or y_raw is None:
                continue
            try:
                x_val = float(x_raw)
                y_val = float(y_raw)
            except (TypeError, ValueError):
                continue
            points.append(_to_pixel_if_normalized(x_val, y_val, width, height))
            continue

        if isinstance(det, (list, tuple)):
            if len(det) >= 3:
                x_idx = 1
                y_idx = 2
            elif len(det) >= 2:
                x_idx = 0
                y_idx = 1
            else:
                continue

            try:
                x_val = float(det[x_idx])
                y_val = float(det[y_idx])
            except (TypeError, ValueError):
                continue

            points.append(_to_pixel_if_normalized(x_val, y_val, width, height))
    return points


def _resolve_visualization_source(casa: dict[str, Any]) -> str:
    """Resolve active detection-source label for tracking/motility visuals."""
    meta = casa.get("meta", {})
    if not isinstance(meta, dict):
        return "groundtruth"

    last_motility = meta.get("last_motility")
    if isinstance(last_motility, dict):
        source = last_motility.get("detection_method")
        if isinstance(source, str) and source:
            return source
        processed = last_motility.get("sources_processed")
        if isinstance(processed, list):
            for item in processed:
                if isinstance(item, str) and item:
                    return item

    last_tracking = meta.get("last_tracking")
    if isinstance(last_tracking, dict):
        source = last_tracking.get("detection_method")
        if isinstance(source, str) and source:
            return source
        processed = last_tracking.get("sources_processed")
        if isinstance(processed, list):
            for item in processed:
                if isinstance(item, str) and item:
                    return item

    return "groundtruth"
