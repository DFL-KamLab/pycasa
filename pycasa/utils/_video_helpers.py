import numpy as np
from typing import Any


def _ensure_video_dimensions(original_video: np.ndarray) -> tuple[int, int, int]:
    """Return ``(frames, height, width)`` for 3D or 4D videos."""
    if original_video.ndim == 3:
        return (
            int(original_video.shape[0]),
            int(original_video.shape[1]),
            int(original_video.shape[2]),
        )
    if original_video.ndim == 4:
        return (
            int(original_video.shape[0]),
            int(original_video.shape[1]),
            int(original_video.shape[2]),
        )
    raise ValueError(f"Unsupported video shape: {original_video.shape}")


def _ensure_original_video(casa: dict[str, Any]) -> np.ndarray:
    """Return validated ``casa["video"]["original_video"]`` for preprocessing use."""
    original_video = casa.get("video", {}).get("original_video")
    if original_video is None:
        raise ValueError("`casa['video']['original_video']` is required before preprocessing.")
    if not isinstance(original_video, np.ndarray):
        raise TypeError("`casa['video']['original_video']` must be a numpy array.")
    if original_video.ndim not in (3, 4):
        raise ValueError(
            "`casa['video']['original_video']` must be a 3D or 4D numpy array "
            "(frames, height, width) or (frames, height, width, channels)."
        )
    return original_video


def _convert_video_to_grayscale(original_video: np.ndarray) -> np.ndarray:
    """Convert a 3D/4D frame stack to grayscale ``uint8``."""
    if original_video.ndim == 3:
        return original_video.astype(np.uint8, copy=True)

    if original_video.ndim != 4:
        raise ValueError(f"Unsupported video shape: {original_video.shape}")

    channels = int(original_video.shape[-1])
    if channels == 1:
        return original_video[..., 0].astype(np.uint8, copy=True)
    if channels < 3:
        raise ValueError(f"Unsupported channel count: {channels}")

    # Input convention is BGR to match load_video/OpenCV path.
    b = original_video[..., 0].astype(np.float32)
    g = original_video[..., 1].astype(np.float32)
    r = original_video[..., 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def _ensure_bgr(frame: np.ndarray, cv2: Any) -> np.ndarray:
    """Convert an arbitrary OpenCV frame to 3-channel BGR format.

    Parameters:
        frame (np.ndarray):
            Input frame returned by OpenCV capture.
        cv2 (Any):
            Imported ``cv2`` module used for color conversions.

    Returns:
        np.ndarray:
            Frame with shape ``(height, width, 3)`` in BGR order.

    Raises:
        ValueError:
            If frame shape/channels are unsupported.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported frame shape: {frame.shape}")
    
    
    
    
    
    
