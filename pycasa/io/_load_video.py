from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .._core._casa import _new_casa
from ..utils import _progress_bar
from ..utils import _ensure_import
from ..utils import _ensure_bgr
from ..utils import set_um_per_px

if TYPE_CHECKING:
    from ..casa import Casa

SUPPORTED_MEDIA_TYPES = (".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv")



def _load_groundtruth_detections(
    groundtruth_path: str,
    file_header: str = "frame-",
    zero_based_index: bool = True,
) -> dict[str, list[list[str]]]:
    """Load frame-indexed detections from per-frame text files.

    Parameters:
        groundtruth_path (str):
            Directory containing text files with one detection per line.
        file_header (str, optional):
            File-name token used to parse frame index from file stem.
            Default expects names like ``frame-170.txt``.
        zero_based_index (bool, optional):
            Whether parsed frame indices are already zero-based. If ``False``,
            each parsed index is shifted by ``-1``.

    Returns:
        dict[str, list[list[str]]]:
            Mapping of frame index (as string) to parsed rows. Each row is a
            whitespace-split list of tokens preserved as strings.

    Notes:
        - Non-matching files are ignored.
        - If ``groundtruth_path`` does not exist, an empty mapping is returned.
        - This parser is intentionally permissive for legacy label formats.
    """
    locations: dict[str, list[list[str]]] = {}
    directory = Path(groundtruth_path)
    if not directory.is_dir():
        return locations

    for file_path in sorted(directory.iterdir(), key=lambda p: p.name):
        if not file_path.is_file() or file_header not in file_path.name:
            continue

        frame_token = file_path.stem.split(file_header)[-1]
        try:
            frame_index = int(frame_token)
        except ValueError:
            continue

        if not zero_based_index:
            frame_index -= 1

        rows: list[list[str]] = []
        with file_path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(stripped.split())

        locations[str(frame_index)] = rows

    return locations
    
    
def _resolve_frame_range(
    total_number_frame: int,
    initial_frame: int,
    final_frame: int | None,
) -> tuple[int, int]:
    """Resolve frame boundaries to an inclusive valid range.

    Parameters:
        total_number_frame (int):
            Total number of frames in the source video.
        initial_frame (int):
            First frame index requested by the caller (0-based).
        final_frame (int | None):
            Last frame index requested by the caller (0-based, inclusive).
            When ``None``, the last available frame is used.

    Returns:
        tuple[int, int]:
            A validated ``(initial_frame, final_frame)`` pair.

    Raises:
        ValueError:
            If the requested range is invalid for the available frame count.
    """
    if total_number_frame <= 0:
        raise ValueError("`total_number_frame` must be > 0.")
    if initial_frame < 0:
        raise ValueError("`initial_frame` must be >= 0.")
    if initial_frame >= total_number_frame:
        raise ValueError(
            f"`initial_frame` ({initial_frame}) is out of range for "
            f"{total_number_frame} total frames."
        )

    if final_frame is None:
        final_frame = total_number_frame - 1
    elif final_frame >= total_number_frame:
        final_frame = total_number_frame - 1

    if final_frame < initial_frame:
        raise ValueError(
            f"`final_frame` ({final_frame}) must be >= `initial_frame` ({initial_frame})."
        )

    return initial_frame, final_frame


def load_video(
    video_path: str,
    groundtruth_path: str | None = None,
    initial_frame: int = 0,
    final_frame: int | None = None,
    sampling_rate: float | None = None,
    show_progress: bool = True,
    verbose: bool = True,
    um_per_px: float | None = None,
    magnification: str | None = None,
) -> "Casa":
    """Load a time-lapse video and return a fluent ``Casa`` object.

    Parameters:
        video_path (str):
            Path to video file. Supported extensions are listed in
            ``SUPPORTED_MEDIA_TYPES``.
        groundtruth_path (str | None, optional):
            Optional directory containing frame-level groundtruth text files.
            If provided, detections are loaded under
            ``casa["detections"]["groundtruth"]``.
        initial_frame (int, optional):
            First frame index to read (0-based). Defaults to ``0``.
        final_frame (int | None, optional):
            Last frame index to read (0-based, inclusive). ``None`` means
            "to the end of the video".
        sampling_rate (float | None, optional):
            Explicit frame-rate override. If ``None``, FPS is read from video
            metadata when available.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa orange `#` progress bar while
            reading frames when ``tqdm`` is available.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for this
            loading step. If ``False``, suppress those summaries. Warnings are
            not affected by this flag.
        um_per_px (float | None, optional):
            Microns-per-pixel calibration stored in metadata.
        magnification (str | None, optional):
            Free-text magnification descriptor stored in metadata.

    Returns:
        Casa:
            A fluent ``Casa`` object with ``meta`` and ``video`` populated,
            plus optional ``detections["groundtruth"]`` if groundtruth path is used.

    Raises:
        FileNotFoundError:
            If ``video_path`` does not exist.
        ValueError:
            If media extension or frame range is invalid.
        RuntimeError:
            If OpenCV cannot open/read any frames from the file.
        ImportError:
            If ``opencv-python`` is not installed.

    Notes:
        - Loaded video frames are stored in BGR order for OpenCV parity.
        - ``final_frame`` is clamped to the last available frame.
        - ``number_frame_used`` reflects actual readable frames, which may be
          smaller than requested if early read termination occurs.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_video("sample.avi", initial_frame=0, final_frame=100)
        >>> isinstance(session, pc.Casa)
        True
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path_obj}")
    if video_path_obj.suffix.lower() not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f"Unsupported media type: {video_path_obj.suffix}. "
            f"Supported: {', '.join(SUPPORTED_MEDIA_TYPES)}"
        )

    cv2 = _ensure_import("cv2", pip_name="opencv-python")

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path_obj}")
    if verbose:
        print(f"Loading video frames from '{video_path_obj.name}'...")

    try:
        detected_sampling_rate = float(cap.get(cv2.CAP_PROP_FPS))
        total_number_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        initial_frame, final_frame = _resolve_frame_range(
            total_number_frame=total_number_frame,
            initial_frame=initial_frame,
            final_frame=final_frame,
        )
        number_frame_used = final_frame - initial_frame + 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
        original_video = np.zeros((number_frame_used, height, width, 3), dtype=np.uint8)

        read_frames = 0
        for frame_idx in _progress_bar(
            range(number_frame_used),
            total=number_frame_used,
            desc="Loading video",
            unit="frame",
            leave=True,
            enabled=show_progress,
        ):
            ret, frame = cap.read()
            if not ret:
                break

            original_video[frame_idx] = _ensure_bgr(frame, cv2)
            read_frames += 1
    finally:
        cap.release()

    if read_frames == 0:
        raise RuntimeError(f"No frames could be read from: {video_path_obj}")

    if read_frames < number_frame_used:
        original_video = original_video[:read_frames]
        final_frame = initial_frame + read_frames - 1
        number_frame_used = read_frames

    casa: dict[str, Any] = _new_casa()
    resolved_sampling_rate: float | None
    if sampling_rate is None:
        resolved_sampling_rate = (
            detected_sampling_rate if detected_sampling_rate > 0 else None
        )
    else:
        resolved_sampling_rate = float(sampling_rate)

    total_duration_sec: float | None
    duration_sec: float | None
    if resolved_sampling_rate and resolved_sampling_rate > 0:
        total_duration_sec = total_number_frame / resolved_sampling_rate
        duration_sec = number_frame_used / resolved_sampling_rate
    else:
        total_duration_sec = None
        duration_sec = None

    if um_per_px is None:
        import warnings

        def _warning_format(message, category, filename, lineno, line=None):
            """Format um_per_px warnings in a highlighted single-line style."""
            return f"\033[93mWarning: {message}\033[0m\n"

        warnings.formatwarning = _warning_format
        warnings.warn(
            "Motility parameter will not compute if um_per_px is None. \n"
            "To properly compute motility parameters, set um_per_px to a positive value. \n"
            "You should either reinitialize your object or use casa.set_um_per_px(value). \n",
            UserWarning,
        )

    casa["meta"].update(
        {
            "video_path": str(video_path_obj),
            "sampling_rate": resolved_sampling_rate,
            "magnification": magnification,
            "width": width,
            "height": height,
            "total_number_frame": total_number_frame,
            "total_duration_sec": total_duration_sec,
            "duration_sec": duration_sec,
        }
    )
    if um_per_px is not None:
        casa = set_um_per_px(casa, um_per_px)
    else:
        casa["meta"]["um_per_px"] = None
    casa["video"].update(
        {
            "path": str(video_path_obj),
            "initial_frame": initial_frame,
            "final_frame": final_frame,
            "number_frame_used": number_frame_used,
            "original_video": original_video,
        }
    )
    if groundtruth_path is not None:
        casa["detections"]["groundtruth"] = _load_groundtruth_detections(
            groundtruth_path
        )
        casa["detections"]["groundtruth_path"] = str(groundtruth_path)
    if verbose:
        fps_text = (
            f"{resolved_sampling_rate:.2f}"
            if isinstance(resolved_sampling_rate, float)
            else "None"
        )
        print(
            "Loaded video summary: "
            f"frames={number_frame_used}/{total_number_frame}, "
            f"range={initial_frame}-{final_frame}, "
            f"size={width}x{height}, "
            f"fps={fps_text}"
        )
        if groundtruth_path is not None:
            groundtruth = casa.get("detections", {}).get("groundtruth", {})
            frames_with_labels = 0
            total_labels = 0
            if isinstance(groundtruth, dict):
                frames_with_labels = len(groundtruth)
                total_labels = sum(
                    len(rows)
                    for rows in groundtruth.values()
                    if isinstance(rows, list)
                )
            print(
                "Groundtruth summary: "
                f"frames_with_labels={frames_with_labels}, "
                f"labels={total_labels}"
            )

    from ..casa import Casa

    return Casa(casa)
