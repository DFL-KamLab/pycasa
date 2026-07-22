import re
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .._core._casa import _new_casa
from ..utils import _progress_bar
from ..utils import _ensure_import
from ..utils import _ensure_bgr
from ..utils import set_um_per_px
from ..utils import set_volume_ml
from ..utils import set_chamber_depth_um
from ..utils import _GROUNDTRUTH_TRACKS_KEY

if TYPE_CHECKING:
    from ..casa import Casa

SUPPORTED_MEDIA_TYPES = (".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv")



def _frame_index_from_filename(name: str, positional_fallback: int) -> int:
    """Derive a global frame index from a label file name.

    Uses the last integer group found in ``name`` (e.g. ``82_frame_1000`` ->
    ``1000``, ``..._gt_frame-0`` -> ``0``, ``60_frame_5_with_ftid`` -> ``5``).
    When the name contains no digits, ``positional_fallback`` is returned so
    files can still be ordered by their position in the sorted listing.

    Parameters:
        name (str):
            File name (with or without extension).
        positional_fallback (int):
            Index to use when ``name`` has no parseable integer group.

    Returns:
        int:
            The resolved global frame index.
    """
    matches = re.findall(r"\d+", name)
    if matches:
        return int(matches[-1])
    return positional_fallback


def _iter_label_files(directory_path: str) -> list[tuple[int, "Path"]]:
    """Return ``(frame_index, path)`` pairs for label files, ordered by frame.

    Collects every ``*.txt`` file in ``directory_path``, derives a frame index
    from each file name via :func:`_frame_index_from_filename`, and returns the
    list sorted by that frame index. Returns an empty list when the directory
    does not exist.
    """
    directory = Path(directory_path)
    if not directory.is_dir():
        return []

    txt_files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )
    indexed = [
        (_frame_index_from_filename(path.stem, positional), path)
        for positional, path in enumerate(txt_files)
    ]
    indexed.sort(key=lambda item: item[0])
    return indexed


def _load_groundtruth_detections(
    groundtruth_detections_path: str,
) -> dict[str, list[list[str]]]:
    """Load frame-indexed detections from per-frame text files.

    Parameters:
        groundtruth_detections_path (str):
            Directory containing ``*.txt`` files with one detection per line.

    Returns:
        dict[str, list[list[str]]]:
            Mapping of frame index (as string) to parsed rows. Each row is a
            whitespace-split list of tokens preserved as strings.

    Notes:
        - Every ``*.txt`` file is used; the frame index is taken from the last
          integer group in the file name (see :func:`_frame_index_from_filename`),
          so folders named ``frame-170.txt``, ``82_frame_170.txt``, etc. all work.
        - If ``groundtruth_detections_path`` does not exist, an empty mapping is
          returned.
        - This parser is intentionally permissive for legacy label formats.
    """
    locations: dict[str, list[list[str]]] = {}

    for frame_index, file_path in _iter_label_files(groundtruth_detections_path):
        rows: list[list[str]] = []
        with file_path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(stripped.split())

        locations[str(frame_index)] = rows

    return locations


def _load_groundtruth_tracks(
    groundtruth_tracks_path: str,
    width: int,
    height: int,
) -> dict[str, dict[str, list[float]]]:
    """Load ground-truth tracks from per-frame YOLO-with-id text files.

    Each label file holds one object per line in the form
    ``track_id class cx cy w h`` (coordinates normalized to ``[0, 1]``). The
    same ``track_id`` recurring across frames defines a persistent identity.
    Only the box center is retained (pixel space) so the result matches the
    shape produced by the tracking backends:
    ``{track_id: {frame: [center_x, center_y]}}``.

    Parameters:
        groundtruth_tracks_path (str):
            Directory containing ``*.txt`` files with one object per line,
            each prefixed by a persistent track id.
        width (int):
            Video frame width in pixels, used to de-normalize ``cx``.
        height (int):
            Video frame height in pixels, used to de-normalize ``cy``.

    Returns:
        dict[str, dict[str, list[float]]]:
            Mapping ``track_id -> {frame_str: [center_x, center_y]}`` with track
            ids normalized to ``t0``, ``t1``, ... in order of first appearance.
            Returns an empty mapping when the directory does not exist.

    Notes:
        - The frame index is taken from the file name (see
          :func:`_frame_index_from_filename`).
        - Rows shorter than ``track_id class cx cy`` (4 tokens) are skipped.
        - Coordinates already in pixel space (any value ``> 1``) are used as-is.
        - Full bounding boxes (``w``/``h``) and class are intentionally dropped
          for now; only centers are stored.
    """
    # Preserve first-appearance order of raw ids across increasing frames.
    raw_tracks: dict[str, dict[str, list[float]]] = {}
    first_seen: dict[str, tuple[int, int]] = {}

    for frame_index, file_path in _iter_label_files(groundtruth_tracks_path):
        frame_key = str(frame_index)
        with file_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle):
                tokens = line.strip().split()
                if len(tokens) < 4:
                    continue
                raw_id = tokens[0]
                try:
                    cx = float(tokens[2])
                    cy = float(tokens[3])
                except ValueError:
                    continue

                if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                    cx *= width
                    cy *= height

                if raw_id not in raw_tracks:
                    raw_tracks[raw_id] = {}
                    first_seen[raw_id] = (frame_index, line_number)
                raw_tracks[raw_id][frame_key] = [cx, cy]

    ordered_ids = sorted(raw_tracks.keys(), key=lambda rid: first_seen[rid])
    tracks: dict[str, dict[str, list[float]]] = {}
    for new_index, raw_id in enumerate(ordered_ids):
        tracks[f"t{new_index}"] = raw_tracks[raw_id]
    return tracks


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
    groundtruth_detections_path: str | None = None,
    groundtruth_tracks_path: str | None = None,
    initial_frame: int = 0,
    final_frame: int | None = None,
    sampling_rate: float | None = None,
    show_progress: bool = True,
    verbose: bool = True,
    um_per_px: float | None = None,
    volume_ml: float | None = None,
    chamber_depth_um: float | None = None,
    magnification: str | None = None,
) -> "Casa":
    """Load a time-lapse video and return a fluent ``Casa`` object.

    Parameters:
        video_path (str):
            Path to video file. Supported extensions are listed in
            ``SUPPORTED_MEDIA_TYPES``.
        groundtruth_detections_path (str | None, optional):
            Optional directory containing frame-level groundtruth detection
            text files (YOLO ``class cx cy w h`` rows). If provided, detections
            are loaded under ``casa["detections"]["groundtruth"]``.
        groundtruth_tracks_path (str | None, optional):
            Optional directory containing frame-level groundtruth track text
            files (YOLO rows prefixed with a persistent ``track_id``). If
            provided, imported truth tracks are loaded under
            ``casa["tracks"]["groundtruth_tracks"]`` as
            ``{track_id: {frame: [center_x, center_y]}}``. This is distinct from
            ``casa["tracks"][backend]["groundtruth"]``, which holds tracks a
            backend computed from groundtruth detections.
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
        volume_ml (float | None, optional):
            Ejaculate volume (mL) stored in metadata for CASA total-count
            reporting. Equivalent to calling ``self.set_volume_ml(...)`` later.
        chamber_depth_um (float | None, optional):
            Counting-chamber depth (um) stored in metadata for CASA
            concentration. Equivalent to ``self.set_chamber_depth_um(...)``.
        magnification (str | None, optional):
            Free-text magnification descriptor stored in metadata.

    Returns:
        Casa:
            A fluent ``Casa`` object with ``meta`` and ``video`` populated, plus
            optional ``detections["groundtruth"]`` and/or
            ``tracks["groundtruth_tracks"]`` if the corresponding groundtruth
            paths are provided.

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
    if volume_ml is not None:
        casa = set_volume_ml(casa, volume_ml)
    if chamber_depth_um is not None:
        casa = set_chamber_depth_um(casa, chamber_depth_um)
    casa["video"].update(
        {
            "path": str(video_path_obj),
            "initial_frame": initial_frame,
            "final_frame": final_frame,
            "number_frame_used": number_frame_used,
            "original_video": original_video,
        }
    )
    if groundtruth_detections_path is not None:
        casa["detections"]["groundtruth"] = _load_groundtruth_detections(
            groundtruth_detections_path
        )
        casa["detections"]["groundtruth_detections_path"] = str(
            groundtruth_detections_path
        )
    if groundtruth_tracks_path is not None:
        casa["tracks"][_GROUNDTRUTH_TRACKS_KEY] = _load_groundtruth_tracks(
            groundtruth_tracks_path, width, height
        )
        casa["meta"]["groundtruth_tracks_path"] = str(groundtruth_tracks_path)
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
        if groundtruth_detections_path is not None:
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
                "Groundtruth detections summary: "
                f"frames_with_labels={frames_with_labels}, "
                f"labels={total_labels}"
            )
        if groundtruth_tracks_path is not None:
            gt_tracks = casa.get("tracks", {}).get(_GROUNDTRUTH_TRACKS_KEY, {})
            track_count = len(gt_tracks) if isinstance(gt_tracks, dict) else 0
            frames_covered = 0
            if isinstance(gt_tracks, dict):
                frames_covered = len(
                    {
                        frame
                        for points in gt_tracks.values()
                        if isinstance(points, dict)
                        for frame in points
                    }
                )
            print(
                "Groundtruth tracks summary: "
                f"tracks={track_count}, "
                f"frames_covered={frames_covered}"
            )

    from ..casa import Casa

    return Casa(casa)
