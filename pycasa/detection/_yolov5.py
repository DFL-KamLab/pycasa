import os
import shutil
import sys
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _ensure_original_video
from ..utils import _ensure_video_dimensions
from ..utils import _clear_predicted_detections
from ..utils import _predicted_detection_keys
from ..utils import _progress_bar
from ..utils import _resolve_sort_track_sources
from ..utils import _warn_yellow

HF_REPO_ID = "DFL-KamLab/HSTLI_A-Dataset-of-Human-Semen-Time-Lapse-Images"
HF_REPO_TYPE = "dataset"
YOLO_WEIGHTS_DIRNAME = "yolov5-weights"

VALID_MODELS = ("yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x")
VALID_WEIGHT_SETS = ("sys-casa", "sys-opt")
VALID_MANAGED_WEIGHTS = tuple(
    f"{weight_set}_{model}.pt"
    for weight_set in VALID_WEIGHT_SETS
    for model in VALID_MODELS
)
DEFAULT_MANAGED_WEIGHTS = "sys-casa_yolov5s.pt"


def _resolve_env_path(env_var: str | None) -> Path | None:
    """Resolve project-root override from an environment variable when set."""
    if not env_var:
        return None
    raw = os.getenv(env_var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _find_project_root(start: Path, *, marker: str = "pyproject.toml") -> Path | None:
    """Return nearest ancestor containing ``marker`` starting from ``start``."""
    cursor = start.resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / marker).is_file():
            return candidate
    return None


def _project_root() -> Path:
    """Resolve project root from env override, marker lookup, then CWD fallback."""
    env_override = _resolve_env_path("PYCASA_PROJECT_ROOT")
    if env_override is not None:
        return env_override
    detected = _find_project_root(Path(__file__))
    if detected is not None:
        return detected
    return Path.cwd().resolve()


def _import_hf_hub_download() -> Callable[..., str]:
    huggingface_hub = _ensure_import("huggingface_hub", pip_name="huggingface_hub")
    return huggingface_hub.hf_hub_download


def _normalize_weights(weights: str) -> str:
    return str(weights).strip().lower()


def _ensure_managed_weights(weights: str) -> str:
    managed_weights = _normalize_weights(weights)
    if managed_weights not in VALID_MANAGED_WEIGHTS:
        raise ValueError(f"`weights` must be one of {VALID_MANAGED_WEIGHTS}.")
    return managed_weights


def _split_managed_weights(weights: str) -> tuple[str, str]:
    managed_weights = _ensure_managed_weights(weights)
    weight_set, model_with_ext = managed_weights.split("_", 1)
    model = model_with_ext[:-3]
    return model, weight_set


def _is_managed_weight_name(weights: str) -> bool:
    return _normalize_weights(weights) in VALID_MANAGED_WEIGHTS


def _build_weight_filename(model: str, weight_set: str) -> str:
    return f"{YOLO_WEIGHTS_DIRNAME}/{weight_set}_{model}.pt"


def _build_legacy_weight_filename(model: str, weight_set: str) -> str:
    return f"{YOLO_WEIGHTS_DIRNAME}/{weight_set}/{weight_set}_{model}.pt"


def _build_weight_filename_from_name(weights: str) -> str:
    model_name, weight_name = _split_managed_weights(weights)
    return _build_weight_filename(model_name, weight_name)


def _build_legacy_weight_filename_from_name(weights: str) -> str:
    model_name, weight_name = _split_managed_weights(weights)
    return _build_legacy_weight_filename(model_name, weight_name)


def _local_managed_weight_path(weights: str) -> Path:
    return _project_root() / _build_weight_filename_from_name(weights)


def _local_legacy_managed_weight_path(weights: str) -> Path:
    return _project_root() / _build_legacy_weight_filename_from_name(weights)


def _promote_legacy_local_weight_if_needed(weights: str) -> Path | None:
    local_path = _local_managed_weight_path(weights)
    if local_path.is_file():
        return local_path

    legacy_path = _local_legacy_managed_weight_path(weights)
    if not legacy_path.is_file():
        return None

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, local_path)
    except OSError:
        return None
    return local_path


def _missing_weight_message(weights: str) -> str:
    managed_weights = _ensure_managed_weights(weights)
    return (
        "The requested YOLO weight file is not cached locally. "
        "Re-run with `download=True` or pass a local custom weights path. "
        f"Missing managed weights: `{managed_weights}`."
    )


def _resolve_custom_weight_path(weights: str) -> str:
    candidate = Path(str(weights).strip()).expanduser()
    if not candidate.is_absolute():
        project_candidate = _project_root() / candidate
        if project_candidate.is_file():
            candidate = project_candidate

    if candidate.is_file():
        return str(candidate.resolve())

    raise FileNotFoundError(
        "Custom `weights` path was not found. "
        "Provide an existing local file path or one managed weight name from "
        f"{VALID_MANAGED_WEIGHTS}."
    )


def _resolve_yolo_weight_path_from_name(
    weights: str,
    *,
    allow_download: bool,
    force_download: bool = False,
) -> str:
    managed_weights = _ensure_managed_weights(weights)
    promoted = _promote_legacy_local_weight_if_needed(managed_weights)
    if promoted is not None and not force_download:
        return str(promoted)

    local_path = _local_managed_weight_path(managed_weights)
    if local_path.is_file() and not force_download:
        return str(local_path)
    if not allow_download:
        raise FileNotFoundError(_missing_weight_message(managed_weights))

    hf_hub_download = _import_hf_hub_download()
    remote_filenames = (
        _build_weight_filename_from_name(managed_weights),
        _build_legacy_weight_filename_from_name(managed_weights),
    )
    last_error: Exception | None = None
    for remote_filename in remote_filenames:
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    filename=remote_filename,
                    local_dir=str(_project_root()),
                    local_files_only=False,
                    force_download=bool(force_download),
                )
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if downloaded_path.resolve() != local_path.resolve():
                shutil.copy2(downloaded_path, local_path)
            return str(local_path)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to resolve managed YOLO weights.")


def _import_torch():
    return _ensure_import("torch")


def _load_torchscript_model(weight_path: str) -> Any:
    torch = _import_torch()
    model = torch.jit.load(weight_path, map_location="cpu")
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def _coerce_frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            "YOLO detection expects frame data with 2D grayscale or 3-channel image data."
        )

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame[..., :3])


def _find_local_yolov5_repo() -> str:
    candidates: list[Path] = []

    env_repo = os.getenv("PYCASA_YOLOV5_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    repo_root = _project_root()
    candidates.append(repo_root.parent / "yolov5")
    candidates.append(repo_root / "yolov5")

    for candidate in candidates:
        if (candidate / "hubconf.py").is_file():
            return str(candidate)

    raise FileNotFoundError(
        "These YOLO weights are standard YOLOv5 checkpoints, not TorchScript archives. "
        "Clone YOLOv5 with `git clone https://github.com/ultralytics/yolov5.git` "
        "next to the `pycasa_as` repo so it resolves as `../yolov5`, or set "
        "`PYCASA_YOLOV5_REPO` to an existing local clone."
    )


def _ensure_standard_checkpoint_dependencies() -> None:
    modules = ("ultralytics", "torchvision", "pandas", "requests", "yaml", "psutil", "PIL")
    for module_name in modules:
        _ensure_import(module_name)


@contextmanager
def _temporary_sys_path(path: str):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)


def _load_standard_yolov5_model(weight_path: str) -> Any:
    torch = _import_torch()
    _ensure_standard_checkpoint_dependencies()
    repo_dir = _find_local_yolov5_repo()

    with _temporary_sys_path(repo_dir):
        from models.common import AutoShape, DetectMultiBackend

        model = DetectMultiBackend(
            weight_path,
            device=torch.device("cpu"),
            fp16=False,
            fuse=False,
        )
        wrapped = AutoShape(model, verbose=False)
    return wrapped


def _prepare_frame_tensor(
    frame: np.ndarray,
    *,
    image_size: int = 640,
) -> tuple[Any, float, int, int, int, int]:
    torch = _import_torch()
    functional = torch.nn.functional

    frame = _coerce_frame_to_bgr(frame)
    rgb_frame = np.ascontiguousarray(frame[..., ::-1])
    orig_h, orig_w = int(rgb_frame.shape[0]), int(rgb_frame.shape[1])
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("YOLO detection requires non-empty frames.")

    scale = min(float(image_size) / float(orig_h), float(image_size) / float(orig_w))
    resized_h = max(1, int(round(orig_h * scale)))
    resized_w = max(1, int(round(orig_w * scale)))

    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = functional.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )

    pad_h = image_size - resized_h
    pad_w = image_size - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    tensor = functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=114.0 / 255.0,
    )
    return tensor, scale, pad_left, pad_top, orig_w, orig_h


def _normalize_model_output(output: Any) -> Any:
    torch = _import_torch()

    if isinstance(output, list | tuple):
        if not output:
            return torch.empty((0, 6), dtype=torch.float32)
        output = output[0]

    if not hasattr(output, "detach"):
        raise TypeError("YOLO model output must be tensor-like.")

    tensor = output.detach().cpu()
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("YOLO model output batch dimension must be 1 when 3D.")
        tensor = tensor[0]
    if tensor.ndim == 1:
        if tensor.numel() == 0:
            return tensor.reshape(0, 6)
        if tensor.numel() < 6:
            raise ValueError("YOLO model output rows must have at least 6 values.")
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError("YOLO model output must be a 2D tensor or batched 3D tensor.")
    if tensor.shape[1] < 6:
        raise ValueError("YOLO model output must have at least 6 columns per detection row.")
    return tensor[:, :6].to(dtype=torch.float32)


def _postprocess_detections(
    detections: Any,
    *,
    conf: float,
    scale: float,
    pad_left: int,
    pad_top: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if detections.numel() == 0:
        return rows

    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        score = float(confidence)
        if score < conf:
            continue

        x1_px = (float(x1) - float(pad_left)) / scale
        y1_px = (float(y1) - float(pad_top)) / scale
        x2_px = (float(x2) - float(pad_left)) / scale
        y2_px = (float(y2) - float(pad_top)) / scale

        x1_px = min(max(x1_px, 0.0), float(frame_width))
        y1_px = min(max(y1_px, 0.0), float(frame_height))
        x2_px = min(max(x2_px, 0.0), float(frame_width))
        y2_px = min(max(y2_px, 0.0), float(frame_height))

        if x2_px <= x1_px or y2_px <= y1_px:
            continue

        rows.append(
            {
                "x1": x1_px,
                "y1": y1_px,
                "x2": x2_px,
                "y2": y2_px,
                "confidence": score,
                "class_id": str(int(round(float(class_id)))),
            }
        )
    return rows


def _infer_frame(model: Any, frame: np.ndarray, *, conf: float) -> list[dict[str, float | str]]:
    torch = _import_torch()
    tensor, scale, pad_left, pad_top, frame_width, frame_height = _prepare_frame_tensor(frame)

    with torch.no_grad():
        raw_output = model(tensor)

    detections = _normalize_model_output(raw_output)
    return _postprocess_detections(
        detections,
        conf=conf,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _infer_frame_with_standard_yolov5(
    model: Any,
    frame: np.ndarray,
    *,
    conf: float,
) -> list[dict[str, float | str]]:
    frame_bgr = _coerce_frame_to_bgr(frame)
    if hasattr(model, "conf"):
        model.conf = float(conf)

    # Suppress noisy per-frame upstream deprecation warning from YOLOv5 internals.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
            category=FutureWarning,
        )
        results = model(frame_bgr, size=640)
    xyxy = getattr(results, "xyxy", None)
    if not isinstance(xyxy, list | tuple) or not xyxy:
        raise TypeError("YOLOv5 fallback inference did not return `results.xyxy` output.")

    detections = _normalize_model_output(xyxy[0])
    return _postprocess_detections(
        detections,
        conf=conf,
        scale=1.0,
        pad_left=0,
        pad_top=0,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
    )


def _iter_frame_indices(num_frames: int, *, show_progress: bool):
    return _progress_bar(
        range(num_frames),
        total=num_frames,
        desc="YOLO inference",
        unit="frame",
        leave=False,
        enabled=show_progress,
    )


def _run_yolo_on_video(
    original_video: np.ndarray,
    *,
    weight_path: str,
    conf: float,
    show_progress: bool,
) -> dict[int, list[dict[str, float | str]]]:
    model_kind = "torchscript"
    try:
        model = _load_torchscript_model(weight_path)
    except RuntimeError as exc:
        if "constants.pkl" not in str(exc):
            raise
        model = _load_standard_yolov5_model(weight_path)
        model_kind = "standard-checkpoint"

    frame_rows: dict[int, list[dict[str, float | str]]] = {}
    num_frames, _, _ = _ensure_video_dimensions(original_video)
    for frame_idx in _iter_frame_indices(num_frames, show_progress=show_progress):
        if model_kind == "torchscript":
            rows = _infer_frame(model, original_video[frame_idx], conf=conf)
        else:
            rows = _infer_frame_with_standard_yolov5(model, original_video[frame_idx], conf=conf)
        if rows:
            frame_rows[frame_idx] = rows
    return frame_rows


def _summarize_confidence_scores(
    confidence_by_frame: dict[str, list[float]],
) -> dict[str, float] | None:
    """Summarize flattened confidence scores with percentile and moment stats."""
    flattened: list[float] = []
    for frame_scores in confidence_by_frame.values():
        if not isinstance(frame_scores, list):
            continue
        for score in frame_scores:
            try:
                flattened.append(float(score))
            except (TypeError, ValueError):
                continue

    if not flattened:
        return None

    scores = np.asarray(flattened, dtype=float)
    return {
        "p05": float(np.percentile(scores, 5)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def yolov5(
    casa: dict[str, Any],
    weights: str = DEFAULT_MANAGED_WEIGHTS,
    conf: float = 0.15,
    delete_temp: bool = True,
    download: bool = True,
    force_download: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run managed YOLOv5 detection on the current in-memory video.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        weights (str, optional):
            Weight selector for inference.
            Managed values are full filenames like ``sys-opt_yolov5m.pt`` and
            ``sys-opt_yolov5n.pt``. You can also pass a custom local weight
            file path directly.
        conf (float, optional):
            Detection confidence threshold applied after model inference.
        delete_temp (bool, optional):
            Legacy compatibility flag. No temporary files are created by the
            in-process implementation, so this flag has no effect.
        download (bool, optional):
            If ``True``, automatically download missing managed weights from
            the official dataset repository.
        force_download (bool, optional):
            If ``True``, force re-download for managed weights even when a
            local cached file exists.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar while running
            per-frame YOLO inference.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries for YOLO
            execution. If ``False``, suppress those summaries. Warnings are
            not affected by this flag.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with normalized YOLO detections stored under
            ``casa['detections']['yolov5']`` and invocation metadata in
            ``casa['meta']['last_detection']``.

    Raises:
        ImportError:
            If the optional YOLO runtime dependencies are not installed.
        TypeError:
            If ``casa['video']['original_video']`` exists but is not a numpy array.
        ValueError:
            If ``weights`` is empty.
        FileNotFoundError:
            If a custom local weight path does not exist, if the requested
            managed weight is not cached locally yet, or if the current
            standard-checkpoint YOLOv5 runtime cannot find a local YOLOv5
            source checkout.
    """
    casa = _ensure_casa(casa)
    detections_root = casa.setdefault("detections", {})
    existing_predicted_methods = _predicted_detection_keys(detections_root)
    if existing_predicted_methods:
        _warn_yellow(
            "Previous detection result overwritten "
            f"({', '.join(existing_predicted_methods)} -> yolov5)."
        )
    _clear_predicted_detections(detections_root)

    weight_selector = str(weights).strip()
    if not weight_selector:
        raise ValueError("`weights` must not be empty.")

    managed_weights: str | None = None
    model_name: str | None = None
    weight_name: str | None = None
    if _is_managed_weight_name(weight_selector):
        managed_weights = weight_selector.lower()
        model_name, weight_name = _split_managed_weights(managed_weights)

    conf_threshold = float(conf)

    if casa.get("video", {}).get("original_video") is None:
        if verbose:
            print("Skipping YOLOv5: no original video is loaded.")
        detections_root["yolov5"] = {}
        casa["meta"]["last_detection"] = {
            "backend": "yolov5",
            "weights": weight_selector,
            "managed_weights": managed_weights,
            "model": model_name,
            "conf": conf_threshold,
            "weight_set": weight_name,
            "delete_temp": delete_temp,
            "skipped": True,
            "reason": "missing_original_video",
        }
        return casa
    original_video = _ensure_original_video(casa)

    num_frames, frame_height, frame_width = _ensure_video_dimensions(original_video)
    if managed_weights is not None:
        weight_path = _resolve_yolo_weight_path_from_name(
            managed_weights,
            allow_download=download,
            force_download=force_download,
        )
    else:
        weight_path = _resolve_custom_weight_path(weight_selector)

    if Path(weight_path).suffix.lower() == ".pt":
        _find_local_yolov5_repo()
    if verbose:
        print("Running YOLOv5 on frames...")
    frame_rows = _run_yolo_on_video(
        original_video,
        weight_path=weight_path,
        conf=conf_threshold,
        show_progress=show_progress,
    )

    initial_frame = int(casa.get("video", {}).get("initial_frame", 0) or 0)
    yolov5_detections: dict[str, list[list[str]]] = {}
    confidence_by_frame: dict[str, list[float]] = {}
    detections_found = 0

    for local_frame_idx in sorted(frame_rows.keys()):
        global_frame_idx = initial_frame + int(local_frame_idx)
        frame_key = str(global_frame_idx)
        formatted_rows: list[list[str]] = []
        formatted_scores: list[float] = []

        for row in frame_rows[local_frame_idx]:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            width_px = x2 - x1
            height_px = y2 - y1
            if width_px <= 0 or height_px <= 0:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            formatted_rows.append(
                [
                    str(row["class_id"]),
                    f"{min(max(center_x / frame_width, 0.0), 1.0):.6f}",
                    f"{min(max(center_y / frame_height, 0.0), 1.0):.6f}",
                    f"{min(max(width_px / frame_width, 0.0), 1.0):.6f}",
                    f"{min(max(height_px / frame_height, 0.0), 1.0):.6f}",
                ]
            )
            formatted_scores.append(float(row["confidence"]))

        if formatted_rows:
            yolov5_detections[frame_key] = formatted_rows
            confidence_by_frame[frame_key] = formatted_scores
            detections_found += len(formatted_rows)

    detections_root["yolov5"] = yolov5_detections
    tracked_sources = _resolve_sort_track_sources(casa.get("tracks", {}))
    if tracked_sources and "yolov5" not in tracked_sources:
        _warn_yellow(
            "Detections were updated to 'yolov5'. "
            "Re-run tracking to generate detection tracks."
        )
    confidence_summary = _summarize_confidence_scores(confidence_by_frame)
    casa["meta"]["last_detection"] = {
        "backend": "yolov5",
        "weights": weight_selector,
        "managed_weights": managed_weights,
        "model": model_name,
        "conf": conf_threshold,
        "weight_set": weight_name,
        "delete_temp": delete_temp,
        "weight_path": weight_path,
        "frames_processed": int(num_frames),
        "detections_found": detections_found,
        "confidence_by_frame": confidence_by_frame,
        "skipped": False,
    }
    if verbose:
        if confidence_summary is None:
            confidence_text = "confidence stats: none"
        else:
            confidence_text = (
                "confidence p5/p50/p95="
                f"{confidence_summary['p05']:.3f}/"
                f"{confidence_summary['p50']:.3f}/"
                f"{confidence_summary['p95']:.3f}, "
                f"mean+/-std={confidence_summary['mean']:.3f}"
                f"+/-{confidence_summary['std']:.3f}"
            )
        print(
            "YOLOv5 summary: "
            f"frames_processed={int(num_frames)}, "
            f"detections_found={detections_found}, "
            f"{confidence_text}"
        )
    return casa
