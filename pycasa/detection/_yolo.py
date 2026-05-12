import os
import shutil
import sys
import warnings
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_ID   = "DFL-KamLab/HSTLI_A-Dataset-of-Human-Semen-Time-Lapse-Images"
HF_REPO_TYPE = "dataset"

YOLOV5_WEIGHTS_DIRNAME = "yolov5-weights"
YOLO26_WEIGHTS_DIRNAME = "yolo26-weights"

VALID_YOLO_MODELS = ("yolov5", "yolo26")
VALID_WEIGHT_SETS = ("sys-casa", "sys-opt")
_MODEL_SIZES      = ("n", "s", "m", "l", "x")

VALID_MANAGED_YOLOV5 = tuple(
    f"{ws}_yolov5{sz}.pt"
    for ws in VALID_WEIGHT_SETS
    for sz in _MODEL_SIZES
)
VALID_MANAGED_YOLO26 = tuple(
    f"{ws}_yolo26{sz}.pt"
    for ws in VALID_WEIGHT_SETS
    for sz in _MODEL_SIZES
)

_DEFAULT_WEIGHTS: dict[str, str] = {
    "yolov5": "sys-casa_yolov5s.pt",
    "yolo26": "sys-casa_yolo26n.pt",
}


# ---------------------------------------------------------------------------
# Project root helpers
# ---------------------------------------------------------------------------
def _resolve_env_path(env_var: str | None) -> Path | None:
    if not env_var:
        return None
    raw = os.getenv(env_var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _find_project_root(start: Path, *, marker: str = "pyproject.toml") -> Path | None:
    cursor = start.resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / marker).is_file():
            return candidate
    return None


def _project_root() -> Path:
    env_override = _resolve_env_path("PYCASA_PROJECT_ROOT")
    if env_override is not None:
        return env_override
    detected = _find_project_root(Path(__file__))
    if detected is not None:
        return detected
    return Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------
def _normalize_weight_name(weights: str) -> str:
    return str(weights).strip().lower()


def _weights_dirname_for(yolo_model: str) -> str:
    return YOLO26_WEIGHTS_DIRNAME if yolo_model == "yolo26" else YOLOV5_WEIGHTS_DIRNAME


def _valid_managed_for(yolo_model: str) -> tuple[str, ...]:
    return VALID_MANAGED_YOLO26 if yolo_model == "yolo26" else VALID_MANAGED_YOLOV5


def _is_managed_weight_for(weights: str, yolo_model: str) -> bool:
    return _normalize_weight_name(weights) in _valid_managed_for(yolo_model)


def _split_managed_weight(weights: str) -> tuple[str, str]:
    """Return (model_variant, weight_set) from a managed weight filename."""
    name = _normalize_weight_name(weights)
    weight_set, model_with_ext = name.split("_", 1)
    model = model_with_ext[:-3]  # strip .pt
    return model, weight_set


def _local_managed_weight_path(weights: str, yolo_model: str) -> Path:
    model_variant, weight_set = _split_managed_weight(weights)
    dirname = _weights_dirname_for(yolo_model)
    return _project_root() / dirname / f"{weight_set}_{model_variant}.pt"


def _local_legacy_yolov5_path(weights: str) -> Path:
    """Old yolov5 layout: yolov5-weights/{weight_set}/{weight_set}_{model}.pt"""
    model_variant, weight_set = _split_managed_weight(weights)
    return (
        _project_root()
        / YOLOV5_WEIGHTS_DIRNAME
        / weight_set
        / f"{weight_set}_{model_variant}.pt"
    )


def _promote_legacy_yolov5_if_needed(weights: str) -> Path | None:
    local_path = _local_managed_weight_path(weights, "yolov5")
    if local_path.is_file():
        return local_path
    legacy_path = _local_legacy_yolov5_path(weights)
    if not legacy_path.is_file():
        return None
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_path, local_path)
    except OSError:
        return None
    return local_path


# Maps Python-interpreted control characters back to their backslash+letter form.
# Covers all single-char escape sequences that appear in Windows paths in practice.
_CTRL_TO_BACKSLASH: dict[str, str] = {
    "\x07": "\\a",  # \appdata, \assets …
    "\x08": "\\b",  # \bin, \build, \best …
    "\x0c": "\\f",  # \files …
    "\x0a": "\\n",  # rare in paths
    "\x0d": "\\r",  # \runs, \resources …
    "\x09": "\\t",  # \temp, \test …
    "\x0b": "\\v",  # rare in paths
}


def _unescape_windows_path(raw: str) -> str | None:
    """Reverse Python escape interpretation in a Windows path string.

    Returns the corrected string if any control chars were found, else None.
    """
    if not any(c in raw for c in _CTRL_TO_BACKSLASH):
        return None
    result = raw
    for ctrl, repl in _CTRL_TO_BACKSLASH.items():
        result = result.replace(ctrl, repl)
    return result


def _try_resolve_path(raw: str) -> Path | None:
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    return candidate if candidate.is_file() else None


def _resolve_custom_weight_path(weights: str) -> str:
    candidate = _try_resolve_path(weights)
    if candidate is not None:
        return str(candidate.resolve())

    # Windows paths typed without r'...' have \b, \r, \t etc. silently mangled
    # by Python's string parser. Try reversing that substitution.
    recovered = _unescape_windows_path(weights)
    if recovered is not None:
        candidate = _try_resolve_path(recovered)
        if candidate is not None:
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Custom `weights` path was not found: {weights!r}. "
        "On Windows use a raw string: r'D:\\path\\to\\weights.pt'"
    )


def _resolve_managed_weight_path(
    weights: str,
    yolo_model: str,
    *,
    allow_download: bool,
    force_download: bool = False,
) -> str:
    name = _normalize_weight_name(weights)

    if yolo_model == "yolov5":
        promoted = _promote_legacy_yolov5_if_needed(name)
        if promoted is not None and not force_download:
            return str(promoted)

    local_path = _local_managed_weight_path(name, yolo_model)
    if local_path.is_file() and not force_download:
        return str(local_path)

    if not allow_download:
        raise FileNotFoundError(
            f"The requested {yolo_model.upper()} weight file is not cached locally. "
            "Re-run with `download=True` or pass a local custom weights path. "
            f"Missing managed weights: `{name}`."
        )

    huggingface_hub = _ensure_import("huggingface_hub", pip_name="huggingface_hub")
    hf_hub_download = huggingface_hub.hf_hub_download

    model_variant, weight_set = _split_managed_weight(name)
    dirname = _weights_dirname_for(yolo_model)
    remote_filenames = [f"{dirname}/{weight_set}_{model_variant}.pt"]
    if yolo_model == "yolov5":
        remote_filenames.append(f"{dirname}/{weight_set}/{weight_set}_{model_variant}.pt")

    last_error: Exception | None = None
    for remote in remote_filenames:
        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    filename=remote,
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
    raise RuntimeError(f"Unable to resolve {yolo_model.upper()} managed weights.")


# ---------------------------------------------------------------------------
# YOLOv5 inference backend
# ---------------------------------------------------------------------------
@contextmanager
def _temporary_sys_path(path: str):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)


def _ensure_yolov5_pkg() -> Path:
    """Clone the yolov5 repo if not present, return its root directory."""
    import subprocess

    yolov5_dir = Path.home() / ".pycasa" / "yolov5"
    if (yolov5_dir / "hubconf.py").is_file():
        return yolov5_dir

    if sys.stdin.isatty():
        print(
            f"Required dependency 'yolov5' repo is missing.\n"
            f"Clone https://github.com/ultralytics/yolov5.git to {yolov5_dir}? [y/N]: ",
            end="",
            flush=True,
        )
        if input().strip().lower() != "y":
            raise SystemExit(
                "YOLOv5 repo not cloned. Re-run and accept, or clone manually:\n"
                f"  git clone https://github.com/ultralytics/yolov5.git {yolov5_dir}"
            )
    else:
        # Non-interactive: clone without prompting (CI / script use)
        print(f"[INFO] Cloning yolov5 repo to {yolov5_dir} ...")

    yolov5_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/ultralytics/yolov5.git",
         str(yolov5_dir)],
        check=True,
    )
    return yolov5_dir


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


def _load_torchscript_model(weight_path: str) -> Any:
    torch = _ensure_import("torch")
    model = torch.jit.load(weight_path, map_location="cpu")
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def _load_standard_yolov5_model(weight_path: str) -> Any:
    torch = _ensure_import("torch")
    for dep in ("torchvision", "pandas", "requests", "yaml", "psutil", "PIL"):
        _ensure_import(dep)
    yolov5_dir = str(_ensure_yolov5_pkg())
    with _temporary_sys_path(yolov5_dir):
        from models.common import AutoShape, DetectMultiBackend
        model = DetectMultiBackend(
            weight_path,
            device=torch.device("cpu"),
            fp16=False,
            fuse=False,
        )
        return AutoShape(model, verbose=False)


def _prepare_frame_tensor(
    frame: np.ndarray,
    *,
    image_size: int = 640,
) -> tuple[Any, float, int, int, int, int]:
    torch = _ensure_import("torch")
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


def _normalize_yolov5_output(output: Any) -> Any:
    torch = _ensure_import("torch")
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


def _postprocess_yolov5_detections(
    detections: Any,
    *,
    conf: float,
    scale: float,
    pad_left: int,
    pad_top: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if detections.numel() == 0:
        return rows
    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        score = float(confidence)
        if score < conf:
            continue
        x1_px = min(max((float(x1) - pad_left) / scale, 0.0), float(frame_width))
        y1_px = min(max((float(y1) - pad_top) / scale, 0.0), float(frame_height))
        x2_px = min(max((float(x2) - pad_left) / scale, 0.0), float(frame_width))
        y2_px = min(max((float(y2) - pad_top) / scale, 0.0), float(frame_height))
        if x2_px <= x1_px or y2_px <= y1_px:
            continue
        rows.append({
            "class_id": int(round(float(class_id))),
            "cx": ((x1_px + x2_px) / 2.0) / frame_width,
            "cy": ((y1_px + y2_px) / 2.0) / frame_height,
            "w": (x2_px - x1_px) / frame_width,
            "h": (y2_px - y1_px) / frame_height,
            "confidence": score,
        })
    return rows


def _infer_yolov5_torchscript(
    model: Any, frame: np.ndarray, *, conf: float
) -> list[dict[str, Any]]:
    torch = _ensure_import("torch")
    tensor, scale, pad_left, pad_top, frame_width, frame_height = _prepare_frame_tensor(frame)
    with torch.no_grad():
        raw_output = model(tensor)
    detections = _normalize_yolov5_output(raw_output)
    return _postprocess_yolov5_detections(
        detections,
        conf=conf,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _infer_yolov5_standard(
    model: Any, frame: np.ndarray, *, conf: float
) -> list[dict[str, Any]]:
    frame_bgr = _coerce_frame_to_bgr(frame)
    frame_height, frame_width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if hasattr(model, "conf"):
        model.conf = float(conf)
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
    detections = _normalize_yolov5_output(xyxy[0])
    return _postprocess_yolov5_detections(
        detections,
        conf=conf,
        scale=1.0,
        pad_left=0,
        pad_top=0,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _run_yolov5_on_video(
    original_video: np.ndarray,
    *,
    weight_path: str,
    conf: float,
    show_progress: bool,
) -> dict[int, list[dict[str, Any]]]:
    model_kind = "torchscript"
    try:
        model = _load_torchscript_model(weight_path)
    except RuntimeError as exc:
        if "constants.pkl" not in str(exc):
            raise
        model = _load_standard_yolov5_model(weight_path)
        model_kind = "standard-checkpoint"

    frame_rows: dict[int, list[dict[str, Any]]] = {}
    num_frames, _, _ = _ensure_video_dimensions(original_video)
    for frame_idx in _progress_bar(
        range(num_frames),
        total=num_frames,
        desc="YOLOv5 inference",
        unit="frame",
        leave=False,
        enabled=show_progress,
    ):
        if model_kind == "torchscript":
            rows = _infer_yolov5_torchscript(model, original_video[frame_idx], conf=conf)
        else:
            rows = _infer_yolov5_standard(model, original_video[frame_idx], conf=conf)
        if rows:
            frame_rows[frame_idx] = rows
    return frame_rows


# ---------------------------------------------------------------------------
# Ultralytics inference backend (yolo26+)
# ---------------------------------------------------------------------------
def _load_ultralytics_model(weight_path: str) -> Any:
    YOLO = _ensure_import("ultralytics", pip_name="ultralytics").YOLO
    return YOLO(weight_path)


def _infer_ultralytics_frame(
    model: Any, frame: np.ndarray, *, conf: float
) -> list[dict[str, Any]]:
    frame_bgr = _coerce_frame_to_bgr(frame)
    results = model(frame_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()
    confs  = boxes.conf.cpu().numpy()
    clss   = boxes.cls.cpu().numpy()
    rows: list[dict[str, Any]] = []
    for i in range(len(xywhn)):
        cx, cy, w, h = (float(xywhn[i, 0]), float(xywhn[i, 1]),
                        float(xywhn[i, 2]), float(xywhn[i, 3]))
        if w <= 0 or h <= 0:
            continue
        rows.append({
            "class_id": int(clss[i]),
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "confidence": float(confs[i]),
        })
    return rows


def _run_ultralytics_on_video(
    original_video: np.ndarray,
    *,
    weight_path: str,
    conf: float,
    show_progress: bool,
) -> dict[int, list[dict[str, Any]]]:
    model = _load_ultralytics_model(weight_path)
    frame_rows: dict[int, list[dict[str, Any]]] = {}
    num_frames, _, _ = _ensure_video_dimensions(original_video)
    for frame_idx in _progress_bar(
        range(num_frames),
        total=num_frames,
        desc="YOLO26 inference",
        unit="frame",
        leave=False,
        enabled=show_progress,
    ):
        rows = _infer_ultralytics_frame(model, original_video[frame_idx], conf=conf)
        if rows:
            frame_rows[frame_idx] = rows
    return frame_rows


# ---------------------------------------------------------------------------
# Confidence summarization
# ---------------------------------------------------------------------------
def _summarize_confidence_scores(
    confidence_by_frame: dict[str, list[float]],
) -> dict[str, float] | None:
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


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def yolo(
    casa: dict[str, Any],
    yolo_model: str = "yolov5",
    weights: str | None = None,
    conf: float = 0.15,
    download: bool = True,
    force_download: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run YOLO detection on the current in-memory video.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        yolo_model (str, optional):
            YOLO architecture to use: ``"yolov5"`` or ``"yolo26"``.
        weights (str | None, optional):
            Managed weight name or a custom local file path. When ``None``,
            the default managed weight for the chosen ``yolo_model`` is used.

            **YOLOv5 managed weights** (downloaded automatically from HuggingFace):

            - ``sys-casa_yolov5n.pt`` — nano, fastest
            - ``sys-casa_yolov5s.pt`` — small *(default)*
            - ``sys-casa_yolov5m.pt`` — medium
            - ``sys-casa_yolov5l.pt`` — large
            - ``sys-casa_yolov5x.pt`` — extra-large
            - ``sys-opt_yolov5n.pt``
            - ``sys-opt_yolov5s.pt``
            - ``sys-opt_yolov5m.pt``
            - ``sys-opt_yolov5l.pt``
            - ``sys-opt_yolov5x.pt``

            **YOLO26 managed weights** (not yet public — contact
            Atilla Sivri atilla.sivri@njit.edu or
            Ludvik Alkhoury ludvik.alkhoury@gmail.com for access):

            - ``sys-casa_yolo26n.pt`` — nano *(default)*
            - ``sys-casa_yolo26s.pt`` — small
            - ``sys-casa_yolo26m.pt`` — medium
            - ``sys-casa_yolo26l.pt`` — large
            - ``sys-casa_yolo26x.pt`` — extra-large
            - ``sys-opt_yolo26n.pt``
            - ``sys-opt_yolo26s.pt``
            - ``sys-opt_yolo26m.pt``
            - ``sys-opt_yolo26l.pt``
            - ``sys-opt_yolo26x.pt``

        conf (float, optional):
            Detection confidence threshold.
        download (bool, optional):
            If ``True``, automatically download missing managed weights from
            the official dataset repository.
        force_download (bool, optional):
            If ``True``, force re-download even when a local cache exists.
        show_progress (bool, optional):
            If ``True``, show the shared pycasa progress bar during inference.
        verbose (bool, optional):
            If ``True``, print concise runtime start/end summaries.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with normalized detections stored under
            ``casa['detections'][yolo_model]`` and metadata in
            ``casa['meta']['last_detection']``.

    Raises:
        ValueError:
            If ``yolo_model`` is not one of ``"yolov5"`` or ``"yolo26"``, or
            if ``weights`` is empty.
        FileNotFoundError:
            If a custom local weight path does not exist, or the requested
            managed weight is not cached and ``download=False``.
    """
    if yolo_model not in VALID_YOLO_MODELS:
        raise ValueError(
            f"`yolo_model` must be one of {VALID_YOLO_MODELS}, got {yolo_model!r}."
        )

    casa = _ensure_casa(casa)
    detections_root = casa.setdefault("detections", {})
    existing_predicted_methods = _predicted_detection_keys(detections_root)
    if existing_predicted_methods:
        _warn_yellow(
            "Previous detection result overwritten "
            f"({', '.join(existing_predicted_methods)} -> {yolo_model})."
        )
    _clear_predicted_detections(detections_root)

    weight_selector = (
        str(weights).strip() if weights is not None else _DEFAULT_WEIGHTS[yolo_model]
    )
    if not weight_selector:
        raise ValueError("`weights` must not be empty.")

    is_managed = _is_managed_weight_for(weight_selector, yolo_model)

    # Warn if the weight name looks like it belongs to the other YOLO version.
    other_model = "yolo26" if yolo_model == "yolov5" else "yolov5"
    if not is_managed and _is_managed_weight_for(weight_selector, other_model):
        _warn_yellow(
            f"Weight name {weight_selector!r} looks like a {other_model} weight "
            f"but `yolo_model` is {yolo_model!r}. Pass the correct `yolo_model`."
        )

    managed_weights: str | None = None
    model_name: str | None = None
    weight_name: str | None = None
    if is_managed:
        managed_weights = _normalize_weight_name(weight_selector)
        model_name, weight_name = _split_managed_weight(managed_weights)

    conf_threshold = float(conf)

    # yolo26 managed weights are not yet public on HuggingFace.
    if yolo_model == "yolo26" and is_managed:
        _warn_yellow(
            "YOLO26 weights are not yet publicly available on HuggingFace. "
            "Use a custom local weights path instead. "
            "To obtain access to unpublished weights, contact: "
            "Atilla Sivri (atilla.sivri@njit.edu) or "
            "Ludvik Alkhoury (ludvik.alkhoury@gmail.com)."
        )
        return casa

    if casa.get("video", {}).get("original_video") is None:
        if verbose:
            print(f"Skipping {yolo_model.upper()}: no original video is loaded.")
        detections_root[yolo_model] = {}
        casa["meta"]["last_detection"] = {
            "backend": yolo_model,
            "weights": weight_selector,
            "managed_weights": managed_weights,
            "model": model_name,
            "weight_set": weight_name,
            "conf": conf_threshold,
            "skipped": True,
            "reason": "missing_original_video",
        }
        return casa

    original_video = _ensure_original_video(casa)
    num_frames, frame_height, frame_width = _ensure_video_dimensions(original_video)

    if is_managed:
        weight_path = _resolve_managed_weight_path(
            managed_weights,
            yolo_model,
            allow_download=download,
            force_download=force_download,
        )
    else:
        weight_path = _resolve_custom_weight_path(weight_selector)

    if verbose:
        print(f"Running {yolo_model.upper()} on frames...")

    if yolo_model == "yolov5":
        frame_rows = _run_yolov5_on_video(
            original_video,
            weight_path=weight_path,
            conf=conf_threshold,
            show_progress=show_progress,
        )
    else:
        frame_rows = _run_ultralytics_on_video(
            original_video,
            weight_path=weight_path,
            conf=conf_threshold,
            show_progress=show_progress,
        )

    initial_frame = int(casa.get("video", {}).get("initial_frame", 0) or 0)
    yolo_detections: dict[str, list[list[str]]] = {}
    confidence_by_frame: dict[str, list[float]] = {}
    detections_found = 0

    for local_frame_idx in sorted(frame_rows.keys()):
        global_frame_idx = initial_frame + int(local_frame_idx)
        frame_key = str(global_frame_idx)
        formatted_rows: list[list[str]] = []
        formatted_scores: list[float] = []

        for row in frame_rows[local_frame_idx]:
            cx = float(row["cx"])
            cy = float(row["cy"])
            w  = float(row["w"])
            h  = float(row["h"])
            if w <= 0 or h <= 0:
                continue
            formatted_rows.append([
                str(row["class_id"]),
                f"{min(max(cx, 0.0), 1.0):.6f}",
                f"{min(max(cy, 0.0), 1.0):.6f}",
                f"{min(max(w,  0.0), 1.0):.6f}",
                f"{min(max(h,  0.0), 1.0):.6f}",
            ])
            formatted_scores.append(float(row["confidence"]))

        if formatted_rows:
            yolo_detections[frame_key] = formatted_rows
            confidence_by_frame[frame_key] = formatted_scores
            detections_found += len(formatted_rows)

    detections_root[yolo_model] = yolo_detections
    tracked_sources = _resolve_sort_track_sources(casa.get("tracks", {}))
    if tracked_sources and yolo_model not in tracked_sources:
        _warn_yellow(
            f"Detections were updated to '{yolo_model}'. "
            "Re-run tracking to generate detection tracks."
        )
    confidence_summary = _summarize_confidence_scores(confidence_by_frame)
    casa["meta"]["last_detection"] = {
        "backend": yolo_model,
        "weights": weight_selector,
        "managed_weights": managed_weights,
        "model": model_name,
        "weight_set": weight_name,
        "conf": conf_threshold,
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
            f"{yolo_model.upper()} summary: "
            f"frames_processed={int(num_frames)}, "
            f"detections_found={detections_found}, "
            f"{confidence_text}"
        )
    return casa
