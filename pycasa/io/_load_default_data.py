import os
from pathlib import Path

from ..utils import _ensure_import
from ._load_video import load_video

_DATASET_REPO_ID = "DFL-KamLab/HSTLI_A-Dataset-of-Human-Semen-Time-Lapse-Images"
_DATASET_REPO_TYPE = "dataset"
_SESSION_REL = Path("sys-casa/rawdata/sub-HC004/ses-01")
_VIDEO_NAME = "sys-casa_sub-HC004_ses-01_run-005_video.avi"
_VIDEO_META_NAME = "sys-casa_sub-HC004_ses-01_run-005_video.json"
_GROUNDTRUTH_DIR_NAME = "sys-casa_sub-HC004_ses-01_run-005_gt"
_README_NAME = "README.md"


def _resolve_data_root(path: str | None) -> Path:
    """Resolve default-data root from argument, env var, or user home fallback."""
    if path:
        return Path(path).expanduser().resolve()
    env_data_path = os.getenv("PYCASA_DATA")
    if env_data_path:
        return Path(env_data_path).expanduser().resolve()
    return (Path.home() / ".pycasa_data").resolve()


def _required_paths(root: Path) -> dict[str, Path]:
    """Build the required HC004 session file/directory paths under ``root``."""
    session_dir = root / _SESSION_REL
    return {
        "session_dir": session_dir,
        "video_path": session_dir / _VIDEO_NAME,
        "video_json_path": session_dir / _VIDEO_META_NAME,
        "groundtruth_path": session_dir / _GROUNDTRUTH_DIR_NAME,
        "readme_path": session_dir / _README_NAME,
    }


def _missing_required(paths: dict[str, Path]) -> list[Path]:
    """Return required files/directories that are missing from disk."""
    missing: list[Path] = []
    for key in ("video_path", "video_json_path", "groundtruth_path", "readme_path"):
        path = paths[key]
        if key == "groundtruth_path":
            if not path.is_dir():
                missing.append(path)
        else:
            if not path.is_file():
                missing.append(path)
    return missing


def _download_required_subset(root: Path) -> None:
    """Download only the minimal default-session subset into ``root``."""
    huggingface_hub = _ensure_import(
        "huggingface_hub",
        pip_name="huggingface_hub",
        prompt_install=True,
        terminate_on_decline=True,
    )
    snapshot_download = getattr(huggingface_hub, "snapshot_download")

    allow_patterns = [
        f"{_SESSION_REL.as_posix()}/{_GROUNDTRUTH_DIR_NAME}/*",
        f"{_SESSION_REL.as_posix()}/{_README_NAME}",
        f"{_SESSION_REL.as_posix()}/{_VIDEO_NAME}",
        f"{_SESSION_REL.as_posix()}/{_VIDEO_META_NAME}",
    ]
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=_DATASET_REPO_ID,
        repo_type=_DATASET_REPO_TYPE,
        allow_patterns=allow_patterns,
        local_dir=str(root),
    )


def load_default_data(
    path: str | None = None,
    *,
    download: bool = True,
    force_download: bool = False,
    initial_frame: int = 0,
    final_frame: int | None = 100,
    sampling_rate: float | None = None,
    um_per_px: float | None = None,
    magnification: str | None = None,
    verbose: bool = True,
) -> "Casa":
    """Load the default HC004 session with MNE-style local caching.

    Parameters:
        path (str | None, optional):
            Root folder to cache/load default data. Resolution order:
            explicit argument, ``PYCASA_DATA``, then ``~/.pycasa_data``.
        download (bool, optional):
            If ``True``, missing required files trigger a selective download.
        force_download (bool, optional):
            If ``True``, always run the selective download before loading.
        initial_frame (int, optional):
            First frame index to read (0-based). Defaults to ``0``.
        final_frame (int | None, optional):
            Last frame index to read (0-based, inclusive). Defaults to ``100``.
        sampling_rate (float | None, optional):
            Optional FPS override forwarded to :func:`pycasa.io.load_video`.
        um_per_px (float | None, optional):
            Optional microns-per-pixel metadata value.
        magnification (str | None, optional):
            Optional magnification metadata value.
        verbose (bool, optional):
            If ``True``, print concise runtime summaries for cache/download
            resolution. If ``False``, suppress those summaries.

    Returns:
        Casa:
            Fluent ``Casa`` object loaded from the default video and groundtruth folder.

    Raises:
        FileNotFoundError:
            If required files are missing and download is disabled, or if they
            remain missing after download.
        ImportError:
            If download is needed and ``huggingface_hub`` is unavailable.

    Notes:
        Only this subset is downloaded:
        - ``sys-casa_sub-HC004_ses-01_run-005_video.avi``
        - ``sys-casa_sub-HC004_ses-01_run-005_video.json``
        - ``sys-casa_sub-HC004_ses-01_run-005_gt/*``
        - ``README.md`` in the same session folder.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.io.load_default_data(download=False)
        >>> isinstance(session, pc.Casa)
        True
    """
    root = _resolve_data_root(path)
    paths = _required_paths(root)
    if verbose:
        print(f"Resolving default data under '{root}'...")

    if force_download:
        if verbose:
            print("force_download=True, downloading required default dataset subset...")
        _download_required_subset(root)
        paths = _required_paths(root)

    missing = _missing_required(paths)
    if missing:
        if not download:
            missing_str = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Default data files are missing and download is disabled. Missing:\n"
                f"{missing_str}"
            )
        if verbose:
            print("Missing required files, downloading default dataset subset...")
        _download_required_subset(root)
        paths = _required_paths(root)
        missing = _missing_required(paths)
        if missing:
            missing_str = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Default data download completed but required files are still missing:\n"
                f"{missing_str}"
            )
    elif verbose:
        print(f"Default data cache is ready: {root}")

    if verbose:
        print(
            "Loading default session video and groundtruth: "
            f"{paths['video_path'].name}"
        )

    return load_video(
        video_path=str(paths["video_path"]),
        groundtruth_path=str(paths["groundtruth_path"]),
        initial_frame=initial_frame,
        final_frame=final_frame,
        sampling_rate=sampling_rate,
        um_per_px=um_per_px,
        magnification=magnification,
        verbose=verbose,
    )
