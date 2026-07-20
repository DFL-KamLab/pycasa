from typing import Any

from .._core._casa import _ensure_casa
from ..utils import _resolve_active_tracking_backend
from ..utils import _resolve_sort_track_sources
from ..utils import _warn_yellow
from ..utils import _GROUNDTRUTH_TRACKS_KEY
from ._kinematic_parameters import _resolve_video_size

_KINEMATIC_KEY = "kinematic_parameters"
_RESULT_KEY = "casa_parameters"
_VELOCITY_METRICS = ("VAP", "VCL", "VSL")
_GRADES = ("rapid", "slow", "non_progressive", "immotile")

# 1 mL = 1 cm^3 = 1e12 um^3, so 1 um^3 = 1e-12 mL.
_UM3_PER_ML = 1e12

# Standard counting-chamber depth (microns) assumed when none is provided.
_DEFAULT_CHAMBER_DEPTH_UM = 20.0


def _to_int(value: Any) -> int | None:
    """Return an integer when conversion succeeds, otherwise ``None``."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean of a non-empty numeric list, else ``None``."""
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(clean) / len(clean) if clean else None


def _classify_track(
    velocity: float,
    straightness: float,
    rapid_threshold: float,
    immotile_threshold: float,
    progressive_str_threshold: float,
) -> str:
    """Assign one WHO motility grade to a track from its mean velocity/STR.

    - ``immotile``       : velocity below ``immotile_threshold`` (grade d).
    - ``non_progressive``: motile but STR below the progressive cutoff (grade c).
    - ``slow``           : progressive, velocity below ``rapid_threshold`` (grade b).
    - ``rapid``          : progressive, velocity at/above ``rapid_threshold`` (grade a).
    """
    if velocity < immotile_threshold:
        return "immotile"
    if straightness >= progressive_str_threshold:
        return "rapid" if velocity >= rapid_threshold else "slow"
    return "non_progressive"


def _source_track_map(
    casa: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Return the raw ``{track_id: {frame: [x, y]}}`` map backing a kinematic source."""
    tracks_root = casa.get("tracks", {})
    if not isinstance(tracks_root, dict):
        return {}
    if source_name == _GROUNDTRUTH_TRACKS_KEY:
        candidate = tracks_root.get(_GROUNDTRUTH_TRACKS_KEY)
    else:
        candidate = _resolve_sort_track_sources(tracks_root).get(source_name)
    return candidate if isinstance(candidate, dict) else {}


def _mean_cells_per_frame(track_map: dict[str, Any]) -> float | None:
    """Mean number of concurrently present cells per annotated frame."""
    per_frame: dict[int, int] = {}
    for points in track_map.values():
        if not isinstance(points, dict):
            continue
        for frame_key in points:
            frame = _to_int(frame_key)
            if frame is not None:
                per_frame[frame] = per_frame.get(frame, 0) + 1
    if not per_frame:
        return None
    return sum(per_frame.values()) / len(per_frame)


def _round(value: float | None, digits: int = 2) -> float | None:
    """Round a value to ``digits`` places, passing ``None`` through."""
    return None if value is None else round(float(value), digits)


def _concentration_million_per_ml(
    cells_per_frame: float | None,
    width_px: int,
    height_px: int,
    um_per_px: float | None,
    chamber_depth_um: float | None,
) -> float | None:
    """Sperm concentration (10^6/mL) from field count, calibration and depth."""
    if (
        cells_per_frame is None
        or um_per_px is None
        or chamber_depth_um is None
        or width_px <= 0
        or height_px <= 0
        or um_per_px <= 0
        or chamber_depth_um <= 0
    ):
        return None
    area_um2 = (width_px * um_per_px) * (height_px * um_per_px)
    field_volume_ml = (area_um2 * chamber_depth_um) / _UM3_PER_ML
    if field_volume_ml <= 0:
        return None
    concentration_per_ml = cells_per_frame / field_volume_ml
    return concentration_per_ml / 1e6


def _summarize_source(
    casa: dict[str, Any],
    source_name: str,
    track_metrics_map: dict[str, Any],
    velocity_metric: str,
    rapid_threshold: float,
    immotile_threshold: float,
    progressive_str_threshold: float,
    um_per_px: float | None,
    chamber_depth_um: float | None,
    volume_ml: float | None,
) -> dict[str, Any]:
    """Classify every track in one source and assemble its CASA parameters."""
    counts = dict.fromkeys(_GRADES, 0)
    for track_metrics in track_metrics_map.values():
        if not isinstance(track_metrics, dict):
            continue
        velocity = _mean(track_metrics.get(velocity_metric) or [])
        straightness = _mean(track_metrics.get("STR") or [])
        if velocity is None or straightness is None:
            continue
        grade = _classify_track(
            velocity,
            straightness,
            rapid_threshold,
            immotile_threshold,
            progressive_str_threshold,
        )
        counts[grade] += 1

    total = sum(counts.values())
    if total == 0:
        return {"skipped": True, "reason": "no_classifiable_tracks", "track_count": 0}

    grades_pct = {grade: _round(counts[grade] / total * 100.0) for grade in _GRADES}

    width_px, height_px = _resolve_video_size(casa)
    cells_per_frame = _mean_cells_per_frame(_source_track_map(casa, source_name))
    concentration = _concentration_million_per_ml(
        cells_per_frame, width_px, height_px, um_per_px, chamber_depth_um
    )
    total_count = (
        concentration * volume_ml
        if (concentration is not None and volume_ml is not None)
        else None
    )

    return {
        "grades": grades_pct,
        "percent_motile": _round(100.0 - counts["immotile"] / total * 100.0),
        "counts": counts,
        "track_count": total,
        "cells_per_frame": _round(cells_per_frame),
        "concentration_M_per_ml": _round(concentration),
        "volume_ml": _round(volume_ml) if volume_ml is not None else None,
        "total_sperm_count_M": _round(total_count),
        "velocity_metric": velocity_metric,
        "thresholds": {
            "rapid": float(rapid_threshold),
            "immotile": float(immotile_threshold),
            "progressive_str": float(progressive_str_threshold),
        },
        "skipped": False,
    }


def _print_source_summary(backend: str, source_name: str, summary: dict[str, Any]) -> None:
    """Print a concise CASA-parameter summary for one source."""
    if summary.get("skipped"):
        print(f"CASA parameters ({backend}:{source_name}): skipped ({summary.get('reason')})")
        return
    grades = summary["grades"]
    thr = summary["thresholds"]
    print(f"CASA parameters ({backend}:{source_name})")
    print(
        f"- tracks classified={summary['track_count']} "
        f"(metric={summary['velocity_metric']}, rapid>={thr['rapid']:g}, "
        f"immotile<{thr['immotile']:g} um/s, progressive STR>={thr['progressive_str']:g})"
    )
    print(
        f"- %rapid={grades['rapid']}, %slow={grades['slow']}, "
        f"%non-progressive={grades['non_progressive']}, %immotile={grades['immotile']} "
        f"(%motile={summary['percent_motile']})"
    )
    if summary["concentration_M_per_ml"] is not None:
        print(
            f"- concentration={summary['concentration_M_per_ml']} x10^6/mL "
            f"(cells/frame={summary['cells_per_frame']})"
        )
    else:
        print("- concentration=n/a (needs um_per_px and tracked cells)")
    if summary["volume_ml"] is not None:
        print(f"- volume={summary['volume_ml']} mL")
    if summary["total_sperm_count_M"] is not None:
        print(f"- total sperm count={summary['total_sperm_count_M']} x10^6")


def casa_parameters(
    casa: dict[str, Any],
    rapid_threshold: float = 25.0,
    immotile_threshold: float = 5.0,
    progressive_str_threshold: float = 0.8,
    velocity_metric: str = "VAP",
    volume_ml: float | None = None,
    chamber_depth_um: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute population-level CASA parameters from kinematic parameters.

    For every source in ``casa["motility"]["kinematic_parameters"]`` each track
    is classified into one of the four WHO motility grades and the population
    percentages are reported: ``%rapid`` (grade a), ``%slow`` (grade b),
    ``%non_progressive`` (grade c) and ``%immotile`` (grade d). A track is
    graded from the mean of its chosen velocity metric and mean ``STR``:

    - ``immotile``        : velocity ``< immotile_threshold``.
    - ``non_progressive`` : motile but ``STR < progressive_str_threshold``.
    - ``slow``            : progressive, velocity ``< rapid_threshold``.
    - ``rapid``           : progressive, velocity ``>= rapid_threshold``.

    Optional physical quantities are added when their inputs are available and
    silently omitted otherwise:

    - **concentration** (10^6/mL) — needs ``um_per_px`` (``chamber_depth_um``
      defaults to ``20``).
    - **volume** (mL) — needs ``volume_ml``.
    - **total sperm count** (10^6) — ``volume_ml x concentration``; needs both.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary; ``kinematic_parameters`` must have been run.
        rapid_threshold (float, optional):
            Velocity (um/s) for the rapid/slow split. Default ``25`` (WHO).
        immotile_threshold (float, optional):
            Velocity (um/s) below which a track is immotile. Default ``5``.
        progressive_str_threshold (float, optional):
            STR (ratio in ``[0, 1]``) for the progressive/non-progressive split.
            Default ``0.8``.
        velocity_metric (str, optional):
            ``"VAP"`` (default), ``"VCL"`` or ``"VSL"``.
        volume_ml (float | None, optional):
            Ejaculate volume (mL); falls back to ``casa["meta"]["volume_ml"]``.
        chamber_depth_um (float | None, optional):
            Counting-chamber depth (um). Resolved as argument, then
            ``casa["meta"]["chamber_depth_um"]``, then the ``20`` um default.
        verbose (bool, optional):
            If ``True``, print a per-source summary.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with output under
            ``casa["motility"]["casa_parameters"][source]`` and run metadata in
            ``casa["meta"]["last_casa_parameters"]``.

    Raises:
        ValueError:
            If ``velocity_metric`` is not one of ``VAP``, ``VCL``, ``VSL``.
        RuntimeError:
            If ``kinematic_parameters`` output is missing.
    """
    casa = _ensure_casa(casa)

    velocity_metric = str(velocity_metric).upper()
    if velocity_metric not in _VELOCITY_METRICS:
        raise ValueError(
            f"`velocity_metric` must be one of {_VELOCITY_METRICS}, got {velocity_metric!r}."
        )

    meta = casa.setdefault("meta", {})
    if volume_ml is None:
        volume_ml = meta.get("volume_ml")
    if chamber_depth_um is None:
        chamber_depth_um = meta.get("chamber_depth_um")
    if chamber_depth_um is None:
        # Standard counting-chamber depth (e.g. Leja/Makler) when unspecified.
        chamber_depth_um = _DEFAULT_CHAMBER_DEPTH_UM
    volume_ml = float(volume_ml) if volume_ml is not None else None
    chamber_depth_um = float(chamber_depth_um) if chamber_depth_um is not None else None
    um_per_px = meta.get("um_per_px")
    um_per_px = float(um_per_px) if isinstance(um_per_px, (int, float)) else None

    motility_root = casa.setdefault("motility", {})
    kinematic_root = motility_root.get(_KINEMATIC_KEY)
    if not isinstance(kinematic_root, dict) or not kinematic_root:
        raise RuntimeError(
            "No kinematic parameters found. Run "
            "`self.motility.kinematic_parameters()` before `casa_parameters()`."
        )

    if um_per_px is None:
        _warn_yellow(
            "um_per_px is not set; velocity thresholds assume um/s but the "
            "kinematic velocities are in px/s, so grades may be meaningless. "
            "Set calibration with self.set_um_per_px(...)."
        )

    backend = _resolve_active_tracking_backend(casa.get("tracks", {})) or "sort"
    results: dict[str, dict[str, Any]] = {}
    processed: list[str] = []

    ordered = sorted(kinematic_root.keys(), key=lambda v: (v != "groundtruth", v))
    for source_name in ordered:
        track_metrics_map = kinematic_root.get(source_name)
        if not isinstance(track_metrics_map, dict) or not track_metrics_map:
            continue
        source_backend = "imported" if source_name == _GROUNDTRUTH_TRACKS_KEY else backend
        summary = _summarize_source(
            casa,
            source_name,
            track_metrics_map,
            velocity_metric,
            rapid_threshold,
            immotile_threshold,
            progressive_str_threshold,
            um_per_px,
            chamber_depth_um,
            volume_ml,
        )
        results[source_name] = summary
        if not summary.get("skipped"):
            processed.append(source_name)
        if verbose:
            _print_source_summary(source_backend, source_name, summary)

    if _RESULT_KEY in motility_root:
        _warn_yellow("Previous casa_parameters result overwritten.")
    motility_root[_RESULT_KEY] = results

    meta["last_casa_parameters"] = {
        "sources_processed": processed,
        "velocity_metric": velocity_metric,
        "rapid_threshold": float(rapid_threshold),
        "immotile_threshold": float(immotile_threshold),
        "progressive_str_threshold": float(progressive_str_threshold),
        "um_per_px": um_per_px,
        "volume_ml": volume_ml,
        "chamber_depth_um": chamber_depth_um,
        "skipped": not bool(processed),
    }
    return casa
