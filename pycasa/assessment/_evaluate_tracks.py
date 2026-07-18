from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import _ensure_import
from ..utils import _resolve_active_tracking_backend
from ..utils import _resolve_sort_track_sources
from ..utils import _GROUNDTRUTH_TRACKS_KEY
from ._evaluate_detections import _to_int_frame

_GT_LABEL = _GROUNDTRUTH_TRACKS_KEY


def _tracks_by_frame(tracks: dict[str, Any]) -> dict[int, dict[str, tuple[float, float]]]:
    """Invert ``{track_id: {frame: [x, y]}}`` to ``{frame: {track_id: (x, y)}}``."""
    by_frame: dict[int, dict[str, tuple[float, float]]] = {}
    for track_id, points in tracks.items():
        if not isinstance(points, dict):
            continue
        for frame_key, coord in points.items():
            frame = _to_int_frame(frame_key)
            if frame is None:
                continue
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            try:
                x = float(coord[0])
                y = float(coord[1])
            except (TypeError, ValueError):
                continue
            by_frame.setdefault(frame, {})[str(track_id)] = (x, y)
    return by_frame


def _id_int_map(tracks: dict[str, Any]) -> dict[str, int]:
    """Stable string-id -> int map (motmetrics stores ids in a float series)."""
    return {str(tid): idx for idx, tid in enumerate(tracks.keys())}


def _resolve_eval_frames(
    casa: dict[str, Any],
    a_by_frame: dict[int, Any],
    b_by_frame: dict[int, Any],
) -> list[int]:
    """Evaluation frames: union of both sets' frames, clamped to the video range."""
    universe = set(a_by_frame.keys()) | set(b_by_frame.keys())
    if not universe:
        return []
    initial_frame = _to_int_frame(casa.get("video", {}).get("initial_frame"))
    final_frame = _to_int_frame(casa.get("video", {}).get("final_frame"))
    if initial_frame is not None and final_frame is not None and final_frame >= initial_frame:
        universe &= set(range(initial_frame, final_frame + 1))
    return sorted(universe)


def _pct(value: Any) -> float | None:
    """Convert a fraction to a 2-dp percentage, mapping NaN to ``None``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return float(np.round(f * 100.0, 2))


def _num(value: Any) -> float | int | None:
    """Coerce a metric value, mapping NaN to ``None`` and whole floats to int."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    if float(f).is_integer():
        return int(f)
    return float(np.round(f, 2))


def _collect_track_sets(
    casa: dict[str, Any],
    backend: str | None,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Return ``{label: track_map}`` for every available track set + backend used.

    Labels are ``"groundtruth_tracks"`` for the imported truth and
    ``"<backend>:<source>"`` for each backend source (e.g. ``"sort:yolov5"``).
    """
    tracks_root = casa.get("tracks", {})
    if not isinstance(tracks_root, dict):
        return {}, None

    sets: dict[str, dict[str, Any]] = {}
    gt = tracks_root.get(_GROUNDTRUTH_TRACKS_KEY)
    if isinstance(gt, dict) and gt:
        sets[_GT_LABEL] = gt

    resolved_backend = backend or _resolve_active_tracking_backend(tracks_root)
    if resolved_backend is not None:
        sources = _resolve_sort_track_sources(tracks_root)
        for source_name in sorted(sources.keys(), key=lambda v: (v != "groundtruth", v)):
            source_tracks = sources.get(source_name)
            if isinstance(source_tracks, dict) and source_tracks:
                sets[f"{resolved_backend}:{source_name}"] = source_tracks
    return sets, resolved_backend


def _pair_metrics(
    mm: Any,
    casa: dict[str, Any],
    ref_by_frame: dict[int, Any],
    hyp_by_frame: dict[int, Any],
    ref_id_int: dict[str, int],
    hyp_id_int: dict[str, int],
    threshold: float,
) -> dict[str, Any]:
    """Compute MOT metrics treating ``ref`` as ground truth and ``hyp`` as prediction."""
    eval_frames = _resolve_eval_frames(casa, ref_by_frame, hyp_by_frame)
    if not eval_frames:
        return {"skipped": True, "reason": "no_overlapping_frames", "evaluated_frames": 0}

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in eval_frames:
        ref_items = ref_by_frame.get(frame, {})
        hyp_items = hyp_by_frame.get(frame, {})
        ref_ids = list(ref_items.keys())
        hyp_ids = list(hyp_items.keys())
        if ref_ids and hyp_ids:
            ref_pts = np.asarray([ref_items[i] for i in ref_ids], dtype=float)
            hyp_pts = np.asarray([hyp_items[i] for i in hyp_ids], dtype=float)
            dist = np.linalg.norm(ref_pts[:, None, :] - hyp_pts[None, :, :], axis=2)
            dist[dist >= threshold] = np.nan
        else:
            dist = np.empty((len(ref_ids), len(hyp_ids)), dtype=float)
        acc.update(
            [ref_id_int[i] for i in ref_ids],
            [hyp_id_int[i] for i in hyp_ids],
            dist,
        )

    metric_names = [
        "mota", "motp", "idf1", "idp", "idr",
        "num_switches", "num_false_positives", "num_misses",
        "num_objects", "num_predictions", "num_matches",
        "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_fragmentations", "precision", "recall",
    ]
    summary = mm.metrics.create().compute(acc, metrics=metric_names, name="acc")

    def g(name: str) -> Any:
        try:
            return summary.loc["acc", name]
        except Exception:
            return None

    return {
        "MOTA": _pct(g("mota")),
        "MOTP": _num(g("motp")),
        "IDF1": _pct(g("idf1")),
        "IDP": _pct(g("idp")),
        "IDR": _pct(g("idr")),
        "precision": _pct(g("precision")),
        "recall": _pct(g("recall")),
        "num_switches": _num(g("num_switches")),
        "num_false_positives": _num(g("num_false_positives")),
        "num_misses": _num(g("num_misses")),
        "num_matches": _num(g("num_matches")),
        "num_objects": _num(g("num_objects")),
        "num_predictions": _num(g("num_predictions")),
        "mostly_tracked": _num(g("mostly_tracked")),
        "partially_tracked": _num(g("partially_tracked")),
        "mostly_lost": _num(g("mostly_lost")),
        "num_fragmentations": _num(g("num_fragmentations")),
        "evaluated_frames": len(eval_frames),
        "skipped": False,
    }


def _print_metric_matrix(
    pairs: dict[str, dict[str, dict[str, Any]]],
    labels: list[str],
    metric: str,
    title: str,
) -> None:
    """Print one metric as an aligned reference x hypothesis matrix."""
    label_w = max(len(lbl) for lbl in labels)
    cell_w = max(10, label_w)
    print(title)
    header = " " * label_w + " | " + " | ".join(lbl.rjust(cell_w) for lbl in labels)
    print(header)
    print("-" * len(header))
    for ref in labels:
        cells = []
        for hyp in labels:
            if ref == hyp:
                cells.append("-".rjust(cell_w))
                continue
            value = pairs.get(ref, {}).get(hyp, {}).get(metric)
            cells.append(("n/a" if value is None else f"{value}").rjust(cell_w))
        print(ref.ljust(label_w) + " | " + " | ".join(cells))


def _skip(casa: dict[str, Any], reason: str, threshold: float) -> dict[str, Any]:
    """Store a skipped pairwise result and return ``casa``."""
    print(f"Warning: track assessment skipped ({reason}).")
    assessment = casa.setdefault("assessment", {})
    assessment["tracking"] = {
        "sources": [],
        "reference": None,
        "match_min_distance_pixel": threshold,
        "pairs": {},
        "skipped": True,
        "reason": reason,
    }
    assessment["last_tracking"] = {
        "sources": [],
        "reference": None,
        "match_min_distance_pixel": threshold,
        "skipped": True,
        "reason": reason,
    }
    return casa


def evaluate_tracks(
    casa: dict[str, Any],
    match_min_distance_pixel: float | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Compare every available track set against every other (pairwise MOT metrics).

    Collects all track sets present in the session — the imported ground-truth
    tracks (``casa['tracks']['groundtruth_tracks']``) and every source of the
    active tracking backend (e.g. ``sort:groundtruth``, ``sort:yolov5``) — and
    computes MOT metrics (MOTA, IDF1, ID-switches, FP/FN, fragmentations) for
    each ordered pair, treating the row set as ground truth and the column set
    as prediction. Matching is by per-frame center-to-center distance.

    MOTA is role-dependent (``MOTA(A,B) != MOTA(B,A)``) while IDF1 is symmetric;
    the ``groundtruth_tracks`` row is true tracking *accuracy*, other rows are
    pairwise *agreement*.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary.
        match_min_distance_pixel (float | None, optional):
            Association distance threshold (pixels). If ``None``, uses
            ``casa['meta']['match_min_distance_pixel']`` when available, else ``20``.
        backend (str | None, optional):
            Tracking backend to read predicted sources from. If ``None``, uses
            the active backend.

    Returns:
        dict[str, Any]:
            Input ``casa`` with results under ``casa['assessment']['tracking']``
            (``{sources, reference, pairs: {ref: {hyp: metrics}}}``) plus
            metadata in ``casa['assessment']['last_tracking']`` and
            ``casa['meta']['last_assessment']``.

    Notes:
        Uses the optional ``motmetrics`` dependency (installed on demand).
    """
    casa = _ensure_casa(casa)

    threshold = (
        float(match_min_distance_pixel)
        if match_min_distance_pixel is not None
        else float(casa.get("meta", {}).get("match_min_distance_pixel", 20.0))
    )
    if threshold <= 0:
        raise ValueError("`match_min_distance_pixel` must be > 0.")

    track_sets, _ = _collect_track_sets(casa, backend)
    if len(track_sets) < 2:
        return _skip(casa, "need_at_least_two_track_sets", threshold)

    mm = _ensure_import(
        "motmetrics",
        pip_name="motmetrics",
        prompt_install=True,
        required=True,
    )

    labels = list(track_sets.keys())
    by_frame = {label: _tracks_by_frame(tracks) for label, tracks in track_sets.items()}
    id_int = {label: _id_int_map(tracks) for label, tracks in track_sets.items()}

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for ref in labels:
        for hyp in labels:
            if ref == hyp:
                continue
            pairs.setdefault(ref, {})[hyp] = _pair_metrics(
                mm,
                casa,
                by_frame[ref],
                by_frame[hyp],
                id_int[ref],
                id_int[hyp],
                threshold,
            )

    reference = _GT_LABEL if _GT_LABEL in track_sets else None
    track_counts = {label: int(len(tracks)) for label, tracks in track_sets.items()}

    assessment = casa.setdefault("assessment", {})
    assessment["tracking"] = {
        "sources": labels,
        "reference": reference,
        "track_counts": track_counts,
        "match_min_distance_pixel": threshold,
        "pairs": pairs,
        "skipped": False,
    }
    assessment["last_tracking"] = {
        "sources": labels,
        "reference": reference,
        "match_min_distance_pixel": threshold,
        "num_pairs": sum(len(v) for v in pairs.values()),
        "skipped": False,
    }
    casa["meta"]["last_assessment"] = {
        "backend": "tracking",
        "detection_method": reference or (labels[0] if labels else None),
        "sources": labels,
        "reference": reference,
        "match_min_distance_pixel": threshold,
    }

    # --- pretty print ---
    ref_note = f"; reference (truth) = {reference}" if reference else " (no imported truth present)"
    print(
        f"Track assessment - pairwise ({len(labels)} track sets, "
        f"threshold={threshold:.0f}px){ref_note}"
    )
    for label in labels:
        print(f"  - {label}: {track_counts[label]} tracks")
    print()
    _print_metric_matrix(
        pairs, labels, "MOTA",
        "MOTA (%)  [row = ground-truth role, col = prediction role]",
    )
    print()
    _print_metric_matrix(
        pairs, labels, "IDF1",
        "IDF1 (%)  [symmetric]",
    )
    if reference:
        print()
        print(f"Accuracy vs truth ({reference}):")
        for hyp, m in pairs.get(reference, {}).items():
            if m.get("skipped"):
                print(f"  - {hyp}: skipped ({m.get('reason')})")
                continue
            print(
                f"  - {hyp}: MOTA={m.get('MOTA')}%, IDF1={m.get('IDF1')}%, "
                f"id_switches={m.get('num_switches')}, "
                f"fp={m.get('num_false_positives')}, fn={m.get('num_misses')}, "
                f"frags={m.get('num_fragmentations')}"
            )
    return casa
