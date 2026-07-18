from decimal import Decimal
from decimal import InvalidOperation
from decimal import ROUND_DOWN
from typing import Any

import numpy as np

from ._tracking_helpers import _GROUNDTRUTH_TRACKS_KEY
from ._tracking_helpers import _resolve_active_tracking_backend
from ._tracking_helpers import _resolve_sort_track_sources


def _coerce_classification_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract compact, stable classification fields for info display."""
    return {
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "F1": metrics.get("F1"),
        "evaluated_frames": metrics.get("evaluated_frames"),
    }


def _average_track_length(track_map: dict[str, Any]) -> float | None:
    """Return mean per-track frame count for ``track_id -> frame -> point`` maps."""
    if not isinstance(track_map, dict) or not track_map:
        return None
    lengths = [len(points) for points in track_map.values() if isinstance(points, dict)]
    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def _truncate_two_decimals(value: Any) -> str | None:
    """Return a string truncated (not rounded) to two decimals."""
    if value is None:
        return None
    try:
        truncated = Decimal(str(value)).quantize(Decimal("0.00"), rounding=ROUND_DOWN)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return format(truncated, "f")


def _is_nested_source_motility_map(candidate: dict[str, Any]) -> bool:
    """Return ``True`` for shape ``source -> track_id -> metric_dict``."""
    for source_value in candidate.values():
        if not isinstance(source_value, dict):
            continue
        if any(isinstance(track_value, dict) for track_value in source_value.values()):
            return True
    return False


def _build_tracking_summary(
    tracks: dict[str, Any],
    meta_last_tracking: dict[str, Any],
) -> dict[str, Any]:
    """Build per-source tracking summary for the active tracking backend."""
    if not isinstance(tracks, dict):
        tracks = {}
    if not isinstance(meta_last_tracking, dict):
        meta_last_tracking = {}

    active_backend = _resolve_active_tracking_backend(tracks)
    sources = _resolve_sort_track_sources(tracks)
    rows: list[dict[str, Any]] = []
    for source_name in sorted(sources.keys(), key=lambda value: (value != "groundtruth", value)):
        track_map = sources[source_name]
        rows.append(
            {
                "source": source_name,
                "track_count": int(len(track_map)),
                "average_track_length": _average_track_length(track_map),
            }
        )

    primary_source = meta_last_tracking.get("detection_method")
    if primary_source is not None:
        primary_source = str(primary_source)
    if primary_source is None and rows:
        if any(row["source"] == "groundtruth" for row in rows):
            primary_source = "groundtruth"
        else:
            primary_source = str(rows[0]["source"])

    backend_label = active_backend if (active_backend and rows) else None

    imported_gt = tracks.get(_GROUNDTRUTH_TRACKS_KEY)
    groundtruth_tracks_summary: dict[str, Any] | None = None
    if isinstance(imported_gt, dict) and imported_gt:
        groundtruth_tracks_summary = {
            "track_count": int(len(imported_gt)),
            "average_track_length": _average_track_length(imported_gt),
        }

    return {
        "backend": backend_label,
        "detection_method": primary_source,
        "sources": rows,
        "groundtruth_tracks": groundtruth_tracks_summary,
    }


def _build_motility_summary(
    motility: dict[str, Any],
    meta_last_motility: dict[str, Any],
    meta_last_tracking: dict[str, Any],
) -> dict[str, Any]:
    """Build per-source motility summary for standard parameters."""
    if not isinstance(motility, dict):
        motility = {}
    if not isinstance(meta_last_motility, dict):
        meta_last_motility = {}
    if not isinstance(meta_last_tracking, dict):
        meta_last_tracking = {}

    tracking_backend = str(meta_last_motility.get("tracking_backend") or "sort")
    primary_source = (
        meta_last_motility.get("detection_method")
        or meta_last_tracking.get("detection_method")
    )
    if primary_source is not None:
        primary_source = str(primary_source)

    method_root = motility.get("standard_motility_parameters")
    if not isinstance(method_root, dict):
        return {"available": False, "tracking_backend": tracking_backend, "sources": []}

    metric_keys = ("VCL", "VSL", "VAP", "LIN", "ALH", "WOB", "STR", "MAD")
    source_maps: dict[str, dict[str, Any]] = {}
    if not _is_nested_source_motility_map(method_root):
        source_maps["groundtruth"] = method_root
    else:
        for source_name, source_data in method_root.items():
            if isinstance(source_data, dict):
                source_maps[str(source_name)] = source_data

    if not source_maps:
        return {"available": False, "tracking_backend": tracking_backend, "sources": []}

    rows: list[dict[str, Any]] = []
    for source_name in sorted(source_maps.keys(), key=lambda value: (value != "groundtruth", value)):
        method_data = source_maps[source_name]
        metric_values: dict[str, list[float]] = {key: [] for key in metric_keys}
        window_counts: list[int] = []

        for track_metrics in method_data.values():
            if not isinstance(track_metrics, dict):
                continue

            vcl_values = track_metrics.get("VCL")
            if isinstance(vcl_values, list):
                window_counts.append(len(vcl_values))
            else:
                window_counts.append(0)

            for metric_key in metric_keys:
                values = track_metrics.get(metric_key)
                if not isinstance(values, list):
                    continue
                for value in values:
                    try:
                        metric_values[metric_key].append(float(value))
                    except (TypeError, ValueError):
                        continue

        row: dict[str, Any] = {
            "source": source_name,
            "track_count": int(len(method_data)),
            "average_windows_per_track": (
                float(sum(window_counts) / len(window_counts))
                if window_counts
                else None
            ),
        }
        for metric_key in metric_keys:
            values = metric_values[metric_key]
            row[f"mean_{metric_key}"] = float(sum(values) / len(values)) if values else None
        rows.append(row)

    if primary_source is None and rows:
        if any(row["source"] == "groundtruth" for row in rows):
            primary_source = "groundtruth"
        else:
            primary_source = str(rows[0]["source"])

    return {
        "available": True,
        "tracking_backend": tracking_backend,
        "detection_method": primary_source,
        "sources": rows,
    }


def _build_casa_info(casa: dict[str, Any]) -> dict[str, Any]:
    """Build a structured, non-mutating session summary for ``Casa.info()``."""
    meta = casa.get("meta", {})
    video = casa.get("video", {})
    detections = casa.get("detections", {})
    tracks = casa.get("tracks", {})
    motility = casa.get("motility", {})
    assessment = casa.get("assessment", {})

    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(video, dict):
        video = {}
    if not isinstance(detections, dict):
        detections = {}
    if not isinstance(tracks, dict):
        tracks = {}
    if not isinstance(motility, dict):
        motility = {}
    if not isinstance(assessment, dict):
        assessment = {}

    session = {
        "video_path": meta.get("video_path") or video.get("path"),
        "groundtruth_detections_path": detections.get("groundtruth_detections_path"),
        "groundtruth_tracks_path": meta.get("groundtruth_tracks_path"),
    }

    video_info = {
        "initial_frame": video.get("initial_frame"),
        "final_frame": video.get("final_frame"),
        "number_frame_used": video.get("number_frame_used"),
        "total_number_frame": meta.get("total_number_frame"),
        "sampling_rate": meta.get("sampling_rate"),
        "duration_sec": meta.get("duration_sec"),
        "total_duration_sec": meta.get("total_duration_sec"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "um_per_px": meta.get("um_per_px"),
        "magnification": meta.get("magnification"),
    }

    video_keys = {
        "original": "original_video",
        "grayscale": "grayscale_video",
        "normalized": "normalized_video",
        "binarized": "binary_video",
        "moving_cells": "binarized_moving_cells_video",
    }
    videos: dict[str, dict[str, Any]] = {}
    for name, video_key in video_keys.items():
        video_data = video.get(video_key)
        if isinstance(video_data, np.ndarray):
            videos[name] = {
                "present": True,
                "shape": tuple(int(dim) for dim in video_data.shape),
                "dtype": str(video_data.dtype),
            }
        else:
            videos[name] = {"present": False, "shape": None, "dtype": None}

    detection_methods = sorted(
        key
        for key, value in detections.items()
        if isinstance(value, dict)
    )
    tracking_methods = sorted(
        key for key, value in tracks.items() if isinstance(value, dict)
    )
    motility_methods = sorted(
        key for key, value in motility.items() if isinstance(value, dict)
    )
    assessment_methods = (
        ["classification"]
        if isinstance(assessment.get("classification"), dict)
        else []
    )
    methods = {
        "detection": detection_methods,
        "tracking": tracking_methods,
        "motility": motility_methods,
        "assessment": assessment_methods,
    }

    meta_last_tracking = meta.get("last_tracking")
    if not isinstance(meta_last_tracking, dict):
        meta_last_tracking = {}

    meta_last_motility = meta.get("last_motility")
    if not isinstance(meta_last_motility, dict):
        meta_last_motility = {}

    tracking_info = _build_tracking_summary(tracks, meta_last_tracking)
    motility_info = _build_motility_summary(motility, meta_last_motility, meta_last_tracking)

    classification_data = assessment.get("classification")
    if not isinstance(classification_data, dict):
        classification_data = {}
    last_classification = assessment.get("last_classification")
    if not isinstance(last_classification, dict):
        last_classification = {}
    assessment_info = {
        "detection_method": last_classification.get("detection_method"),
        "match_min_distance_pixel": last_classification.get("match_min_distance_pixel"),
        "frame_summary": last_classification.get("frame_summary"),
        "classification": (
            _coerce_classification_metrics(classification_data)
            if classification_data
            else {}
        ),
    }

    last_tracking_summary = None
    tracking_backend = meta_last_tracking.get("backend")
    tracking_method = meta_last_tracking.get("detection_method")
    if tracking_method is None:
        processed = meta_last_tracking.get("sources_processed")
        if isinstance(processed, list) and processed:
            tracking_method = ",".join(str(item) for item in processed if isinstance(item, str))
    if tracking_backend and tracking_method:
        last_tracking_summary = f"{tracking_backend}:{tracking_method}"
    elif tracking_backend:
        last_tracking_summary = str(tracking_backend)

    last_motility_summary = None
    motility_backend = (
        meta_last_motility.get("tracking_backend")
        or meta_last_motility.get("backend")
    )
    motility_method = meta_last_motility.get("detection_method")
    if motility_method is None:
        processed = meta_last_motility.get("sources_processed")
        if isinstance(processed, list) and processed:
            motility_method = ",".join(str(item) for item in processed if isinstance(item, str))
    if motility_backend and motility_method:
        last_motility_summary = f"{motility_backend}:{motility_method}"
    elif motility_backend:
        last_motility_summary = str(motility_backend)

    last_assessment_summary = None
    meta_last_assessment = meta.get("last_assessment")
    if isinstance(meta_last_assessment, dict):
        backend = meta_last_assessment.get("backend")
        detection_method = meta_last_assessment.get("detection_method")
        if backend and detection_method:
            last_assessment_summary = f"{backend}:{detection_method}"
        elif backend:
            last_assessment_summary = str(backend)
    elif meta_last_assessment is not None:
        last_assessment_summary = str(meta_last_assessment)

    last_performed_operations = {
        "last_preprocessing": meta.get("last_preprocessing"),
        "last_tracking": last_tracking_summary,
        "last_motility": last_motility_summary,
        "last_assessment": last_assessment_summary,
    }

    return {
        "session": session,
        "video": video_info,
        "videos": videos,
        "methods": methods,
        "tracking": tracking_info,
        "motility": motility_info,
        "assessment": assessment_info,
        "last_performed_operations": last_performed_operations,
    }


def _print_casa_info(info: dict[str, Any]) -> None:
    """Print a stable sectioned summary of ``Casa.info()`` output."""
    print("Casa Info")
    section_order = (
        "session",
        "video",
        "videos",
        "methods",
        "tracking",
        "motility",
        "assessment",
        "last_performed_operations",
    )

    def _duration_with_unit(value: Any) -> str:
        truncated = _truncate_two_decimals(value)
        if truncated is None:
            return "None"
        return f"{truncated} sec"

    section_titles = {
        "video": "video specifications",
        "videos": "videos",
        "last_performed_operations": "last performed operations",
    }

    for section in section_order:
        print("")
        print(f"[{section_titles.get(section, section)}]")
        section_data = info.get(section, {})
        if not isinstance(section_data, dict):
            print(f"- value: {section_data}")
            continue

        if section == "video":
            total_number_frame = section_data.get("total_number_frame")
            total_number_frame_text = (
                f"{total_number_frame} frames"
                if total_number_frame is not None
                else "None"
            )
            width = section_data.get("width")
            width_text = f"{width} px" if width is not None else "None"
            height = section_data.get("height")
            height_text = f"{height} px" if height is not None else "None"
            um_per_px = section_data.get("um_per_px")
            um_per_px_text = f"{um_per_px} um/px" if um_per_px is not None else "None"
            sampling_rate = section_data.get("sampling_rate")
            sampling_rate_text = (
                f"{sampling_rate} fps"
                if sampling_rate is not None
                else "None"
            )
            number_frame_used = section_data.get("number_frame_used")
            number_frame_used_text = (
                f"{number_frame_used} frames"
                if number_frame_used is not None
                else "None"
            )

            print("- total number of frames: " f"{total_number_frame_text}")
            print(
                "- total video duration: "
                f"{_duration_with_unit(section_data.get('total_duration_sec'))}"
            )
            print(f"- width: {width_text}")
            print(f"- height: {height_text}")
            print(f"- um per px: {um_per_px_text}")
            print(f"- magnification: {section_data.get('magnification')}")
            print(f"- sampling rate: {sampling_rate_text}")
            print("")
            print("- number of processed frames: " f"{number_frame_used_text}")
            print(
                "- processed video duration: "
                f"{_duration_with_unit(section_data.get('duration_sec'))}"
            )
            print(f"- initial frame: {section_data.get('initial_frame')}")
            print(f"- final frame: {section_data.get('final_frame')}")
            continue

        if section == "videos":
            for key, value in section_data.items():
                if isinstance(value, dict):
                    print(
                        f"- {key}: present={value.get('present')} "
                        f"shape={value.get('shape')} dtype={value.get('dtype')}"
                    )
                else:
                    print(f"- {key}: {value}")
            continue

        if section == "tracking":
            backend = section_data.get("backend")
            groundtruth_tracks = section_data.get("groundtruth_tracks")
            printed_any = False

            if backend:
                sources = section_data.get("sources")
                if isinstance(sources, list) and sources:
                    for row in sources:
                        if not isinstance(row, dict):
                            continue
                        print(f"- {backend}: {row.get('source')}")
                        print(
                            "- "
                            f"tracks={row.get('track_count')}, "
                            "average track length="
                            f"{_truncate_two_decimals(row.get('average_track_length'))}"
                        )
                        printed_any = True
                else:
                    print(f"- {backend}: none")
                    printed_any = True

            if isinstance(groundtruth_tracks, dict):
                print("- groundtruth_tracks: imported")
                print(
                    "- "
                    f"tracks={groundtruth_tracks.get('track_count')}, "
                    "average track length="
                    f"{_truncate_two_decimals(groundtruth_tracks.get('average_track_length'))}"
                )
                printed_any = True

            if not printed_any:
                print("- tracking: none")
            continue

        if section == "motility":
            print("- Standard Motility Parameters")
            if not bool(section_data.get("available")):
                print("- none")
                continue
            sources = section_data.get("sources")
            if not isinstance(sources, list) or not sources:
                print("- none")
                continue
            for row in sources:
                if not isinstance(row, dict):
                    continue
                print("")
                if row.get("source") == _GROUNDTRUTH_TRACKS_KEY:
                    print("- groundtruth_tracks (imported)")
                else:
                    print(
                        "- "
                        f"{section_data.get('tracking_backend')}: "
                        f"{row.get('source')}"
                    )
                print(
                    "- "
                    f"tracks={row.get('track_count')}, "
                    "average windows per track="
                    f"{_truncate_two_decimals(row.get('average_windows_per_track'))}"
                )
                print(
                    "- "
                    f"VCL={_truncate_two_decimals(row.get('mean_VCL'))}, "
                    f"VSL={_truncate_two_decimals(row.get('mean_VSL'))}"
                )
                print(
                    "- "
                    f"VAP={_truncate_two_decimals(row.get('mean_VAP'))}, "
                    f"LIN={_truncate_two_decimals(row.get('mean_LIN'))}"
                )
                print(
                    "- "
                    f"ALH={_truncate_two_decimals(row.get('mean_ALH'))}, "
                    f"WOB={_truncate_two_decimals(row.get('mean_WOB'))}"
                )
                print(
                    "- "
                    f"STR={_truncate_two_decimals(row.get('mean_STR'))}, "
                    f"MAD={_truncate_two_decimals(row.get('mean_MAD'))}"
                )
            continue

        if section == "assessment":
            detection_method = section_data.get("detection_method") or "none"
            metrics = section_data.get("classification")
            if not isinstance(metrics, dict) or not metrics:
                print("- classification: none")
                continue

            print(f"- classification: {detection_method}")
            print(
                "- "
                f"tp={metrics.get('tp')}, "
                f"fp={metrics.get('fp')}"
            )
            print(
                "- "
                f"fn={metrics.get('fn')}, "
                f"precision={metrics.get('precision')}"
            )
            print(
                "- "
                f"recall={metrics.get('recall')}, "
                f"F1={metrics.get('F1')}"
            )
            print(f"- evaluated frames={metrics.get('evaluated_frames')}")
            frame_summary = section_data.get("frame_summary")
            if frame_summary:
                print(f"- {frame_summary}")
            continue

        if section == "last_performed_operations":
            print(f"- last_preprocessing: {section_data.get('last_preprocessing')}")
            print(f"- last_tracking: {section_data.get('last_tracking')}")
            print(f"- last_motility: {section_data.get('last_motility')}")
            print(f"- last_assessment: {section_data.get('last_assessment')}")
            continue

        for key, value in section_data.items():
            print(f"- {key}: {value}")
