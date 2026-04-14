# Package Architecture

This section explains how pycasa is organized and how data moves through the package.

## High-Level Layout

```text
pycasa/
  __init__.py
  casa/
    casa.py
    io/
    preprocessing/
    detection/
    tracking/
    motility/
    assessment/
    visualization/
  io/
  preprocessing/
  detection/
  tracking/
  motility/
  assessment/
  visualization/
  _core/
  utils/
```

## Responsibility by Layer

- `pycasa.casa.*`:
  fluent session wrappers users call directly (`self.detection.yolov5()`, etc.).
- `pycasa.<namespace>`:
  implementation functions for each domain area.
- `pycasa._core`:
  session contract and validation.
- `pycasa.utils`:
  shared helpers for dependencies, state updates, and cross-cutting behavior.

## Helper and Structure Rules

- Public API surface:
  call methods through `pc.io.*` and `self.<namespace>.*`.
- Internal helpers:
  underscore-prefixed modules/functions (for example `_core`, helper internals in `utils`) are not stable user APIs.
- Separation of concerns:
  wrappers in `pycasa.casa.*` manage fluent orchestration, while implementation logic stays in namespace modules (`pycasa/io`, `pycasa/detection`, etc.).
- Traceability rule:
  each stage updates `meta["last_*"]` fields so the session records which pipeline step ran last and with what settings.

## Data Flow

1. I/O creates a validated session dictionary.
2. Preprocessing enriches `video` state.
3. Detection stores active predicted detections.
4. Tracking converts detections into tracks.
5. Motility/Assessment compute analysis outputs.
6. Visualization renders inspection and reporting views.
