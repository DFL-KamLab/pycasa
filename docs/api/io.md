# API: I/O

Purpose:
I/O is the entry point into a pycasa workflow. This section covers the public methods that create/populate `Casa` sessions from either your own video files or the packaged default dataset cache.

## Public Methods In This Section

- `pc.io.load_video(...)`
- `pc.io.load_default_data(...)`
- `self.io.load_video(...)`
- `self.io.load_default_data(...)`

## `pc.io.load_video(...)`

Load a user-provided video and optional groundtruth folder.

Key inputs:

- `video_path`
- `groundtruth_path` (optional)
- `initial_frame`, `final_frame`
- `sampling_rate`, `um_per_px`, `magnification`
- `show_progress`, `verbose`

Example:

```python
import pycasa as pc

self = pc.io.load_video(
    video_path="path/to/video.avi",
    groundtruth_path="path/to/gt",
    initial_frame=0,
    final_frame=200,
)
```

## `pc.io.load_default_data(...)`

Load a default bundled session with local caching.

Key inputs:

- `path` (cache root override)
- `download`, `force_download`
- frame range and metadata fields

Related environment variable:

- `PYCASA_DATA`

Example:

```python
import pycasa as pc
self = pc.io.load_default_data(download=True)
```

## Returns

Both methods return a `Casa` object with populated `meta` and `video`, plus optional groundtruth detections.
