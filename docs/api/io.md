# API: I/O

I/O is the entry point into a pycasa workflow. This section covers the public methods that create and populate `Casa` sessions from either your own video files or the packaged default dataset cache.

## Public Methods In This Section

- `pc.io.load_video(...)`
- `pc.io.load_default_data(...)`
- `self.io.load_video(...)` *(same as above, via session namespace)*
- `self.io.load_default_data(...)` *(same as above, via session namespace)*

---

## `pc.io.load_video(...)`

Load a time-lapse video file and return a fluent `Casa` object.

Supported file extensions: `.avi`, `.mp4`, `.mov`, `.mkv`, `.flv`, `.wmv`

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `str` | *(required)* | Path to the video file. Must exist and have a supported extension. |
| `groundtruth_path` | `str \| None` | `None` | Optional path to a directory containing per-frame groundtruth text files (one file per frame, named `frame-<index>.txt`). When provided, detections are loaded under `casa["detections"]["groundtruth"]`. |
| `initial_frame` | `int` | `0` | First frame index to read (0-based). |
| `final_frame` | `int \| None` | `None` | Last frame index to read (0-based, inclusive). `None` reads to the end of the video. Clamped to the last available frame if out of range. |
| `sampling_rate` | `float \| None` | `None` | Explicit frame-rate (FPS) override. When `None`, FPS is read from the video file's metadata. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while reading frames (requires `tqdm`). |
| `verbose` | `bool` | `True` | Print concise start/end summaries for the loading step. Does not suppress warnings. |
| `um_per_px` | `float \| None` | `None` | Microns-per-pixel calibration stored in metadata. Required for motility unit conversion. Omitting this value emits a `UserWarning`. |
| `magnification` | `str \| None` | `None` | Free-text magnification descriptor stored in `casa["meta"]["magnification"]` (e.g., `"10x"`). |

**Returns**

`Casa` — a fluent session object with `meta` and `video` populated. If `groundtruth_path` is provided, `casa["detections"]["groundtruth"]` is also populated.

**Raises**

- `FileNotFoundError` — if `video_path` does not exist.
- `ValueError` — if the file extension is unsupported, or if the requested frame range is invalid.
- `RuntimeError` — if OpenCV cannot open the file or no frames can be read.
- `ImportError` — if `opencv-python` is not installed.

**Notes**

- Loaded frames are stored in BGR channel order (OpenCV convention).
- `number_frame_used` reflects actually-read frames, which may be less than requested if early read termination occurs.
- If `um_per_px` is not provided, motility parameter computation will fail later. Use `self.set_um_per_px(value)` to set it after loading.

**Example**

```python
import pycasa as pc

session = pc.io.load_video(
    video_path="path/to/video.avi",
    groundtruth_path="path/to/gt",
    initial_frame=0,
    final_frame=200,
    um_per_px=0.24,
)
session.info()
```

---

## `pc.io.load_default_data(...)`

Load the default bundled HC004 session from HuggingFace with MNE-style local caching.

The cache root is resolved in this order:
1. Explicit `path` argument
2. `PYCASA_DATA` environment variable
3. `~/.pycasa_data` (home directory fallback)

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| None` | `None` | Root folder to cache/load default data. If `None`, resolution falls through to `PYCASA_DATA` or `~/.pycasa_data`. |
| `download` | `bool` | `True` | If `True`, missing required files trigger a selective download from HuggingFace. |
| `force_download` | `bool` | `False` | If `True`, always re-download the required subset even when cached files exist. |
| `initial_frame` | `int` | `0` | First frame index to read (0-based). |
| `final_frame` | `int \| None` | `100` | Last frame index to read (0-based, inclusive). Defaults to frame 100. |
| `sampling_rate` | `float \| None` | `None` | Optional FPS override forwarded to `load_video`. |
| `um_per_px` | `float \| None` | `None` | Microns-per-pixel calibration. When `None`, the HSTLI dataset value `0.24` is set automatically (and a one-line confirmation is printed when `verbose=True`). Pass an explicit value to override. |
| `magnification` | `str \| None` | `None` | Optional magnification metadata value. |
| `verbose` | `bool` | `True` | Print concise cache/download resolution summaries. |

**Returns**

`Casa` — fluent session object loaded from the default video and its groundtruth folder.

**Raises**

- `FileNotFoundError` — if required files are missing and `download=False`, or if files remain missing after download.
- `ImportError` — if download is needed and `huggingface_hub` is unavailable.

**Notes**

Only this subset of the dataset is downloaded (not the full repository):

- `sys-casa_sub-HC004_ses-01_run-005_video.avi`
- `sys-casa_sub-HC004_ses-01_run-005_video.json`
- `sys-casa_sub-HC004_ses-01_run-005_gt/` (groundtruth directory)
- `README.md` (session folder)

Pixel calibration (`um_per_px`) is set automatically to `0.24` — the value matching the HSTLI HC004 acquisition — when not provided. Downstream motility computation will therefore work without a manual `self.set_um_per_px(...)` call. Pass an explicit `um_per_px` argument to override.

**Example**

```python
import pycasa as pc

session = pc.io.load_default_data(download=True)
session.info()
```

Override the cache location:

```python
session = pc.io.load_default_data(path="/data/pycasa_cache", initial_frame=0, final_frame=50)
```
