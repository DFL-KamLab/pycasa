# Example: Load Custom Video

Use this workflow when you have your own input file and optional groundtruth annotations.

## Install

```bash
pip install "git+https://github.com/DFL-KamLab/pycasa.git"
```

## Script

```python
import pycasa as pc

self = pc.io.load_video(
    video_path="path/to/video.avi",
    groundtruth_path="path/to/groundtruth_folder",  # optional
    initial_frame=0,
    final_frame=300,
    sampling_rate=50.0,  # optional override
    um_per_px=0.25,      # optional calibration
    magnification="20x", # optional metadata tag
)

self.preprocessing.grayscale(overwrite=False)
self.info()
```

## When To Use This

- You have local AVI/MP4-style microscopy videos.
- You want custom frame ranges.
- You want to inject calibration values during load.

## Common Tips

- Use `final_frame=None` to read until the end of the video.
- Keep `overwrite=False` during preprocessing if you want to preserve original frames.
