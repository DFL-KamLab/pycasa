# pycasa Documentation

`pycasa` is a Python toolkit for computer-assisted semen analysis workflows, from video loading to preprocessing, detection, tracking, motility analysis, assessment, and visualization.

Use this documentation to:

- install dependencies by feature,
- run end-to-end examples,
- understand the `Casa` session model and namespace APIs,
- debug common setup/runtime issues.

## Documentation Roadmap

- Start with [Setup & Requirements](getting-started/setup.md) for installation and dependency choices.
- Continue with [Quickstart](getting-started/quickstart.md) for the shortest working pipeline.
- Move to [Examples](examples/default-data-otsu-yolo.md) for complete workflow scenarios.
- Use the [API Guide](api/casa-session.md) for namespace-level method references.

## First Pipeline

```python
import pycasa as pc
self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolov5()
```

On first run, pycasa may download default dataset assets and YOLO weight files.
