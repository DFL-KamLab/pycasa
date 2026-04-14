# pycasa

`pycasa` is a Python toolkit for step-by-step computer-assisted semen analysis workflows, including video loading, preprocessing, detection, tracking, and downstream analysis.

## Install

Quick install from GitHub:

```bash
pip install "git+https://github.com/DFL-KamLab/pycasa.git"
```

Optional: install extras for the full default-data + YOLO example:

```bash
pip install "pycasa[io,yolo] @ git+https://github.com/DFL-KamLab/pycasa.git"
```


## Small Example

```python
import pycasa as pc
self = pc.io.load_default_data()
self.preprocessing.binarization.otsu()
self.detection.yolov5()
```

First run may download default dataset assets and YOLO weight files.
