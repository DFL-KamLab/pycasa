# API: Preprocessing

Preprocessing prepares loaded frames for downstream detection and tracking by generating stable visual representations: grayscale conversions, normalized intensity variants, and binary masks.

All preprocessing methods:
- Write derived arrays to `casa["video"]`.
- Update `casa["meta"]["last_preprocessing"]`.
- Return the same `Casa` instance for fluent chaining.

## Public Methods In This Section

- `self.preprocessing.grayscale(...)`
- `self.preprocessing.normalization.min_max(...)`
- `self.preprocessing.normalization.z_score(...)`
- `self.preprocessing.normalization.hist_equal(...)`
- `self.preprocessing.normalization.clahe(...)`
- `self.preprocessing.normalization.log(...)`
- `self.preprocessing.normalization.median(...)`
- `self.preprocessing.binarization.otsu(...)`
- `self.preprocessing.binarization.adaptive_mean(...)`
- `self.preprocessing.binarization.adaptive_gaussian(...)`
- `self.preprocessing.binarization.sauvola(...)`
- `self.preprocessing.binarization.niblack(...)`
- `self.preprocessing.binarization.urbano(...)`

---

## Grayscale

### `self.preprocessing.grayscale(overwrite=False, show_progress=True, verbose=True)`

Convert loaded video frames to grayscale.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overwrite` | `bool` | `False` | If `True`, replaces `casa["video"]["original_video"]` with the grayscale result in addition to writing `grayscale_video`. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. Does not suppress warnings. |

**Output**

- Writes `casa["video"]["grayscale_video"]` (uint8 single-channel array).
- Sets `casa["meta"]["last_preprocessing"]`.

**Returns**

`Casa` — the same session instance.

**Example**

```python
self.preprocessing.grayscale()
```

---

## Normalization Methods

Normalization methods produce a floating-point or uint8 intensity-scaled video stored in `casa["video"]["normalized_video"]`. The method used is recorded in `casa["video"]["normalized_type"]`. Only one normalization result is kept at a time; running a second method overwrites the previous one.

### Common Parameters (all normalization methods)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overwrite` | `bool` | `False` | If `True`, replaces `original_video` with the normalized result in addition to writing `normalized_video`. |
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. |

---

### `self.preprocessing.normalization.min_max(...)`

Per-frame min-max scaling to the `[0, 255]` range.

Each frame is independently scaled so that its minimum becomes 0 and its maximum becomes 255.

- `normalized_type` value: `"min-max"`

**Example**

```python
self.preprocessing.normalization.min_max()
```

---

### `self.preprocessing.normalization.z_score(...)`

Per-frame z-score standardization: `(frame - mean) / std`.

Output is `float32` and is not clamped to `[0, 255]`.

- `normalized_type` value: `"z-score"`

**Example**

```python
self.preprocessing.normalization.z_score()
```

---

### `self.preprocessing.normalization.hist_equal(...)`

Global histogram equalization applied per frame.

For color frames, equalization is applied channel-wise.

- `normalized_type` value: `"hist-equal"`

**Example**

```python
self.preprocessing.normalization.hist_equal()
```

---

### `self.preprocessing.normalization.clahe(...)`

CLAHE (Contrast-Limited Adaptive Histogram Equalization) applied per frame and per channel.

**Additional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_limit` | `float` | `2.0` | Threshold for contrast limiting. Higher values allow more contrast enhancement. |
| `tile_grid_size` | `tuple[int, int]` | `(8, 8)` | Size of the grid of tiles used for local histogram equalization. |

- `normalized_type` value: `"clahe"`

**Example**

```python
self.preprocessing.normalization.clahe(clip_limit=3.0, tile_grid_size=(8, 8))
```

---

### `self.preprocessing.normalization.log(...)`

Log normalization: applies `np.log1p` to pixel values, then min-max scales the result to `[0, 255]`.

Useful for compressing high-intensity outliers.

- `normalized_type` value: `"log"`

**Example**

```python
self.preprocessing.normalization.log()
```

---

### `self.preprocessing.normalization.median(...)`

Median-centering normalization: subtracts the per-frame median, then min-max scales to `[0, 255]`.

- `normalized_type` value: `"median"`

**Example**

```python
self.preprocessing.normalization.median()
```

---

## Binarization Methods

Binarization methods produce a uint8 binary mask stored in `casa["video"]["binary_video"]`, where foreground pixels are `255` and background pixels are `0`. The method used is recorded in `casa["video"]["binary_type"]`. Only one binarization result is kept at a time.

### Common Parameters (all binarization methods)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_progress` | `bool` | `True` | Show the pycasa progress bar while processing frames. |
| `verbose` | `bool` | `True` | Print start/end summaries. |

---

### `self.preprocessing.binarization.otsu(...)`

Per-frame Otsu thresholding with automatic threshold calculation.

Otsu's method finds the threshold that minimizes intra-class intensity variance. Well-suited for bimodal intensity histograms.

- `binary_type` value: `"otsu"`

**Example**

```python
self.preprocessing.binarization.otsu()
```

---

### `self.preprocessing.binarization.adaptive_mean(...)`

Adaptive mean thresholding using OpenCV's `cv2.ADAPTIVE_THRESH_MEAN_C`.

The threshold for each pixel is the mean of its neighborhood minus a constant.

**Additional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | `int` | `11` | Size of the neighborhood area. Must be an odd number ≥ 3. |
| `c` | `float` | `2.0` | Constant subtracted from the neighborhood mean to compute the threshold. |

- `binary_type` value: `"adaptive-mean"`

**Example**

```python
self.preprocessing.binarization.adaptive_mean(block_size=15, c=3.0)
```

---

### `self.preprocessing.binarization.adaptive_gaussian(...)`

Adaptive Gaussian thresholding using OpenCV's `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`.

The threshold for each pixel is a Gaussian-weighted sum of its neighborhood minus a constant. Typically produces smoother results than adaptive mean.

**Additional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | `int` | `11` | Size of the neighborhood area. Must be an odd number ≥ 3. |
| `c` | `float` | `2.0` | Constant subtracted from the Gaussian-weighted mean. |

- `binary_type` value: `"adaptive-gaussian"`

**Example**

```python
self.preprocessing.binarization.adaptive_gaussian(block_size=11, c=2.0)
```

---

### `self.preprocessing.binarization.sauvola(...)`

Sauvola local thresholding.

The threshold at each pixel is computed as `mean * (1 + k * (std / r - 1))`, which adapts to local contrast. Well-suited for uneven illumination.

**Additional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `25` | Size of the local neighborhood window. Must be ≥ 3. |
| `k` | `float` | `0.2` | Contrast sensitivity factor. Controls how much local standard deviation affects the threshold. |
| `r` | `float` | `128.0` | Dynamic range of the standard deviation. Typically set to half the maximum possible pixel value. |

- `binary_type` value: `"sauvola"`

**Example**

```python
self.preprocessing.binarization.sauvola(window_size=25, k=0.2, r=128.0)
```

---

### `self.preprocessing.binarization.niblack(...)`

Niblack local thresholding.

The threshold at each pixel is `mean + k * std`, where `k` is typically negative to classify pixels below the local mean as foreground.

**Additional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `25` | Size of the local neighborhood window. Must be ≥ 3. |
| `k` | `float` | `-0.2` | Scaling factor for the local standard deviation. Negative values classify darker pixels as foreground. |

- `binary_type` value: `"niblack"`

**Example**

```python
self.preprocessing.binarization.niblack(window_size=25, k=-0.2)
```

---

### `self.preprocessing.binarization.urbano(...)`

Placeholder Urbano-style binarization. Currently produces a zero-valued binary video.

- `binary_type` value: `"urbano"`

**Example**

```python
self.preprocessing.binarization.urbano()
```
