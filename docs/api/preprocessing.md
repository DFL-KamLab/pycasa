# API: Preprocessing

Purpose:
Preprocessing prepares loaded frames for downstream detection and tracking by generating stable visual representations (grayscale, normalized intensity, and binary masks).

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

## Grayscale

```python
self.preprocessing.grayscale(overwrite=False)
```

## Normalization Methods

- `min_max`
- `z_score`
- `hist_equal`
- `clahe`
- `log`
- `median`

Example:

```python
self.preprocessing.normalization.clahe(overwrite=False, clip_limit=2.0)
```

## Binarization Methods

- `otsu`
- `adaptive_mean`
- `adaptive_gaussian`
- `sauvola`
- `niblack`
- `urbano`

Example:

```python
self.preprocessing.binarization.otsu(show_progress=True, verbose=True)
```

## Output Behavior

- Writes derived arrays to `casa["video"]`.
- Updates `casa["meta"]["last_preprocessing"]`.
- Returns the same `Casa` instance for fluent chaining.
