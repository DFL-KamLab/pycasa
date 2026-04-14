# API: Visualization

Purpose:
Visualization provides interactive and static inspection surfaces for every stage of the session, from raw frames and overlays to track exploration and motility summaries.

## Public Methods In This Section

- `self.visualization.plot_frame(...)`
- `self.visualization.timelapse(...)`
- `self.visualization.interactive_motility_calculator(...)`
- `self.visualization.motility_radar(...)`
- `self.visualization.motility_density_scatter(...)`

## Quick Usage

```python
self.visualization.plot_frame(frame_index=10, show_detections=True)
self.visualization.timelapse(
    video_type="original",
    show_detections=True,
    show_tracks=True,
    show_groundtruth=True,
)
```

## Typical Use Order

1. Use `plot_frame` for spot checks.
2. Use `timelapse` for visual pipeline review.
3. Use `motility_radar` and `motility_density_scatter` after motility computation.
