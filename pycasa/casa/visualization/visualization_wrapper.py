from typing import Any
from typing import Iterable


class _SessionVisualizationNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the visualization namespace for a ``Casa`` session."""
        self._session = session

    def plot_frame(
        self,
        image_type: str | Iterable[str] = "original",
        frame_index: int | None = None,
        show_detections: bool = True,
    ) -> "Casa":
        """Plot one frame across one or more image representations.

        Parameters:
            image_type (str | Iterable[str], optional):
                One or more image types (single value, ``+``/``,`` separated string,
                or iterable) among ``original``, ``grayscale``, ``normalized``,
                ``binarized``, and ``moving_cells``.
            frame_index (int | None, optional):
                Zero-based local index to display. If ``None``, the middle frame is
                selected from available videos.
            show_detections (bool, optional):
                Whether to overlay detections on the plotted frame.

        Returns:
            Casa:
                The same session instance with visualization metadata in
                ``casa['meta']['last_visualization']``.

        Raises:
            ValueError:
                If a requested image video is missing or frame selection is invalid.
            ImportError:
                If ``matplotlib`` is unavailable.
        """
        from ...visualization import plot_frame

        return self._session._sync_from(
            plot_frame(
                self._session._as_dict(),
                image_type=image_type,
                frame_index=frame_index,
                show_detections=show_detections,
            )
        )

    def timelapse(
        self,
        video_type: str = "original",
        image_type: str | None = None,
        show_detections: bool = True,
        show_tracks: bool = False,
        show_groundtruth: bool = True,
        show_track_ids: bool = False,
        detection_color: str | None = None,
        groundtruth_color: str | None = None,
        track_colors: dict[str, Any] | None = None,
    ) -> "Casa":
        """Open an interactive time-lapse viewer for selected video representations.

        Parameters:
            video_type (str | Iterable[str], optional):
                Video representation(s) to show. Accepts single value or combined
                values separated by ``+`` or ``,``. Supported names:
                ``original``, ``grayscale``/``gray``, ``normalized``,
                ``binarized``/``binary``, ``moving_cells``.
            image_type (str | None, optional):
                Deprecated compatibility alias for ``video_type``. If provided, it
                overrides ``video_type``.
            show_detections (bool, optional):
                Initial visibility state for active detection overlays.
            show_tracks (bool, optional):
                Initial visibility state for tracks.
            show_groundtruth (bool, optional):
                Initial visibility state for groundtruth detections.
            show_track_ids (bool, optional):
                Whether to annotate each track head with track ID text. This can be
                expensive for large track sets.
            detection_color (str | None, optional):
                Matplotlib color used for predicted-detection bounding boxes.
                ``None`` keeps the default (``"#ff4d4d"``).
            groundtruth_color (str | None, optional):
                Matplotlib color used for groundtruth bounding boxes. ``None``
                keeps the default (``"lime"``).
            track_colors (dict[str, Any] | None, optional):
                Per-source color override for track lines. Keys may be the
                literal source name (e.g. ``"groundtruth"``, ``"yolo"``) or a
                role alias: ``"groundtruth"`` for the groundtruth source and
                ``"detection"`` for the active predicted-detection source.

        Returns:
            Casa:
                The same session instance with viewer metadata written to
                ``casa['meta']['last_visualization']``.

        Raises:
            ValueError:
                If requested video layers are unavailable or frame data is invalid.
            ImportError:
                If ``matplotlib`` is unavailable.

        Notes:
            Controls:
            - Slider: scrub frames.
            - Play/Pause button: animate frames at session sampling rate.
            - Checkboxes: toggle detections/tracks/groundtruth overlays.
            - Keyboard: left/right arrows step frames, space toggles play.
            Track sources include any active backend sources plus imported
            ground-truth tracks (``casa["tracks"]["groundtruth_tracks"]``), each
            with its own toggle; enable with ``show_tracks=True``.

        Examples:
            >>> session = session.visualization.timelapse(
            ...     detection_color="blue",
            ...     groundtruth_color="red",
            ...     track_colors={"groundtruth": "yellow", "detection": "orange"},
            ... )
        """
        from ...visualization import timelapse as timelapse_fn

        return self._session._sync_from(
            timelapse_fn(
                self._session._as_dict(),
                video_type=video_type,
                image_type=image_type,
                show_detections=show_detections,
                show_tracks=show_tracks,
                show_groundtruth=show_groundtruth,
                show_track_ids=show_track_ids,
                detection_color=detection_color,
                groundtruth_color=groundtruth_color,
                track_colors=track_colors,
            )
        )

    def interactive_motility_calculator(
        self,
        frame_rate: float | None = None,
        smoothing_window: int = 5,
    ) -> "Casa":
        """Open an interactive motility-parameter explorer for tracked sources.

        Works on the active tracking backend's sources and on imported
        ground-truth tracks (``casa["tracks"]["groundtruth_tracks"]``); when more
        than one source is available, a source switcher is shown.

        Parameters:
            frame_rate (float | None, optional):
                FPS override for segment metric computation. If ``None``, uses
                ``casa["meta"]["sampling_rate"]`` when available, otherwise ``30``.
            smoothing_window (int, optional):
                Smoothing window used by VAP/ALH calculations in the segment
                preview panel.

        Returns:
            Casa:
                The same session instance with metadata written to
                ``casa["meta"]["last_visualization"]``.

        Raises:
            ValueError:
                If video width/height metadata cannot be resolved.
            RuntimeError:
                If no tracks are available or no track has enough points
                for interactive exploration.
            ImportError:
                If ``matplotlib`` is unavailable.
        """
        from ...visualization import interactive_motility_calculator

        return self._session._sync_from(
            interactive_motility_calculator(
                self._session._as_dict(),
                frame_rate=frame_rate,
                smoothing_window=smoothing_window,
            )
        )

    def motility_radar(
        self,
        axis: Any = None,
        show_legend: bool = True,
        show_text: bool = True,
    ) -> "Casa":
        """Render a radar-chart summary for active kinematic parameters.

        Parameters:
            axis (Any, optional):
                Existing matplotlib polar axis. If ``None``, a new figure/axis
                is created.
            show_legend (bool, optional):
                Whether to display the legend.
            show_text (bool, optional):
                Whether to display numeric metric means below the chart.

        Returns:
            Casa:
                The same session instance with metadata written to
                ``casa["meta"]["last_visualization"]``.

        Raises:
            RuntimeError:
                If active kinematic results are missing.
            ImportError:
                If ``matplotlib`` is unavailable.
        """
        from ...visualization import motility_radar

        return self._session._sync_from(
            motility_radar(
                self._session._as_dict(),
                axis=axis,
                show_legend=show_legend,
                show_text=show_text,
            )
        )

    def motility_density_scatter(self) -> "Casa":
        """Render KDE density scatter plots for active kinematic metrics.

        Returns:
            Casa:
                The same session instance with metadata written to
                ``casa["meta"]["last_visualization"]``.

        Raises:
            RuntimeError:
                If active kinematic results are missing or contain no
                numeric values for plotting.
            ImportError:
                If ``matplotlib`` or ``scipy`` is unavailable.
        """
        from ...visualization import motility_density_scatter

        return self._session._sync_from(
            motility_density_scatter(self._session._as_dict())
        )
