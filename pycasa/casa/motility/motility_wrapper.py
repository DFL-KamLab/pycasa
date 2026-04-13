class _SessionMotilityNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the motility namespace for a ``Casa`` session."""
        self._session = session

    def standard_motility_parameters(
        self,
        frame_rate: float | None = None,
        window_size: int = 10,
        overlap: float = 0.2,
        smoothing_window: int | None = None,
        conversion_required: bool = True,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Compute legacy-standard motility parameters from tracked trajectories.

        Parameters:
            frame_rate (float | None, optional):
                FPS override. If ``None``, uses ``casa["meta"]["sampling_rate"]``
                when available, otherwise ``30``.
            window_size (int, optional):
                Number of points per sliding window.
            overlap (float, optional):
                Window overlap ratio used in legacy step calculation.
            smoothing_window (int | None, optional):
                Smoothing window for VAP/ALH. Defaults to
                ``max(2, window_size // 2)`` when ``None``.
            conversion_required (bool, optional):
                If ``True``, requires positive finite ``casa["meta"]["um_per_px"]``.
                If ``False``, missing calibration keeps output in pixel units.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing tracks.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If conversion is required but ``um_per_px`` is missing/invalid.
            RuntimeError:
                If no tracks have enough points for the requested window.

        Notes:
            Writes per-source output to
            ``casa["motility"]["standard_motility_parameters"][source]`` and
            stores run metadata in ``casa["meta"]["last_motility"]``.

        Examples:
            >>> session = session.motility.standard_motility_parameters()
        """
        from ...motility import standard_motility_parameters

        return self._session._sync_from(
            standard_motility_parameters(
                self._session._as_dict(),
                frame_rate=frame_rate,
                window_size=window_size,
                overlap=overlap,
                smoothing_window=smoothing_window,
                conversion_required=conversion_required,
                show_progress=show_progress,
                verbose=verbose,
            )
        )
