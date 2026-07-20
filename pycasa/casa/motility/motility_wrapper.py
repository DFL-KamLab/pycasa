class _SessionMotilityNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the motility namespace for a ``Casa`` session."""
        self._session = session

    def kinematic_parameters(
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
        """Compute per-track kinematic parameters from tracked trajectories.

        Kinematic parameters are the per-track, per-window velocities and shape
        descriptors: ``VCL``, ``VSL``, ``VAP``, ``LIN``, ``ALH``, ``WOB``,
        ``STR``, ``MAD``. Run this before :meth:`casa_parameters`.

        Parameters:
            frame_rate (float | None, optional):
                FPS override. If ``None``, uses ``casa["meta"]["sampling_rate"]``
                (read from the video). If that is also unavailable, a warning
                is issued and ``30`` is used as a fallback.
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
            ``casa["motility"]["kinematic_parameters"][source]`` and
            stores run metadata in ``casa["meta"]["last_motility"]``.

        Examples:
            >>> session = session.motility.kinematic_parameters()
        """
        from ...motility import kinematic_parameters

        return self._session._sync_from(
            kinematic_parameters(
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

    def casa_parameters(
        self,
        rapid_threshold: float = 25.0,
        immotile_threshold: float = 5.0,
        progressive_str_threshold: float = 0.8,
        velocity_metric: str = "VAP",
        volume_ml: float | None = None,
        chamber_depth_um: float | None = None,
        *,
        verbose: bool = True,
    ) -> "Casa":
        """Compute population-level CASA parameters from kinematic parameters.

        Classifies every track into the four WHO motility grades and reports
        their population percentages (``%rapid``, ``%slow``,
        ``%non_progressive``, ``%immotile``). When the required physical inputs
        are available, it also reports sperm **concentration** (needs
        ``chamber_depth_um`` + ``um_per_px``), **volume** (needs ``volume_ml``),
        and **total sperm count** (needs both). Missing inputs simply omit the
        corresponding output — the grades always compute.

        Run :meth:`kinematic_parameters` first.

        Parameters:
            rapid_threshold (float, optional):
                Velocity (um/s) at/above which a progressive track is *rapid*
                (WHO grade a). Default ``25``.
            immotile_threshold (float, optional):
                Velocity (um/s) below which a track is *immotile* (WHO grade d).
                Default ``5``.
            progressive_str_threshold (float, optional):
                STR (VSL/VAP, ratio in ``[0, 1]``) at/above which a motile track
                is *progressive*; below it the track is *non-progressive*
                (WHO grade c). Default ``0.8``.
            velocity_metric (str, optional):
                Which kinematic velocity drives the grade thresholds
                (``"VAP"``, ``"VCL"`` or ``"VSL"``). Default ``"VAP"``.
            volume_ml (float | None, optional):
                Ejaculate volume (mL). Overrides ``casa["meta"]["volume_ml"]``.
                Enables volume + total-count reporting when set.
            chamber_depth_um (float | None, optional):
                Counting-chamber depth (um). Resolved as argument, then
                ``casa["meta"]["chamber_depth_um"]``, then the ``20`` um
                default. With ``um_per_px`` present this enables concentration
                (and total-count) reporting.
            verbose (bool, optional):
                If ``True``, print a concise per-source summary.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Notes:
            Writes per-source output to
            ``casa["motility"]["casa_parameters"][source]`` and stores run
            metadata in ``casa["meta"]["last_casa_parameters"]``.

        Examples:
            >>> session = session.motility.kinematic_parameters()
            >>> session = session.motility.casa_parameters(chamber_depth_um=20, volume_ml=3.5)
        """
        from ...motility import casa_parameters

        return self._session._sync_from(
            casa_parameters(
                self._session._as_dict(),
                rapid_threshold=rapid_threshold,
                immotile_threshold=immotile_threshold,
                progressive_str_threshold=progressive_str_threshold,
                velocity_metric=velocity_metric,
                volume_ml=volume_ml,
                chamber_depth_um=chamber_depth_um,
                verbose=verbose,
            )
        )
