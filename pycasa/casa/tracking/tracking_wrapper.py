class _SessionTrackingNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the tracking namespace for a ``Casa`` session."""
        self._session = session

    def jpdaf(
        self,
        skip_gt: bool = False,
        frame_rate: float | None = None,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Track detections using JPDAF and store tracks in session state.

        Implements the Joint Probabilistic Data Association Filter from
        Urbano et al. (2017), IEEE Transactions on Medical Imaging 36(3),
        792–801.  All algorithm parameters (σₙ, γᵥ, λ, etc.) are drawn
        directly from the paper and scaled to pixel space using
        ``casa['meta']['um_per_px']``.

        Parameters:
            skip_gt (bool, optional):
                If ``False`` (default), run JPDAF on both available sources:
                ``groundtruth`` and the active predicted detection method.
                If ``True``, skip groundtruth and track only predictions.
            frame_rate (float | None, optional):
                Frames per second.  When ``None``, read from
                ``casa['meta']['sampling_rate']``.  Required to derive the
                CWNA motion-model frame period ``T = 1 / frame_rate``.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar during
                per-frame tracking.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``frame_rate`` cannot be resolved from parameters or
                session metadata, or if video dimensions are unavailable.

        Notes:
            Writes per-source tracks to ``casa['tracks']['jpdaf'][source]``
            and stores invocation metadata in ``casa['meta']['last_tracking']``.

        Examples:
            >>> session = session.tracking.jpdaf(skip_gt=False)
        """
        from ...tracking import jpdaf

        return self._session._sync_from(
            jpdaf(
                self._session._as_dict(),
                skip_gt=skip_gt,
                frame_rate=frame_rate,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def deepsort(
        self,
        skip_gt: bool = False,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 1.0,
        nn_budget: int | None = None,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Track detections using DeepSORT and store tracks in session state.

        Uses the original nwojke/deep_sort implementation, auto-cloned to
        ``~/.pycasa/deepsort/`` on first use (requires ``git`` on PATH).

        By default (``max_cosine_distance=1.0``) appearance-based matching is
        disabled and association relies purely on Kalman-filter predictions and
        IoU gating — recommended for biological cells whose visual appearance
        is near-identical across tracks.

        Parameters:
            skip_gt (bool, optional):
                If ``False`` (default), run DeepSORT on both available sources:
                ``groundtruth`` and active predicted detections. If ``True``,
                skip groundtruth and track only predictions.
            max_age (int, optional):
                Maximum number of missed frames before dropping a track.
            n_init (int, optional):
                Minimum consecutive detections needed before a track is
                confirmed (equivalent to ``min_hits`` in SORT).
            max_iou_distance (float, optional):
                Maximum IoU *distance* gate (distance = 1 − IoU).
                Default 0.7 corresponds to a minimum IoU of 0.3.
            max_cosine_distance (float, optional):
                Appearance-feature gate threshold.  ``1.0`` (default)
                disables appearance-based matching entirely.
            nn_budget (int | None, optional):
                Maximum appearance features stored per track.
                ``None`` = unlimited.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while running
                per-frame tracking.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Notes:
            Writes per-source tracks to ``casa['tracks']['deepsort'][source]``
            and stores invocation metadata in ``casa['meta']['last_tracking']``.

        Examples:
            >>> session = session.tracking.deepsort(skip_gt=False)
        """
        from ...tracking import deepsort as _deepsort

        return self._session._sync_from(
            _deepsort(
                self._session._as_dict(),
                skip_gt=skip_gt,
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def sort(
        self,
        skip_gt: bool = False,
        delete_temp: bool = True,
        max_age: int = 25,
        min_hits: int = 3,
        iou_threshold: float = 0.1,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Track detections using SORT and store tracks in session state.

        Parameters:
            skip_gt (bool, optional):
                If ``False`` (default), run SORT on both available sources:
                ``groundtruth`` and active predicted detections. If ``True``,
                run detections-only and skip groundtruth.
            delete_temp (bool, optional):
                Legacy compatibility flag. No temporary files are created in
                this in-process implementation.
            max_age (int, optional):
                Maximum number of missed frames before dropping a track.
            min_hits (int, optional):
                Minimum associated detections before a track is emitted.
            iou_threshold (float, optional):
                Minimum IoU threshold for assignment.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while running
                per-frame tracking.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If tracking cannot resolve frame geometry from metadata or the
                loaded video array.
            ImportError:
                If SciPy is unavailable when Hungarian assignment is needed.

        Notes:
            Writes per-source tracks to ``casa['tracks']['sort'][source]`` and
            stores invocation metadata in ``casa['meta']['last_tracking']``.

        Examples:
            >>> session = session.tracking.sort(skip_gt=False)
        """
        from ...tracking import sort

        return self._session._sync_from(
            sort(
                self._session._as_dict(),
                skip_gt=skip_gt,
                delete_temp=delete_temp,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                show_progress=show_progress,
                verbose=verbose,
            )
        )
