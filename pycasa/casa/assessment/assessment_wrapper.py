class _SessionAssessmentNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the assessment namespace for a ``Casa`` session."""
        self._session = session

    def evaluate_detections(
        self,
        match_min_distance_pixel: float | None = None,
    ) -> "Casa":
        """Compute detection assessment metrics against groundtruth detections.

        Parameters:
            match_min_distance_pixel (float | None, optional):
                Pixel-distance threshold for matched true positives. If
                ``None``, uses ``casa['meta']['match_min_distance_pixel']``
                when available, else ``20``.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Notes:
            Writes results to:
            - ``casa['assessment']['detection']``
            - ``casa['assessment']['detection_log']``
            - ``casa['assessment']['last_detection']``
        """
        from ...assessment import evaluate_detections

        return self._session._sync_from(
            evaluate_detections(
                self._session._as_dict(),
                match_min_distance_pixel=match_min_distance_pixel,
            )
        )

    def evaluate_tracks(
        self,
        match_min_distance_pixel: float | None = None,
        backend: str | None = None,
    ) -> "Casa":
        """Compare every available track set against every other (MOTA/IDF1).

        Collects all track sets in the session — imported ground-truth tracks
        (``casa['tracks']['groundtruth_tracks']``) and every source of the
        active tracking backend (e.g. ``sort:groundtruth``, ``sort:yolov5``) —
        and computes pairwise MOT metrics (MOTA, IDF1, ID-switches, FP/FN,
        fragmentations) for each ordered pair via per-frame center-distance
        matching. The ``groundtruth_tracks`` row is true accuracy; other rows
        are pairwise agreement.

        Parameters:
            match_min_distance_pixel (float | None, optional):
                Association distance threshold (pixels). If ``None``, uses
                ``casa['meta']['match_min_distance_pixel']`` when available,
                else ``20``.
            backend (str | None, optional):
                Tracking backend to read predicted sources from. If ``None``,
                uses the active backend.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Notes:
            Writes results to:
            - ``casa['assessment']['tracking']`` (``{sources, reference, pairs}``)
            - ``casa['assessment']['last_tracking']``
            Requires the optional ``motmetrics`` dependency (installed on demand).
        """
        from ...assessment import evaluate_tracks

        return self._session._sync_from(
            evaluate_tracks(
                self._session._as_dict(),
                match_min_distance_pixel=match_min_distance_pixel,
                backend=backend,
            )
        )
