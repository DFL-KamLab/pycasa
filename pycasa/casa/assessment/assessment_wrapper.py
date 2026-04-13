class _SessionAssessmentNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the assessment namespace for a ``Casa`` session."""
        self._session = session

    def classification(
        self,
        match_min_distance_pixel: float | None = None,
    ) -> "Casa":
        """Compute detection classification metrics against groundtruth.

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
            - ``casa['assessment']['classification']``
            - ``casa['assessment']['classification_log']``
            - ``casa['assessment']['last_classification']``
        """
        from ...assessment import classification

        return self._session._sync_from(
            classification(
                self._session._as_dict(),
                match_min_distance_pixel=match_min_distance_pixel,
            )
        )
