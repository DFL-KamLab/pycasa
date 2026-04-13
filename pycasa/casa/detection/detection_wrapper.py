class _SessionDetectionNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the detection namespace for a ``Casa`` session."""
        self._session = session

    def detect_moving_cells(
        self,
        method: str = "cv-gmg",
        *,
        show_progress: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> "Casa":
        """Detect moving cells from video using legacy-parity logic.

        Parameters:
            method (str, optional):
                Moving-cells extraction backend identifier:
                ``cv-gmg``, ``cv-mog``, ``cv-mog2``, or ``gm``.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.
            **kwargs:
                Method-specific controls forwarded unchanged to
                :func:`pycasa_as.detection.detect_moving_cells`.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.
        """
        from ...detection import detect_moving_cells

        return self._session._sync_from(
            detect_moving_cells(
                self._session._as_dict(),
                method=method,
                show_progress=show_progress,
                verbose=verbose,
                **kwargs,
            )
        )

    def digital_washing(
        self,
        motion_threshold: float = 3.0,
        number_training_frames: int = 20,
        blob_min_pixel_area: int = 20,
        k_val: float = 1.7,
        border_margin_px: int = 20,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Run Digital Washing detection on the current in-memory video.

        Parameters:
            motion_threshold (float, optional):
                Sigma threshold for Gaussian-mixture motion extraction.
            number_training_frames (int, optional):
                Warm-up frame count used by motion/background separation.
            blob_min_pixel_area (int, optional):
                Minimum connected-component area retained as candidate.
            k_val (float, optional):
                Standard-deviation multiplier used in local detector rules.
            border_margin_px (int, optional):
                Border exclusion margin in pixels.
            show_progress (bool, optional):
                If ``True``, show shared pycasa progress bars for iterative
                Digital Washing stages.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ValueError:
                If ``casa["video"]["original_video"]`` has unsupported shape.

        Notes:
            - Stores detections under ``casa["detections"]["digital_washing"]``.
            - Writes intermediate outputs to:
              ``digital_washing_motion_video``,
              ``digital_washing_binarized_video``,
              ``digital_washing_background_video``.
            - Updates ``casa["meta"]["last_detection"]``.

        Examples:
            >>> session = session.detection.digital_washing(show_progress=False)
        """
        from ...detection import digital_washing

        return self._session._sync_from(
            digital_washing(
                self._session._as_dict(),
                motion_threshold=motion_threshold,
                number_training_frames=number_training_frames,
                blob_min_pixel_area=blob_min_pixel_area,
                k_val=k_val,
                border_margin_px=border_margin_px,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def yolov5(
        self,
        weights: str = "sys-casa_yolov5s.pt",
        conf: float = 0.15,
        delete_temp: bool = True,
        download: bool = True,
        force_download: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Run YOLOv5 detection on the current in-memory video.

        Parameters:
            weights (str, optional):
                Managed weight name (for example ``sys-opt_yolov5m.pt``) or
                custom local weight file path.
            conf (float, optional):
                Detection confidence threshold.
            delete_temp (bool, optional):
                Legacy compatibility flag preserved for API parity.
            download (bool, optional):
                If ``True``, automatically download missing managed weights.
            force_download (bool, optional):
                If ``True``, force a fresh download for managed weights.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while running
                per-frame YOLO inference.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries and
                confidence statistics.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.
        """
        from ...detection import yolov5

        return self._session._sync_from(
            yolov5(
                self._session._as_dict(),
                weights=weights,
                conf=conf,
                delete_temp=delete_temp,
                download=download,
                force_download=force_download,
                show_progress=show_progress,
                verbose=verbose,
            )
        )
