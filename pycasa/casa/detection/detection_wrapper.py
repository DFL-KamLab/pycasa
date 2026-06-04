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
                :func:`pycasa.detection.detect_moving_cells`.

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

    def yolo(
        self,
        yolo_model: str = "yolo26",
        weights: str | None = None,
        conf: float = 0.15,
        download: bool = True,
        force_download: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Run YOLO detection on the current in-memory video.

        Parameters:
            yolo_model (str, optional):
                YOLO architecture to use: ``"yolov5"`` or ``"yolo26"``.
            weights (str | None, optional):
                Managed weight name or a custom local file path. When ``None``,
                the default managed weight for the chosen ``yolo_model`` is used.

                **YOLOv5 managed weights** (downloaded automatically):

                - ``sys-casa_yolov5n/s/m/l/x.pt``
                - ``sys-opt_yolov5n/s/m/l/x.pt``

                Default: ``sys-casa_yolov5s.pt``

                **YOLO26 managed weights** (downloaded automatically):

                - ``sys-casa_yolo26n/s/m/l/x.pt``
                - ``sys-opt_yolo26n/s/m/l/x.pt``

                Default: ``sys-casa_yolo26n.pt``

            conf (float, optional):
                Detection confidence threshold.
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

        Notes:
            Detections are stored under
            ``casa["detections"][yolo_model]`` (``"yolov5"`` or ``"yolo26"``).

        Examples:
            >>> session = session.detection.yolo(yolo_model="yolo26",
            ...     weights=r"D:\\weights\\best.pt", download=False)
            >>> session = session.detection.yolo(yolo_model="yolov5")
        """
        from ...detection import yolo

        return self._session._sync_from(
            yolo(
                self._session._as_dict(),
                yolo_model=yolo_model,
                weights=weights,
                conf=conf,
                download=download,
                force_download=force_download,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def urbano_detection(
        self,
        weight: float = 1.0,
        gaussian_size: int = 11,
        gaussian_iters: int = 5,
        log_size: int = 9,
        min_pixels: int = 5,
        *,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Run Urbano et al. (2017) LoG-based sperm detection on the current video.

        Parameters:
            weight (float, optional):
                Multiplier applied to Otsu's per-frame threshold. Values > 1
                raise the threshold (fewer detections); values < 1 lower it.
            gaussian_size (int, optional):
                Side length in pixels of the Gaussian kernel (paper: 11).
            gaussian_iters (int, optional):
                Number of times the Gaussian filter is applied (paper: 5).
            log_size (int, optional):
                Side length in pixels of the LoG kernel (paper: 9).
            min_pixels (int, optional):
                Minimum connected-component area in pixels to keep as a
                detection (paper: 5).
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Notes:
            Detections are stored under
            ``casa["detections"]["urbano_detection"]``.

        Examples:
            >>> session = session.detection.urbano_detection()
        """
        from ...detection import urbano_detection as _urbano_detection

        return self._session._sync_from(
            _urbano_detection(
                self._session._as_dict(),
                weight=weight,
                gaussian_size=gaussian_size,
                gaussian_iters=gaussian_iters,
                log_size=log_size,
                min_pixels=min_pixels,
                show_progress=show_progress,
                verbose=verbose,
            )
        )
