
class _SessionIONamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the fluent I/O namespace for a ``Casa`` session."""
        self._session = session

    def load_default_data(
        self,
        path: str | None = None,
        *,
        download: bool = True,
        force_download: bool = False,
        initial_frame: int = 0,
        final_frame: int | None = 100,
        sampling_rate: float | None = None,
        um_per_px: float | None = None,
        magnification: str | None = None,
        verbose: bool = True,
    ) -> "Casa":
        """Load the default HC004 session with MNE-style local caching.

        Parameters:
            path (str | None, optional):
                Root folder to cache/load default data. Resolution order:
                explicit argument, ``PYCASA_DATA``, then ``~/.pycasa_data``.
            download (bool, optional):
                If ``True``, missing required files trigger a selective download.
            force_download (bool, optional):
                If ``True``, always run the selective download before loading.
            initial_frame (int, optional):
                First frame index to read (0-based). Defaults to ``0``.
            final_frame (int | None, optional):
                Last frame index to read (0-based, inclusive). Defaults to ``100``.
            sampling_rate (float | None, optional):
                Optional FPS override forwarded to :func:`pycasa.io.load_video`.
            um_per_px (float | None, optional):
                Optional microns-per-pixel metadata value.
            magnification (str | None, optional):
                Optional magnification metadata value.
            verbose (bool, optional):
                If ``True``, print concise runtime summaries while resolving
                default-data cache/download behavior.

        Returns:
            Casa:
                Fluent ``Casa`` object loaded from the default video and groundtruth folder.

        Raises:
            FileNotFoundError:
                If required files are missing and download is disabled, or if they
                remain missing after download.
            ImportError:
                If download is needed and ``huggingface_hub`` is unavailable.

        Notes:
            Only this subset is downloaded:
            - ``sys-casa_sub-HC004_ses-01_run-005_video.avi``
            - ``sys-casa_sub-HC004_ses-01_run-005_video.json``
            - ``sys-casa_sub-HC004_ses-01_run-005_gt/*``
            - ``README.md`` in the same session folder.

        Examples:
            >>> import pycasa as pc
            >>> session = pc.io.load_default_data(download=False)
            >>> isinstance(session, pc.Casa)
            True
        """
        from ...io import load_default_data

        loaded = load_default_data(
            path=path,
            download=download,
            force_download=force_download,
            initial_frame=initial_frame,
            final_frame=final_frame,
            sampling_rate=sampling_rate,
            um_per_px=um_per_px,
            magnification=magnification,
            verbose=verbose,
        )
        return self._session._sync_from(loaded._as_dict())

    def load_video(
        self,
        video_path: str,
        groundtruth_detections_path: str | None = None,
        groundtruth_tracks_path: str | None = None,
        initial_frame: int = 0,
        final_frame: int | None = None,
        sampling_rate: float | None = None,
        show_progress: bool = True,
        verbose: bool = True,
        um_per_px: float | None = None,
        magnification: str | None = None,
    ) -> "Casa":
        """Load a time-lapse video and return a fluent ``Casa`` object.

        Parameters:
            video_path (str):
                Path to video file. Supported extensions are listed in
                ``SUPPORTED_MEDIA_TYPES``.
            groundtruth_detections_path (str | None, optional):
                Optional directory containing frame-level groundtruth detection
                text files (YOLO ``class cx cy w h`` rows). If provided,
                detections are loaded under ``casa["detections"]["groundtruth"]``.
            groundtruth_tracks_path (str | None, optional):
                Optional directory containing frame-level groundtruth track text
                files (YOLO rows prefixed with a persistent ``track_id``). If
                provided, imported truth tracks are loaded under
                ``casa["tracks"]["groundtruth_tracks"]``.
            initial_frame (int, optional):
                First frame index to read (0-based). Defaults to ``0``.
            final_frame (int | None, optional):
                Last frame index to read (0-based, inclusive). ``None`` means
                "to the end of the video".
            sampling_rate (float | None, optional):
                Explicit frame-rate override. If ``None``, FPS is read from video
                metadata when available.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa orange `#` progress bar while
                reading frames when ``tqdm`` is available.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries for the
                loading step.
            um_per_px (float | None, optional):
                Microns-per-pixel calibration stored in metadata.
            magnification (str | None, optional):
                Free-text magnification descriptor stored in metadata.

        Returns:
            Casa:
                A fluent ``Casa`` object with ``meta`` and ``video`` populated,
                plus optional ``detections["groundtruth"]`` and/or
                ``tracks["groundtruth_tracks"]`` if the corresponding groundtruth
                paths are provided.

        Raises:
            FileNotFoundError:
                If ``video_path`` does not exist.
            ValueError:
                If media extension or frame range is invalid.
            RuntimeError:
                If OpenCV cannot open/read any frames from the file.
            ImportError:
                If ``opencv-python`` is not installed.

        Notes:
            - Loaded video frames are stored in BGR order for OpenCV parity.
            - ``final_frame`` is clamped to the last available frame.
            - ``number_frame_used`` reflects actual readable frames, which may be
              smaller than requested if early read termination occurs.

        Examples:
            >>> import pycasa as pc
            >>> session = pc.io.load_video("sample.avi", initial_frame=0, final_frame=100)
            >>> isinstance(session, pc.Casa)
            True
        """
        from ...io import load_video

        loaded = load_video(
            video_path=video_path,
            groundtruth_detections_path=groundtruth_detections_path,
            groundtruth_tracks_path=groundtruth_tracks_path,
            initial_frame=initial_frame,
            final_frame=final_frame,
            sampling_rate=sampling_rate,
            show_progress=show_progress,
            verbose=verbose,
            um_per_px=um_per_px,
            magnification=magnification,
        )
        return self._session._sync_from(loaded._as_dict())
