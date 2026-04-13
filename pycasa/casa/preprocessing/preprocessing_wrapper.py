class _SessionBinarizationNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the binarization namespace for a ``Casa`` session."""
        self._session = session

    def otsu(self, *, show_progress: bool = True, verbose: bool = True) -> "Casa":
        """Binarize frames using per-frame Otsu thresholding.

        Parameters:
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``casa["video"]["original_video"]`` is missing or has invalid shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Input frames are converted to grayscale internally.
            - Output is ``uint8`` with values ``0`` and ``255``.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "otsu"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.otsu()
        """
        from ...preprocessing.binarization import otsu

        return self._session._sync_from(
            otsu(
                self._session._as_dict(),
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def adaptive_mean(
        self,
        *,
        block_size: int = 11,
        c: float = 2.0,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Binarize frames with OpenCV adaptive mean thresholding.

        Parameters:
            block_size (int, optional):
                Odd neighborhood size (>= 3) used by adaptive thresholding.
            c (float, optional):
                Constant subtracted from neighborhood mean before thresholding.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``block_size`` is not an odd integer >= 3, or if the video
                array is missing/invalid.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ImportError:
                If ``opencv-python`` is unavailable.

        Notes:
            - Input frames are converted to grayscale internally.
            - Output is ``uint8`` with values ``0`` and ``255``.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "adaptive-mean"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.adaptive_mean(block_size=11, c=2.0)
        """
        from ...preprocessing.binarization import adaptive_mean

        return self._session._sync_from(
            adaptive_mean(
                self._session._as_dict(),
                block_size=block_size,
                c=c,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def adaptive_gaussian(
        self,
        *,
        block_size: int = 11,
        c: float = 2.0,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Binarize frames with OpenCV adaptive Gaussian thresholding.

        Parameters:
            block_size (int, optional):
                Odd neighborhood size (>= 3) used by adaptive thresholding.
            c (float, optional):
                Constant subtracted from neighborhood-weighted mean.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``block_size`` is not an odd integer >= 3, or if the video
                array is missing/invalid.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ImportError:
                If ``opencv-python`` is unavailable.

        Notes:
            - Input frames are converted to grayscale internally.
            - Output is ``uint8`` with values ``0`` and ``255``.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "adaptive-gaussian"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.adaptive_gaussian(block_size=11, c=2.0)
        """
        from ...preprocessing.binarization import adaptive_gaussian

        return self._session._sync_from(
            adaptive_gaussian(
                self._session._as_dict(),
                block_size=block_size,
                c=c,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def sauvola(
        self,
        *,
        window_size: int = 25,
        k: float = 0.2,
        r: float = 128.0,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Binarize frames with Sauvola local thresholding.

        Parameters:
            window_size (int, optional):
                Local window size (>= 3) used to estimate local mean and
                standard deviation.
            k (float, optional):
                Sauvola contrast factor.
            r (float, optional):
                Dynamic range parameter used in the Sauvola formula.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``window_size`` is < 3, or if the video is
                missing/invalid.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ImportError:
                If ``scipy`` is unavailable.

        Notes:
            - Input frames are converted to grayscale internally.
            - Output is ``uint8`` with values ``0`` and ``255``.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "sauvola"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.sauvola(window_size=25, k=0.2, r=128.0)
        """
        from ...preprocessing.binarization import sauvola

        return self._session._sync_from(
            sauvola(
                self._session._as_dict(),
                window_size=window_size,
                k=k,
                r=r,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def niblack(
        self,
        *,
        window_size: int = 25,
        k: float = -0.2,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Binarize frames with Niblack local thresholding.

        Parameters:
            window_size (int, optional):
                Local window size (>= 3) used to estimate local mean and
                standard deviation.
            k (float, optional):
                Niblack scaling factor applied to local standard deviation.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``window_size`` is < 3, or if the video is
                missing/invalid.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ImportError:
                If ``scipy`` is unavailable.

        Notes:
            - Input frames are converted to grayscale internally.
            - Output is ``uint8`` with values ``0`` and ``255``.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "niblack"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.niblack(window_size=25, k=-0.2)
        """
        from ...preprocessing.binarization import niblack

        return self._session._sync_from(
            niblack(
                self._session._as_dict(),
                window_size=window_size,
                k=k,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def urbano(self, *, show_progress: bool = True, verbose: bool = True) -> "Casa":
        """Run the placeholder Urbano-style binarization implementation.

        Parameters:
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``casa["video"]["original_video"]`` is missing or has invalid shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - This is currently a placeholder implementation that writes
              zero-valued binary frames.
            - Writes ``casa["video"]["binary_video"]`` and
              ``casa["video"]["binary_type"] = "urbano"``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.binarization.urbano()
        """
        from ...preprocessing.binarization import urbano

        return self._session._sync_from(
            urbano(
                self._session._as_dict(),
                show_progress=show_progress,
                verbose=verbose,
            )
        )


class _SessionNormalizationNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the normalization namespace for a ``Casa`` session."""
        self._session = session

    def min_max(
        self,
        *,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames using per-frame min-max scaling to ``[0, 255]``.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "min-max"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.min_max(overwrite=False)
        """
        from ...preprocessing.normalization import min_max

        return self._session._sync_from(
            min_max(
                self._session._as_dict(),
                overwrite=overwrite,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def z_score(
        self,
        *,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames using per-frame z-score standardization.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Per frame, output is ``(frame - mean) / std`` in ``float32``.
            - Zero-variance frames become zeros.
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "z-score"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.z_score(overwrite=False)
        """
        from ...preprocessing.normalization import z_score

        return self._session._sync_from(
            z_score(
                self._session._as_dict(),
                overwrite=overwrite,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def hist_equal(
        self,
        *,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames with global histogram equalization.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Equalization is applied per frame.
            - For color frames, equalization is applied channel-wise.
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "hist-equal"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.hist_equal(overwrite=False)
        """
        from ...preprocessing.normalization import hist_equal

        return self._session._sync_from(
            hist_equal(
                self._session._as_dict(),
                overwrite=overwrite,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def clahe(
        self,
        *,
        overwrite: bool = False,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames using CLAHE (contrast-limited adaptive histogram equalization).

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            clip_limit (float, optional):
                CLAHE clip limit passed to OpenCV.
            tile_grid_size (tuple[int, int], optional):
                Tile grid size passed to OpenCV CLAHE.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.
            ImportError:
                If ``opencv-python`` is unavailable.

        Notes:
            - CLAHE is applied per frame.
            - For color frames, CLAHE is applied channel-wise.
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "clahe"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.clahe(overwrite=False)
        """
        from ...preprocessing.normalization import clahe

        return self._session._sync_from(
            clahe(
                self._session._as_dict(),
                overwrite=overwrite,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def log(
        self,
        *,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames using ``log1p`` followed by min-max scaling.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Per frame, values are transformed with ``np.log1p`` then scaled to
              ``[0, 255]``.
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "log"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.log(overwrite=False)
        """
        from ...preprocessing.normalization import log

        return self._session._sync_from(
            log(
                self._session._as_dict(),
                overwrite=overwrite,
                show_progress=show_progress,
                verbose=verbose,
            )
        )

    def median(
        self,
        *,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Normalize frames by median-centering then min-max scaling.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with normalized
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If the video is missing or does not have 3D/4D shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Per frame, the median is subtracted before scaling to ``[0, 255]``.
            - Output is stored in ``casa["video"]["normalized_video"]``.
            - Method tag is stored in ``casa["video"]["normalized_type"] = "median"``.
            - When ``overwrite=True``, ``casa["video"]["original_video"]`` is replaced.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.normalization.median(overwrite=False)
        """
        from ...preprocessing.normalization import median

        return self._session._sync_from(
            median(
                self._session._as_dict(),
                overwrite=overwrite,
                show_progress=show_progress,
                verbose=verbose,
            )
        )


class _SessionPreprocessingNamespace:
    def __init__(self, session: "Casa") -> None:
        """Initialize the preprocessing namespace for a ``Casa`` session."""
        self._session = session

    @property
    def binarization(self) -> _SessionBinarizationNamespace:
        """Return binarization methods for this session."""
        return _SessionBinarizationNamespace(self._session)

    @property
    def normalization(self) -> _SessionNormalizationNamespace:
        """Return normalization methods for this session."""
        return _SessionNormalizationNamespace(self._session)

    def grayscale(
        self,
        overwrite: bool = False,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> "Casa":
        """Convert loaded video frames to grayscale.

        Parameters:
            overwrite (bool, optional):
                If ``True``, replace ``casa["video"]["original_video"]`` with grayscale
                output. If ``False``, keep the original array unchanged.
            show_progress (bool, optional):
                If ``True``, show the shared pycasa progress bar while
                processing frames.
            verbose (bool, optional):
                If ``True``, print concise runtime start/end summaries.

        Returns:
            Casa:
                The same fluent ``Casa`` session instance.

        Raises:
            ValueError:
                If ``casa["video"]["original_video"]`` is missing or has invalid shape.
            TypeError:
                If ``casa["video"]["original_video"]`` is not a numpy array.

        Notes:
            - Always writes ``casa["video"]["grayscale_video"]``.
            - When ``overwrite=True``, also writes ``casa["video"]["original_video"]``.
            - Updates ``casa["meta"]["last_preprocessing"]``.

        Examples:
            >>> session = session.preprocessing.grayscale(overwrite=False)
        """
        from ...preprocessing import grayscale

        processed = grayscale(
            self._session._as_dict(),
            overwrite=overwrite,
            show_progress=show_progress,
            verbose=verbose,
        )
        return self._session._sync_from(processed)
