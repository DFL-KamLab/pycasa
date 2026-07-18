from typing import Any

from .._core._casa import _ensure_casa
from ..utils import _build_casa_info
from ..utils import copy_casa
from ..utils import get_assesment
from ..utils import get_assessment
from ..utils import get_casa
from ..utils import get_detections
from ..utils import get_groundtruth
from ..utils import get_meta
from ..utils import get_motility
from ..utils import get_tracks
from ..utils import get_video
from ..utils import _GROUNDTRUTH_TRACKS_KEY
from ..utils import _print_casa_info
from ..utils import set_um_per_px
from .assessment import _SessionAssessmentNamespace
from .detection import _SessionDetectionNamespace
from .io import _SessionIONamespace
from .motility import _SessionMotilityNamespace
from .preprocessing import _SessionPreprocessingNamespace
from .tracking import _SessionTrackingNamespace
from .visualization import _SessionVisualizationNamespace



class Casa:
    def __init__(self, casa=None):
        """Create a ``Casa`` wrapper around a validated session dictionary."""
        self._casa = _ensure_casa(casa)

    def _as_dict(self):
        """Return the underlying mutable CASA session dictionary."""
        return self._casa

    def info(self) -> "Casa":
        """Print a structured summary of the current CASA session and return ``self``."""
        info = _build_casa_info(self._casa)
        _print_casa_info(info)
        return self

    def _sync_from(self, casa):
        """Replace internal session state from another CASA dictionary and return ``self``."""
        self._casa = _ensure_casa(casa)
        return self

    def copy(self) -> "Casa":
        """Return a deep-copied ``Casa`` session independent from this instance."""
        return Casa(copy_casa(self._casa))

    def set_um_per_px(self, um_per_px: float) -> "Casa":
        """Set microns-per-pixel calibration on the current session.

        Parameters:
            um_per_px (float):
                Positive finite microns-per-pixel value.

        Returns:
            Casa:
                The same fluent session instance.
        """
        return self._sync_from(set_um_per_px(self._as_dict(), um_per_px))

    def get_casa(self) -> dict[str, Any]:
        """Return the validated CASA dictionary."""
        return get_casa(self._as_dict())

    def get_meta(self) -> dict[str, Any]:
        """Return ``casa['meta']``."""
        return get_meta(self._as_dict())

    def get_video(self) -> dict[str, Any]:
        """Return ``casa['video']``."""
        return get_video(self._as_dict())

    def get_detections(self, *, include_groundtruth: bool = False) -> dict[str, Any]:
        """Return active predicted detections, or all detections when requested."""
        return get_detections(
            self._as_dict(),
            include_groundtruth=include_groundtruth,
        )

    def get_groundtruth(self) -> dict[str, Any]:
        """Return ``casa['detections']['groundtruth']``."""
        return get_groundtruth(self._as_dict())

    def get_tracks(self, *, backend: str | None = None) -> dict[str, Any]:
        """Return all tracks or one backend bucket (for example ``sort``)."""
        return get_tracks(self._as_dict(), backend=backend)

    def get_groundtruth_tracks(self) -> dict[str, Any]:
        """Return imported ground-truth tracks (``casa['tracks']['groundtruth_tracks']``).

        These are persistent identities read from label files via
        ``load_video(..., groundtruth_tracks_path=...)``, distinct from
        ``casa['tracks'][backend]['groundtruth']`` (tracks a backend computed
        from groundtruth detections).
        """
        return get_tracks(self._as_dict(), backend=_GROUNDTRUTH_TRACKS_KEY)

    def get_motility(self) -> dict[str, Any]:
        """Return ``casa['motility']``."""
        return get_motility(self._as_dict())

    def get_assessment(self) -> dict[str, Any]:
        """Return ``casa['assessment']``."""
        return get_assessment(self._as_dict())

    def get_assesment(self) -> dict[str, Any]:
        """Compatibility spelling alias for :meth:`get_assessment`."""
        return get_assesment(self._as_dict())

    @property
    def io(self) -> _SessionIONamespace:
        """Access session I/O methods.

        Purpose:
            Load videos or default bundled data into the current ``Casa`` session.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionIONamespace:
                Namespace exposing:
                - ``load_video(...)``
                - ``load_default_data(...)``
        """
        return _SessionIONamespace(self)

    @property
    def preprocessing(self) -> _SessionPreprocessingNamespace:
        """Access preprocessing methods for the loaded video.

        Purpose:
            Apply grayscale, normalization, and binarization operations.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionPreprocessingNamespace:
                Namespace exposing:
                - ``grayscale(...)``
                - ``normalization.<method>(...)``
                - ``binarization.<method>(...)``
        """
        return _SessionPreprocessingNamespace(self)

    @property
    def detection(self) -> _SessionDetectionNamespace:
        """Access detection backends for sperm/cell localization.

        Purpose:
            Run a single active predicted detection method and store detections.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionDetectionNamespace:
                Namespace exposing:
                - ``detect_moving_cells(...)``
                - ``digital_washing(...)``
                - ``urbano_detection(...)``
                - ``yolo(...)``
        """
        return _SessionDetectionNamespace(self)

    @property
    def tracking(self) -> _SessionTrackingNamespace:
        """Access tracking methods.

        Purpose:
            Build trajectories from available detections/groundtruth.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionTrackingNamespace:
                Namespace exposing:
                - ``sort(...)``
                - ``deepsort(...)``
                - ``jpdaf(...)``
        """
        return _SessionTrackingNamespace(self)

    @property
    def motility(self) -> _SessionMotilityNamespace:
        """Access motility-parameter computation methods.

        Purpose:
            Compute standard motility metrics from tracked trajectories.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionMotilityNamespace:
                Namespace exposing:
                - ``standard_motility_parameters(...)``
        """
        return _SessionMotilityNamespace(self)

    @property
    def assessment(self) -> _SessionAssessmentNamespace:
        """Access assessment/evaluation methods.

        Purpose:
            Evaluate predictions against groundtruth (detections and tracks).

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionAssessmentNamespace:
                Namespace exposing:
                - ``evaluate_detections(...)``
                - ``evaluate_tracks(...)``
        """
        return _SessionAssessmentNamespace(self)

    @property
    def visualization(self) -> _SessionVisualizationNamespace:
        """Access visualization and interactive inspection tools.

        Purpose:
            Display frames, overlays, trajectories, and motility visual summaries.

        Inputs:
            None. This property returns a namespace object.

        Returns:
            _SessionVisualizationNamespace:
                Namespace exposing:
                - ``plot_frame(...)``
                - ``timelapse(...)``
                - ``interactive_motility_calculator(...)``
                - ``motility_radar(...)``
                - ``motility_density_scatter(...)``
        """
        return _SessionVisualizationNamespace(self)
