"""Joint Probabilistic Data Association Filter (JPDAF) tracker.

Implements the algorithm described in:

    Urbano, L.F., Masson, P., VerMilyea, M., & Kam, M. (2017).
    Automatic Tracking and Motility Analysis of Human Sperm in
    Time-Lapse Images. IEEE Transactions on Medical Imaging, 36(3),
    792–801. https://doi.org/10.1109/TMI.2016.2630720

All parameter names and equation references throughout this module
correspond directly to the paper's notation (Sections II.C and Appendix).
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

from .._core._casa import _ensure_casa
from ..utils import (
    _clear_backend_tracks,
    _ensure_video_dimensions,
    _KNOWN_TRACKING_BACKENDS,
    _msg_yellow,
    _progress_bar,
    _resolve_active_predicted_detection_method,
    _warn_yellow,
)

# ---------------------------------------------------------------------------
# Paper constants (all in µm / µm² units — scaled to pixels inside jpdaf())
# ---------------------------------------------------------------------------

_SIGMA_N_UM: float = 2.0
"""Measurement noise standard deviation (µm). Paper Appendix A, σₙ = 2 µm."""

_DELTA_V_UM: float = 20.0
"""Velocity-uncertainty amplitude (µm/sec). Paper Appendix A, q̃₀ = 20 µm/sec.

The CWNA process-noise spectral density is derived as ``q₀ = (Δv/um_per_px)² / T``
(MATLAB: ``qIntensity = deltaV^2 / T``).  This is NOT applied directly as q̃₀
but first squared and divided by the frame period T before building Q₀.
"""

_GAMMA_P2: float = 11.6183
"""Position-gate threshold (dimensionless). χ²₂ CDF at P_G = 0.997. Paper Appendix C."""

_GAMMA_V_UM: float = 300.0
"""Velocity-gate threshold (µm/sec). Paper Appendix C, γᵥ = 300 µm/sec."""

_C1: float = 0.30
"""Fading-memory coefficient c₁. Paper Eq. 8."""

_C2: float = 0.50
"""Fading-memory coefficient c₂. Paper Eq. 8."""

_C3: float = 0.20
"""Fading-memory coefficient c₃ (= 1 − c₁ − c₂). Paper Eq. 8."""

_PD: float = 0.95
"""Per-target detection probability P_D. Paper Eq. 11."""

_LAMBDA_UM2: float = 1e-6
"""Poisson clutter spatial density λ (per µm²). MATLAB: ``lam_f = 1e-6``."""

_LAMBDA_N_UM2: float = 1e-5
"""New-target spatial density λₙ (per µm²). MATLAB: ``lam_n = 1e-5``.
Must satisfy λₙ > λ so that new tracks start with a positive initial score.
"""

_PDELETE: float = 1e-6
"""P(delete a true track by mistake). MATLAB: ``Pdelete = 1e-6``."""

_PCONFIRM: float = 1e-5
"""P(confirm a false track by mistake). MATLAB: ``Pconfirm = 1e-5``."""

_REDUNDANT_UM: float = 0.01
"""Track-suppression distance threshold (µm). Paper Appendix D."""

_LARGE_CLUSTER_PRUNE: float = 1e-6
"""Events with probability < this fraction of the best event are dropped for
large clusters (> 8 measurements) to keep enumeration tractable."""


# ---------------------------------------------------------------------------
# Helper: detection → centroid
# ---------------------------------------------------------------------------

def _detection_to_centroid(
    det: Any,
    width: int,
    height: int,
) -> list[float] | None:
    """Convert one detection row to pixel-space ``[cx, cy]``.

    Accepts the same flexible input formats as the SORT helper:
    dict with ``x1/y1/x2/y2``, dict with ``x/cx + w/h``, or a
    list/tuple in ``[label, cx, cy, w, h]`` or ``[x1, y1, x2, y2]``
    form.  Normalised [0, 1] coordinates are scaled up automatically.

    Parameters:
        det (Any):
            One detection row in any supported format.
        width (int):
            Frame width in pixels (used to de-normalise coordinates).
        height (int):
            Frame height in pixels (used to de-normalise coordinates).

    Returns:
        list[float] | None:
            ``[cx, cy]`` in pixel space, or ``None`` if the detection
            cannot be parsed.
    """
    cx: float
    cy: float

    if isinstance(det, dict):
        if all(k in det for k in ("x1", "y1", "x2", "y2")):
            try:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
            except (TypeError, ValueError):
                return None
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            x_raw = det.get("x", det.get("cx"))
            y_raw = det.get("y", det.get("cy"))
            if x_raw is None or y_raw is None:
                return None
            try:
                cx = float(x_raw)
                cy = float(y_raw)
            except (TypeError, ValueError):
                return None
    elif isinstance(det, (list, tuple)):
        if len(det) >= 5:
            try:
                cx = float(det[1])
                cy = float(det[2])
            except (TypeError, ValueError):
                return None
        elif len(det) == 4:
            try:
                x1 = float(det[0])
                y1 = float(det[1])
                x2 = float(det[2])
                y2 = float(det[3])
            except (TypeError, ValueError):
                return None
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            return None
    else:
        return None

    if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
        cx *= width
        cy *= height

    return [cx, cy]


def _get_frame_detections(raw_detections: dict[Any, Any], frame_key: int) -> list[Any]:
    """Fetch frame detections with string/int key compatibility.

    Parameters:
        raw_detections (dict[Any, Any]):
            Detection dict keyed by frame index (str or int).
        frame_key (int):
            Global frame index to look up.

    Returns:
        list[Any]:
            List of detection rows for the requested frame (may be empty).
    """
    for key in (str(frame_key), frame_key):
        data = raw_detections.get(key)
        if isinstance(data, list):
            return data
    return []


# ---------------------------------------------------------------------------
# CWNA system matrices  (Paper Appendix A, Eqs. 3–4)
# ---------------------------------------------------------------------------

def _build_matrices(
    T: float,
    sigma_n: float,
    q0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build constant CWNA Kalman matrices for the given frame period.

    Implements the system matrices from Paper Appendix A, Equations 3–4.

    Parameters:
        T (float):
            Time between consecutive frames in seconds.
        sigma_n (float):
            Measurement noise standard deviation in pixels.
        q0 (float):
            Initial process-noise spectral density in pixels/sec
            (corresponds to q̃₀ in the paper, already unit-converted).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ``(F, H, N, Q0)`` where

            * ``F`` (4 × 4) — state-transition matrix
            * ``H`` (2 × 4) — measurement matrix
            * ``N`` (2 × 2) — measurement noise covariance
            * ``Q0`` (4 × 4) — initial process noise covariance
    """
    I2 = np.eye(2)
    O2 = np.zeros((2, 2))

    F = np.block([[I2, I2 * T], [O2, I2]])           # Eq. 3
    H = np.block([[I2, O2]])                           # Eq. 3
    N = (sigma_n ** 2) * I2                            # σₙ²·I₂
    T3, T2 = T ** 3 / 3.0, T ** 2 / 2.0
    Q0 = q0 * np.block([[I2 * T3, I2 * T2],           # Eq. 4
                         [I2 * T2, I2 * T]])
    return F, H, N, Q0


# ---------------------------------------------------------------------------
# _KalmanTrack
# ---------------------------------------------------------------------------

class _KalmanTrack:
    """Per-sperm Kalman filter with CWNA motion model and adaptive process noise.

    Implements a single-target tracker as described in Paper Appendix A–B.
    Each instance maintains its own state estimate, covariance, and track
    score used for life-cycle management (tentative → confirmed → deleted).

    Attributes:
        track_id (int):
            Unique integer identifier assigned at construction.
        x (np.ndarray):
            Current filtered state ``[px, py, vx, vy]ᵀ`` (shape 4,).
        P (np.ndarray):
            Current state covariance (shape 4 × 4).
        Q (np.ndarray):
            Current adaptive process noise covariance (shape 4 × 4).
        Q0 (np.ndarray):
            Initial process noise covariance (shape 4 × 4) — frozen copy
            used in the fading-memory update.
        F (np.ndarray):
            State-transition matrix (shape 4 × 4).
        H (np.ndarray):
            Measurement matrix (shape 2 × 4).
        N (np.ndarray):
            Measurement noise covariance (shape 2 × 2).
        x_pred (np.ndarray):
            Predicted state ``x̂(k|k-1)`` set by :meth:`predict`.
        P_pred (np.ndarray):
            Predicted covariance ``P(k|k-1)`` set by :meth:`predict`.
        x_prev (np.ndarray):
            Previous filtered state ``x̂(k-1|k-1)`` saved before prediction,
            required for the fading-memory noise update (Paper Eq. 8).
        score (float):
            Running log-likelihood ratio score ℓₜ(k). Paper Eq. 23.
        max_score (float):
            Maximum score reached so far; used in deletion criterion.
        confirmed (bool):
            ``True`` once the score exceeds ηc (confirmation threshold).
        age (int):
            Total number of predict steps taken since creation.
    """

    _count: int = 0

    def __init__(
        self,
        measurement: np.ndarray,
        F: np.ndarray,
        H: np.ndarray,
        N: np.ndarray,
        Q0: np.ndarray,
        initial_score: float,
    ) -> None:
        """Initialise a new tentative track from a single measurement.

        Parameters:
            measurement (np.ndarray):
                Initial position observation ``[cx, cy]`` in pixels
                (shape 2,).
            F (np.ndarray):
                State-transition matrix (shape 4 × 4).
            H (np.ndarray):
                Measurement matrix (shape 2 × 4).
            N (np.ndarray):
                Measurement noise covariance (shape 2 × 2).
            Q0 (np.ndarray):
                Initial process noise covariance (shape 4 × 4).
            initial_score (float):
                Starting log-likelihood score ℓₜ(k₀⁰).  Paper Appendix D.
        """
        _KalmanTrack._count += 1
        self.track_id: int = _KalmanTrack._count

        self.F = F
        self.H = H
        self.N = N
        self.Q0 = Q0.copy()
        self.Q = Q0.copy()

        # Initial state: position from measurement, zero velocity
        self.x = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=float)
        self.x_prev = self.x.copy()

        # Initial covariance: position uncertainty from N, velocity from Q0
        I2, O2 = np.eye(2), np.zeros((2, 2))
        P_pos = N.copy()
        P_vel = Q0[2:, 2:].copy()
        self.P = np.block([[P_pos, O2], [O2, P_vel]])

        # Will be populated by predict()
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

        # Track life-cycle
        self.score: float = initial_score
        self.max_score: float = initial_score
        self.confirmed: bool = False
        self.age: int = 0

    # ------------------------------------------------------------------
    # Predict step  (Paper Eqs. 5–8)
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Propagate state one frame forward using the CWNA motion model.

        Saves the previous filtered state into ``x_prev``, computes the
        predicted state and covariance, and updates the adaptive process
        noise covariance using the fading-memory filter (Paper Eq. 8).

        Paper equations used:

        * Eq. 5 — predicted state: ``x̂(k|k-1) = F · x̂(k-1|k-1)``
        * Eq. 6 — predicted covariance: ``P(k|k-1) = F·P·Fᵀ + Q(k)``
        * Eq. 8 — adaptive Q: ``Q(k) = c₁·Q(k-1) + c₂·νₜ·νₜᵀ + c₃·Q₀``
          where ``νₜ = x̂(k|k-1) − x̂(k-1|k-1)``
        """
        self.x_prev = self.x.copy()

        x_pred = self.F @ self.x

        # Fading-memory process noise update (Eq. 8)
        nu_t = x_pred - self.x_prev
        self.Q = _C1 * self.Q + _C2 * np.outer(nu_t, nu_t) + _C3 * self.Q0

        P_pred = self.F @ self.P @ self.F.T + self.Q

        self.x_pred = x_pred
        self.P_pred = P_pred
        self.age += 1

    # ------------------------------------------------------------------
    # Innovation  (Paper Eqs. 9–10, 12–13)
    # ------------------------------------------------------------------

    def innovation(
        self,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute the innovation, residual covariance, and Mahalanobis distance.

        Parameters:
            measurement (np.ndarray):
                Observed position ``[cx, cy]`` in pixels (shape 2,).

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                * ``nu`` (shape 2,) — residual ``zⱼ − ẑ(k|k-1)`` (Eq. 9)
                * ``S``  (shape 2 × 2) — residual covariance (Eq. 10)
                * ``d2`` (float) — normalised Mahalanobis distance ``d²_jt``
                  (Eq. 13)
        """
        z_pred = self.H @ self.x_pred                    # Eq. 7: ẑ(k|k-1)
        nu = measurement - z_pred                         # Eq. 9
        S = self.H @ self.P_pred @ self.H.T + self.N     # Eq. 10
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        d2 = float(nu @ S_inv @ nu)                       # Eq. 13
        return nu, S, d2

    # ------------------------------------------------------------------
    # Kalman update  (Paper Eqs. 15–21)
    # ------------------------------------------------------------------

    def update(
        self,
        betas: list[float],
        residuals: list[np.ndarray],
        beta0: float,
    ) -> np.ndarray:
        """Apply probability-weighted Kalman update (JPDAF equations 15–21).

        Parameters:
            betas (list[float]):
                List of marginal association probabilities ``β_jt`` for each
                validated measurement ``j`` that belongs to this track's
                cluster assignment.  Must be non-empty.
            residuals (list[np.ndarray]):
                Corresponding innovations ``ν_jt`` (each shape 2,).
            beta0 (float):
                Probability that the track is not associated with any
                measurement: ``β₀t``.

        Returns:
            np.ndarray:
                Pseudo-measurement ``z̃ₜ(k)`` (shape 2,) as defined in
                Paper Eq. 18.  Needed for the track-score update.
        """
        # Innovation covariance (same for all measurements of this track)
        S = self.H @ self.P_pred @ self.H.T + self.N     # Eq. 10
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        W = self.P_pred @ self.H.T @ S_inv                # Eq. 15

        # Combined residual  νₜ = Σⱼ β_jt · ν_jt          (Eq. 16)
        nu_combined = sum(b * r for b, r in zip(betas, residuals))

        # State update  x̂(k|k) = x̂(k|k-1) + W·νₜ         (Eq. 17)
        self.x = self.x_pred + W @ nu_combined

        # Covariance update  (Eqs. 19–21)
        Pc = self.P_pred - W @ S @ W.T                    # Eq. 20

        # Spread of innovations P̃ₜ  (Eq. 21)
        weighted_outer = sum(
            b * np.outer(r, r) for b, r in zip(betas, residuals)
        )
        P_tilde = W @ (weighted_outer - np.outer(nu_combined, nu_combined)) @ W.T

        self.P = beta0 * self.P_pred + (1.0 - beta0) * Pc + P_tilde

        # Pseudo-measurement z̃ₜ = νₜ + ẑ(k|k-1)           (Eq. 18)
        z_pred = self.H @ self.x_pred
        pseudo_meas = nu_combined + z_pred
        return pseudo_meas

    # ------------------------------------------------------------------
    # Track score  (Paper Eqs. 23–25)
    # ------------------------------------------------------------------

    def score_step(
        self,
        pseudo_meas: np.ndarray | None,
        updated: bool,
        PD: float,
        lambda_c: float,
    ) -> None:
        """Increment the log-likelihood track score for life-cycle management.

        Implements Paper Equations 23–25.

        Parameters:
            pseudo_meas (np.ndarray | None):
                Pseudo-measurement ``z̃ₜ(k)`` (shape 2,) from :meth:`update`.
                Ignored (and may be ``None``) when ``updated=False``.
            updated (bool):
                ``True`` if the track received a probability-weighted
                measurement update this frame; ``False`` if missed.
            PD (float):
                Per-target detection probability.
            lambda_c (float):
                Poisson clutter spatial density (per px²).
        """
        if not updated:
            # Track not associated with any measurement (Eq. 25, top branch)
            delta = math.log(1.0 - PD)
        else:
            # Gaussian likelihood of pseudo-measurement given track (Eq. 25, bottom)
            S = self.H @ self.P_pred @ self.H.T + self.N
            z_pred = self.H @ self.x_pred
            nu_tilde = pseudo_meas - z_pred
            try:
                S_inv = np.linalg.inv(S)
                sign, log_det = np.linalg.slogdet(2.0 * math.pi * S)
                if sign <= 0:
                    log_det = 0.0
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
                log_det = 0.0
            log_gauss = -0.5 * (float(nu_tilde @ S_inv @ nu_tilde) + log_det)
            delta = math.log(max(lambda_c, 1e-300)) * -1 + math.log(max(PD, 1e-300)) + log_gauss

        self.score += delta
        if self.score > self.max_score:
            self.max_score = self.score


# ---------------------------------------------------------------------------
# Validation matrix  (Paper Appendix C)
# ---------------------------------------------------------------------------

def _build_validation_matrix(
    tracks: list[_KalmanTrack],
    measurements: np.ndarray,
    gamma_p2: float,
    gamma_v: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the binary validation matrix **A** and distance matrix **D**.

    A measurement ``j`` is validated by track ``t`` if it simultaneously
    satisfies both the position gate and the velocity gate (Paper Appendix C).

    * Position gate: ``d²_jt ≤ γ²_p`` (Chi-square gate, Paper Appendix C)
    * Velocity gate: ``‖ν_jt‖ / T ≤ γᵥ``

    Parameters:
        tracks (list[_KalmanTrack]):
            Active tracks with up-to-date predicted states.
        measurements (np.ndarray):
            Array of shape ``(m, 2)`` — all position measurements in the
            current frame.
        gamma_p2 (float):
            Position-gate threshold ``γ²_p`` (dimensionless).
        gamma_v (float):
            Velocity-gate threshold ``γᵥ`` in pixels/sec.
        T (float):
            Frame period in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            * ``A`` (shape ``m × n``, dtype bool) — binary validation matrix
            * ``D`` (shape ``m × n``, dtype float) — Mahalanobis distances;
              set to ``inf`` for pairs outside the gate.
    """
    m = len(measurements)
    n = len(tracks)
    A = np.zeros((m, n), dtype=bool)
    D = np.full((m, n), np.inf)

    for t_idx, track in enumerate(tracks):
        for j_idx, meas in enumerate(measurements):
            nu, S, d2 = track.innovation(meas)
            if d2 > gamma_p2:
                continue
            implied_speed = float(np.linalg.norm(nu)) / T
            if implied_speed > gamma_v:
                continue
            A[j_idx, t_idx] = True
            D[j_idx, t_idx] = d2

    return A, D


# ---------------------------------------------------------------------------
# Kusiak's clustering algorithm  (Paper Appendix C)
# ---------------------------------------------------------------------------

def _find_clusters(
    A: np.ndarray,
) -> list[tuple[list[int], list[int]]]:
    """Partition tracks and measurements into independent JPDAF clusters.

    Uses Kusiak's connected-component algorithm (Paper Appendix C, Ref. [28]).
    A cluster is a maximal subset of tracks and measurements such that at
    least one measurement validates at least one track in the subset.

    Measurements that validate no track, and tracks that validate no
    measurement, are returned as singleton clusters so that every entity
    is represented in the output.

    Parameters:
        A (np.ndarray):
            Binary validation matrix of shape ``(m, n)`` as returned by
            :func:`_build_validation_matrix`.

    Returns:
        list[tuple[list[int], list[int]]]:
            List of ``(track_indices, meas_indices)`` pairs, one per
            independent cluster.  Singletons with no validated pairs
            have an empty list for the other side.
    """
    m, n = A.shape
    # Union-Find on combined nodes: [0..n-1] = tracks, [n..n+m-1] = measurements
    parent = list(range(n + m))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for j in range(m):
        for t in range(n):
            if A[j, t]:
                _union(t, n + j)

    from collections import defaultdict
    groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for t in range(n):
        groups[_find(t)][0].append(t)
    for j in range(m):
        groups[_find(n + j)][1].append(j)

    # Keep only groups that have at least one track or one measurement
    result: list[tuple[list[int], list[int]]] = []
    seen_tracks: set[int] = set()
    seen_meas: set[int] = set()
    for t_list, j_list in groups.values():
        if t_list or j_list:
            result.append((t_list, j_list))
            seen_tracks.update(t_list)
            seen_meas.update(j_list)

    # Isolated tracks (no validated measurement) — each is its own cluster
    for t in range(n):
        if t not in seen_tracks:
            result.append(([t], []))
    # Isolated measurements (validate no track) — each is its own cluster
    for j in range(m):
        if j not in seen_meas:
            result.append(([], [j]))

    return result


# ---------------------------------------------------------------------------
# Feasible-event enumeration  (Paper Eqs. 11–14)
# ---------------------------------------------------------------------------

def _enumerate_events(
    cluster_A: np.ndarray,
    cluster_D: np.ndarray,
    gauss_log_likelihoods: np.ndarray,
    PD: float,
    lambda_c: float,
) -> list[tuple[dict[int, int], float]]:
    """Enumerate all feasible joint association events for one cluster.

    A feasible joint event assigns each measurement either to one track
    (if validated) or to clutter, with no two measurements sharing the
    same track.

    Joint event probability (Paper Eq. 11):

        ``P{θ|Zᵏ} ∝ ∏_j [λ⁻¹·f_tj(zⱼ)]^τⱼ · ∏_t PD^δt · (1-PD)^(1-δt)``

    Parameters:
        cluster_A (np.ndarray):
            Validation sub-matrix for this cluster, shape ``(m_c, n_c)``.
        cluster_D (np.ndarray):
            Mahalanobis distance sub-matrix, shape ``(m_c, n_c)``.
            (Unused in probability but kept for future reference.)
        gauss_log_likelihoods (np.ndarray):
            Pre-computed log Gaussian likelihoods ``ln f_tj``, shape
            ``(m_c, n_c)``.  Set to ``-inf`` for invalid pairs.
        PD (float):
            Per-target detection probability.
        lambda_c (float):
            Poisson clutter density (per px²).

    Returns:
        list[tuple[dict[int, int], float]]:
            List of ``(assignment, log_prob)`` pairs.

            * ``assignment`` maps local measurement index → local track
              index (or ``-1`` for clutter).
            * ``log_prob`` is the unnormalised log-probability.
    """
    m_c, n_c = cluster_A.shape
    log_PD = math.log(max(PD, 1e-300))
    log_1mPD = math.log(max(1.0 - PD, 1e-300))
    log_inv_lambda = -math.log(max(lambda_c, 1e-300))

    # For each measurement: list of valid track assignments + clutter (-1)
    meas_options: list[list[int]] = []
    for j in range(m_c):
        options = [-1]  # clutter always valid
        for t in range(n_c):
            if cluster_A[j, t]:
                options.append(t)
        meas_options.append(options)

    events: list[tuple[dict[int, int], float]] = []
    best_log_prob = -math.inf

    for assignment_tuple in product(*meas_options):
        # Feasibility: each track assigned at most once
        assigned_tracks = [t for t in assignment_tuple if t >= 0]
        if len(assigned_tracks) != len(set(assigned_tracks)):
            continue

        # Compute log probability of this joint event
        log_prob = 0.0
        detected_tracks: set[int] = set()

        for j, t in enumerate(assignment_tuple):
            if t >= 0:
                # Measurement j associated to track t
                log_prob += log_inv_lambda + gauss_log_likelihoods[j, t] + log_PD
                detected_tracks.add(t)
            # clutter measurement: contributes a factor of 1 (log 0 = no change)

        # Undetected tracks contribute (1 - P_D)
        for t in range(n_c):
            if t not in detected_tracks:
                log_prob += log_1mPD

        events.append(({j: t for j, t in enumerate(assignment_tuple)}, log_prob))

        if log_prob > best_log_prob:
            best_log_prob = log_prob

    # Prune improbable events for large clusters
    if m_c > 8:
        log_threshold = best_log_prob + math.log(_LARGE_CLUSTER_PRUNE)
        events = [(a, lp) for a, lp in events if lp >= log_threshold]

    return events


def _compute_betas(
    events: list[tuple[dict[int, int], float]],
    n_tracks: int,
    n_meas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginalise event probabilities to per-pair association probabilities.

    Paper Eqs. 14 and the β₀t definition.

    Parameters:
        events (list[tuple[dict[int, int], float]]):
            List of ``(assignment, log_prob)`` pairs from
            :func:`_enumerate_events`.
        n_tracks (int):
            Number of tracks in the cluster (n_c).
        n_meas (int):
            Number of measurements in the cluster (m_c).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            * ``beta`` (shape ``(m_c, n_c)``) — marginal probabilities
              ``β_jt``:  P(measurement j associated with track t).
            * ``beta0`` (shape ``(n_c,)``) — miss probabilities ``β₀t``:
              P(track t not associated with any measurement).
    """
    if not events:
        return np.zeros((n_meas, n_tracks)), np.ones(n_tracks)

    log_probs = np.array([lp for _, lp in events])
    # Numerically stable softmax-style normalisation
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()

    beta = np.zeros((n_meas, n_tracks))
    beta0 = np.zeros(n_tracks)

    for (assignment, _), p in zip(events, probs):
        detected: set[int] = set()
        for j, t in assignment.items():
            if t >= 0:
                beta[j, t] += p
                detected.add(t)
        for t in range(n_tracks):
            if t not in detected:
                beta0[t] += p

    return beta, beta0


# ---------------------------------------------------------------------------
# Track-management thresholds  (Paper Appendix D)
# ---------------------------------------------------------------------------

def _management_thresholds(
    lambda_c: float,
    lambda_n: float,
    pdelete: float,
    pconfirm: float,
) -> tuple[float, float, float]:
    """Compute confirmation threshold eta_c, deletion threshold eta_d, and initial score.

    Matches MATLAB VideoSpermTracker (Urbano et al. 2017):

    .. code-block:: matlab

        initialScore  = log(lam_n / lam_f)
        threshConfirm = log((1-Pdelete) / Pconfirm) + initialScore
        threshDelete  = log(Pdelete / (1 - Pconfirm))

    Deletion is applied as an **absolute** score threshold: delete when
    ``track.score < eta_d``.  Confirmation is also absolute: confirm when
    ``track.score > eta_c``.

    Parameters:
        lambda_c (float):
            Clutter density (per px²).  MATLAB ``lam_f``.
        lambda_n (float):
            New-target density (per px²).  MATLAB ``lam_n``.  Must be
            greater than ``lambda_c`` so that ``score_init > 0``.
        pdelete (float):
            Desired P(delete a true track by mistake).  MATLAB ``Pdelete = 1e-6``.
        pconfirm (float):
            Desired P(confirm a false track by mistake).  MATLAB ``Pconfirm = 1e-5``.

    Returns:
        tuple[float, float, float]:
            ``(eta_c, eta_d, score_init)`` — confirmation threshold,
            deletion threshold, and initial track score.
    """
    score_init = math.log(max(lambda_n / lambda_c, 1e-300))
    eta_c = math.log(max((1.0 - pdelete) / pconfirm, 1e-300)) + score_init
    eta_d = math.log(max(pdelete / (1.0 - pconfirm), 1e-300))
    return eta_c, eta_d, score_init


# ---------------------------------------------------------------------------
# Track-key helpers  (mirrors _sort.py)
# ---------------------------------------------------------------------------

def _sort_track_key(track_key: str) -> tuple[int, str]:
    """Build numeric-friendly sort key for track IDs shaped like ``t1``."""
    suffix = str(track_key)[1:]
    if suffix.isdigit():
        return (int(suffix), str(track_key))
    return (10 ** 9, str(track_key))


def _sorted_tracks(
    tracks: dict[str, dict[int, list[float]]],
) -> dict[str, dict[int, list[float]]]:
    """Sort track dictionary by track ID suffix (``t1``, ``t2``, ...)."""
    return {key: tracks[key] for key in sorted(tracks.keys(), key=_sort_track_key)}


def _resolve_tracking_frame_range(
    video_info: dict[str, Any],
    video_array: np.ndarray | None,
    raw_detections: dict[Any, Any],
) -> tuple[int, int] | None:
    """Resolve ``(initial_frame, number_frame_used)`` for tracking."""
    initial_frame = int(video_info.get("initial_frame", 0) or 0)
    number_frame_used = int(video_info.get("number_frame_used", 0) or 0)

    if number_frame_used <= 0 and isinstance(video_array, np.ndarray):
        number_frame_used = int(video_array.shape[0])

    if number_frame_used > 0:
        return initial_frame, number_frame_used

    frame_ids: list[int] = []
    for key in raw_detections.keys():
        try:
            frame_ids.append(int(str(key)))
        except (TypeError, ValueError):
            continue
    if not frame_ids:
        return None

    initial_frame = min(frame_ids)
    number_frame_used = max(frame_ids) - initial_frame + 1
    return initial_frame, number_frame_used


# ---------------------------------------------------------------------------
# Per-source JPDAF loop
# ---------------------------------------------------------------------------

def _run_jpdaf_on_source(
    raw_detections: dict[Any, Any],
    video_info: dict[str, Any],
    video_array: np.ndarray | None,
    width: int,
    height: int,
    source_name: str,
    T: float,
    sigma_n: float,
    q0: float,
    gamma_p2: float,
    gamma_v: float,
    lambda_c: float,
    lambda_n: float,
    PD: float,
    pdelete: float,
    pconfirm: float,
    redundant_dist: float,
    show_progress: bool,
    initial_frame: int = 0,
) -> tuple[dict[str, dict[str, list[float]]], int] | None:
    """Run the full JPDAF pipeline on one detection source.

    Parameters:
        raw_detections (dict[Any, Any]):
            Frame-keyed detection dict for this source.
        video_info (dict[str, Any]):
            ``casa['video']`` sub-dict (used for frame range resolution).
        video_array (np.ndarray | None):
            Optional video array (used for frame range resolution).
        width (int):
            Frame width in pixels.
        height (int):
            Frame height in pixels.
        source_name (str):
            Human-readable label shown in the progress bar.
        T (float):
            Frame period in seconds.
        sigma_n (float):
            Measurement noise std in pixels.
        q0 (float):
            Process noise spectral density in pixels²/sec³.
            Must be computed as ``(delta_v_px)² / T`` before calling.
        gamma_p2 (float):
            Position gate threshold (dimensionless).
        gamma_v (float):
            Velocity gate threshold in pixels/sec.
        lambda_c (float):
            Clutter density in per-px².
        lambda_n (float):
            New-target measurement density in per-px².
        PD (float):
            Detection probability.
        pdelete (float):
            P(delete a true track by mistake).  MATLAB ``Pdelete``.
        pconfirm (float):
            P(confirm a false track by mistake).  MATLAB ``Pconfirm``.
        redundant_dist (float):
            Distance below which the lower-scoring of two tracks is
            suppressed (pixels).
        show_progress (bool):
            Whether to display a tqdm progress bar.
        initial_frame (int):
            Offset (in frames) from the start of the analyzed video at which
            to begin tracking. Frames before this offset are skipped entirely.

    Returns:
        tuple[dict[str, dict[str, list[float]]], int] | None:
            ``(globalised_tracks, effective_frame_count)`` where
            ``globalised_tracks`` is ``{track_id: {frame_str: [cx, cy]}}``
            and ``effective_frame_count`` is the number of frames actually
            processed. Returns ``None`` if the frame range cannot be
            resolved or ``initial_frame`` exceeds the video length.
    """
    frame_range = _resolve_tracking_frame_range(
        video_info=video_info if isinstance(video_info, dict) else {},
        video_array=video_array if isinstance(video_array, np.ndarray) else None,
        raw_detections=raw_detections,
    )
    if frame_range is None:
        return None

    video_initial_frame, number_frame_used = frame_range

    start_offset = max(0, int(initial_frame))
    if start_offset >= number_frame_used:
        return None

    effective_initial_frame = video_initial_frame + start_offset
    number_frame_used = number_frame_used - start_offset

    F, H, N, Q0 = _build_matrices(T, sigma_n, q0)
    eta_c, eta_d, score_init = _management_thresholds(lambda_c, lambda_n, pdelete, pconfirm)

    _KalmanTrack._count = 0
    active_tracks: list[_KalmanTrack] = []
    track_history: dict[str, dict[int, list[float]]] = {}

    for local_idx in _progress_bar(
        range(number_frame_used),
        total=number_frame_used,
        desc=f"Tracking jpdaf ({source_name})",
        unit="frame",
        leave=True,
        enabled=show_progress,
    ):
        global_frame = effective_initial_frame + local_idx
        frame_data = _get_frame_detections(raw_detections, global_frame)

        # ---- Parse measurements ----
        raw_meas: list[list[float]] = []
        for det in frame_data:
            pt = _detection_to_centroid(det, width=width, height=height)
            if pt is not None:
                raw_meas.append(pt)
        measurements = np.array(raw_meas, dtype=float) if raw_meas else np.empty((0, 2))

        # ---- Predict all active tracks ----
        for track in active_tracks:
            track.predict()

        if len(measurements) == 0:
            # No measurements: all tracks score a miss
            for track in active_tracks:
                track.score_step(None, updated=False, PD=PD, lambda_c=lambda_c)
        else:
            # ---- Build validation matrix ----
            A, D = _build_validation_matrix(
                tracks=active_tracks,
                measurements=measurements,
                gamma_p2=gamma_p2,
                gamma_v=gamma_v,
                T=T,
            )

            # ---- Pre-compute Gaussian log-likelihoods ----
            m_total = len(measurements)
            n_total = len(active_tracks)
            log_lik = np.full((m_total, n_total), -np.inf)
            for t_idx, track in enumerate(active_tracks):
                for j_idx, meas in enumerate(measurements):
                    if A[j_idx, t_idx]:
                        nu, S, d2 = track.innovation(meas)
                        try:
                            sign, log_det = np.linalg.slogdet(2.0 * math.pi * S)
                            if sign > 0:
                                log_lik[j_idx, t_idx] = -0.5 * (d2 + log_det)
                        except np.linalg.LinAlgError:
                            pass

            # ---- Cluster and run JPDAF per cluster ----
            clusters = _find_clusters(A)
            meas_updated = np.zeros(m_total, dtype=bool)

            for t_indices, j_indices in clusters:
                if not t_indices:
                    # Unvalidated measurements — handled below for track initiation
                    continue
                if not j_indices:
                    # Isolated tracks with no in-gate measurement: score miss
                    for t_idx in t_indices:
                        active_tracks[t_idx].score_step(
                            None, updated=False, PD=PD, lambda_c=lambda_c
                        )
                    continue

                cluster_tracks = [active_tracks[t] for t in t_indices]
                cluster_A = A[np.ix_(j_indices, t_indices)]
                cluster_D = D[np.ix_(j_indices, t_indices)]
                cluster_log_lik = log_lik[np.ix_(j_indices, t_indices)]

                events = _enumerate_events(
                    cluster_A, cluster_D, cluster_log_lik, PD, lambda_c
                )
                beta, beta0 = _compute_betas(
                    events, n_tracks=len(t_indices), n_meas=len(j_indices)
                )

                # Update each track in the cluster
                for local_t, t_idx in enumerate(t_indices):
                    track = active_tracks[t_idx]
                    # Collect validated measurements and their betas
                    betas_j: list[float] = []
                    residuals_j: list[np.ndarray] = []
                    for local_j, j_idx in enumerate(j_indices):
                        if cluster_A[local_j, local_t]:
                            b = float(beta[local_j, local_t])
                            nu, _, _ = track.innovation(measurements[j_idx])
                            betas_j.append(b)
                            residuals_j.append(nu)

                    if betas_j and (1.0 - float(beta0[local_t])) > 1e-9:
                        pseudo = track.update(betas_j, residuals_j, float(beta0[local_t]))
                        track.score_step(pseudo, updated=True, PD=PD, lambda_c=lambda_c)
                    else:
                        track.score_step(None, updated=False, PD=PD, lambda_c=lambda_c)

                # Mark cluster measurements as updated (used for new-track detection)
                for j_idx in j_indices:
                    meas_updated[j_idx] = True

            # ---- Initiate tentative tracks on unassociated measurements ----
            for j_idx in range(m_total):
                if not meas_updated[j_idx]:
                    new_track = _KalmanTrack(
                        measurement=measurements[j_idx],
                        F=F, H=H, N=N, Q0=Q0,
                        initial_score=score_init,
                    )
                    active_tracks.append(new_track)

        # ---- Confirm tentative tracks (before storing, so frame of confirmation
        #      is included in the track history)  ----
        for track in active_tracks:
            if not track.confirmed and track.score > eta_c:
                track.confirmed = True

        # ---- Store confirmed track positions ----
        for track in active_tracks:
            if track.confirmed:
                key = f"t{track.track_id}"
                track_history.setdefault(key, {})[local_idx] = [
                    float(track.x[0]), float(track.x[1])
                ]

        # ---- Delete tracks whose absolute score has fallen below eta_d.
        #      MATLAB: TrackFile(trk,2) < threshDelete  (absolute threshold).
        active_tracks = [t for t in active_tracks if t.score >= eta_d]

        # ---- Suppress redundant tracks ----
        to_delete: set[int] = set()
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                ti, tj = active_tracks[i], active_tracks[j]
                dist = float(np.linalg.norm(ti.x[:2] - tj.x[:2]))
                if dist < redundant_dist:
                    loser = i if ti.score < tj.score else j
                    to_delete.add(loser)
        active_tracks = [t for idx, t in enumerate(active_tracks) if idx not in to_delete]

    # ---- Globalise frame indices ----
    sorted_local = _sorted_tracks(track_history)
    globalised: dict[str, dict[str, list[float]]] = {}
    for track_id, local_track in sorted_local.items():
        global_track: dict[str, list[float]] = {}
        for local_idx, coords in local_track.items():
            global_frame = effective_initial_frame + int(local_idx)
            global_track[str(global_frame)] = coords
        globalised[track_id] = global_track

    return globalised, int(number_frame_used)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def jpdaf(
    casa: dict[str, Any],
    skip_gt: bool = False,
    frame_rate: float | None = None,
    initial_frame: int = 0,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Track detections using the JPDAF and store per-source results.

    Implements the Joint Probabilistic Data Association Filter described in
    Urbano et al. (2017), IEEE Transactions on Medical Imaging 36(3), 792–801.
    Each detection source (groundtruth and/or predicted) is tracked
    independently.  All algorithm parameters are taken directly from the
    paper and scaled to pixel space using ``casa['meta']['um_per_px']``.

    Parameters:
        casa (dict[str, Any]):
            Session dictionary (standard pycasa Casa schema).
        skip_gt (bool, optional):
            If ``False`` (default), run tracking on both ``groundtruth``
            detections and the active predicted detection method when
            available.  If ``True``, skip groundtruth and only track
            predicted detections.
        frame_rate (float | None, optional):
            Frames per second.  When ``None``, the value is read from
            ``casa['meta']['sampling_rate']``.  The frame period
            ``T = 1 / frame_rate`` is used in the CWNA motion model.
        initial_frame (int, optional):
            Offset (in frames) from the start of the analyzed video at which
            to begin tracking. Frames before this offset are skipped entirely
            so no track history is accumulated from them. Default ``0``.
        show_progress (bool, optional):
            If ``True``, display the shared pycasa progress bar during
            per-frame tracking.
        verbose (bool, optional):
            If ``True``, print concise start/end summaries.  Warnings
            are not affected by this flag.

    Returns:
        dict[str, Any]:
            Updated ``casa`` with per-source tracks stored in
            ``casa['tracks']['jpdaf'][source]`` and invocation metadata
            in ``casa['meta']['last_tracking']``.

    Raises:
        ValueError:
            If ``frame_rate`` cannot be determined from parameters or
            session metadata, or if video width/height cannot be resolved.

    Notes:
        - Track output schema is identical to the SORT backend:
          ``{track_id: {frame_str: [cx, cy]}}``.
        - All paper parameters (σₙ, q̃₀, γᵥ, λ, etc.) are converted from
          µm to pixels using ``um_per_px``.  If calibration is absent a
          warning is issued and approximate pixel-space defaults are used.
        - The confirmation threshold ηc and deletion threshold ηd are
          derived from ``PDT = 0.95`` and ``PCF = 0.05`` (Paper Appendix D).
        - ``casa['tracks']`` is cleared and rebuilt on each call.

    Examples:
        >>> import pycasa as pc
        >>> session = pc.Casa()
        >>> session = session.tracking.jpdaf(skip_gt=False)
    """
    _msg_yellow(
        "JPDAF tracker based on: Urbano et al. (2017), "
        "'Automatic Tracking and Motility Analysis of Human Sperm in "
        "Time-Lapse Images', IEEE TMI 36(3), 792-801."
    )

    casa = _ensure_casa(casa)

    # ---- Resolve frame rate ----
    if frame_rate is None:
        meta_fps = casa.get("meta", {}).get("sampling_rate")
        if meta_fps is not None:
            try:
                frame_rate = float(meta_fps)
            except (TypeError, ValueError):
                frame_rate = None
    if frame_rate is None or not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(
            "frame_rate must be a positive finite number.  "
            "Pass it explicitly or set casa['meta']['sampling_rate']."
        )
    T = 1.0 / frame_rate

    # ---- Unit scaling ----
    um_per_px: float | None = None
    meta_umpp = casa.get("meta", {}).get("um_per_px")
    if meta_umpp is not None:
        try:
            v = float(meta_umpp)
            if v > 0.0:
                um_per_px = v
        except (TypeError, ValueError):
            pass

    if um_per_px is not None:
        sigma_n = _SIGMA_N_UM / um_per_px
        gamma_v = _GAMMA_V_UM / um_per_px
        # CWNA spectral density: qIntensity = deltaV² / T  (MATLAB VideoSpermTracker line ~144)
        # _DELTA_V_UM is the velocity-uncertainty amplitude; squaring and dividing by T
        # converts it to the power-spectral-density form that _build_matrices expects.
        delta_v = _DELTA_V_UM / um_per_px           # px/sec
        q0 = (delta_v ** 2) / T                     # px²/sec³
        lambda_c = _LAMBDA_UM2 * (um_per_px ** 2)   # lam_f in px²
        lambda_n = _LAMBDA_N_UM2 * (um_per_px ** 2) # lam_n in px²
        redundant_dist = _REDUNDANT_UM / um_per_px
    else:
        _warn_yellow(
            "um_per_px not set; using pixel-space JPDAF defaults "
            "(approximate for 0.857 µm/px microscopy)."
        )
        sigma_n = 2.5
        gamma_v = 350.0
        delta_v_fallback = 23.5                      # px/sec  (≈ 20 µm/s at 0.857 µm/px)
        q0 = (delta_v_fallback ** 2) / T             # px²/sec³
        lambda_c = 7.3e-7                            # per px²  (1e-6 µm⁻² × 0.857² µm²/px²)
        lambda_n = 7.3e-6                            # per px²  (1e-5 µm⁻² × 0.857² µm²/px²)
        redundant_dist = 0.012

    # ---- Resolve detection sources ----
    detections_root = casa.get("detections", {})
    if not isinstance(detections_root, dict):
        detections_root = {}

    active_predicted_method = _resolve_active_predicted_detection_method(detections_root)
    gt_detections = detections_root.get("groundtruth", {})
    has_groundtruth = isinstance(gt_detections, dict) and bool(gt_detections)
    predicted_detections: dict[str, Any] = {}
    if active_predicted_method is not None:
        candidate = detections_root.get(active_predicted_method, {})
        if isinstance(candidate, dict):
            predicted_detections = candidate
    has_detections = bool(predicted_detections)

    tracks_root = casa.setdefault("tracks", {})
    existing_backends = [
        k
        for k in _KNOWN_TRACKING_BACKENDS
        if isinstance(tracks_root.get(k), dict)
    ]
    if existing_backends:
        _warn_yellow(
            f"Previous tracking result overwritten "
            f"({', '.join(existing_backends)} -> jpdaf)."
        )
    _clear_backend_tracks(tracks_root)
    tracks_root["jpdaf"] = {}
    jpdaf_root = tracks_root["jpdaf"]

    source_order: list[str] = []
    reason: str | None = None

    if skip_gt:
        if has_detections and active_predicted_method is not None:
            source_order.append(active_predicted_method)
        else:
            reason = "missing_detections_skip_gt" if has_groundtruth else "missing_detections_and_groundtruth"
            if has_groundtruth:
                _warn_yellow("No detections found, tracking skipped because skip_gt=True")
            else:
                _warn_yellow("No detections/GT found, either import GT or run detection")
    else:
        if has_groundtruth and has_detections and active_predicted_method is not None:
            print("GT and detections found, tracking will run on both")
            source_order.extend(["groundtruth", active_predicted_method])
        elif has_groundtruth:
            _warn_yellow("No detections found, tracking will only run on GT")
            source_order.append("groundtruth")
        elif has_detections and active_predicted_method is not None:
            _warn_yellow("No GT found, tracking will only run on detections")
            source_order.append(active_predicted_method)
        else:
            reason = "missing_detections_and_groundtruth"
            _warn_yellow("No detections/GT found, either import GT or run detection")

    if reason is not None:
        casa["meta"]["last_tracking"] = {
            "backend": "jpdaf",
            "detection_method": None,
            "sources_requested": [],
            "sources_processed": [],
            "per_source": {},
            "skip_gt": bool(skip_gt),
            "frame_rate": float(frame_rate),
            "sigma_n": float(sigma_n),
            "gamma_v": float(gamma_v),
            "um_per_px": um_per_px,
            "initial_frame": int(max(0, initial_frame)),
            "input_frames": 0,
            "output_tracks": 0,
            "skipped": True,
            "reason": reason,
        }
        return casa

    # ---- Resolve video dimensions ----
    video_info = casa.get("video", {})
    meta_info = casa.get("meta", {})
    video_array = video_info.get("array") if isinstance(video_info, dict) else None
    width = int(meta_info.get("width") or 0)
    height = int(meta_info.get("height") or 0)
    if isinstance(video_array, np.ndarray):
        _, video_height, video_width = _ensure_video_dimensions(video_array)
        if width <= 0:
            width = int(video_width)
        if height <= 0:
            height = int(video_height)
    if width <= 0 or height <= 0:
        raise ValueError(
            "Tracking requires video width/height in casa['meta'] or a valid video array."
        )

    if verbose and source_order:
        print(
            f"Running JPDAF tracking on frames "
            f"(sources={', '.join(source_order)}, fps={frame_rate:.2f})..."
        )

    per_source: dict[str, dict[str, Any]] = {}
    total_output_tracks = 0
    total_input_frames = 0
    processed_sources: list[str] = []

    for source_name in source_order:
        raw_detections = detections_root.get(source_name, {})
        if not isinstance(raw_detections, dict) or not raw_detections:
            jpdaf_root[source_name] = {}
            per_source[source_name] = {
                "input_frames": 0,
                "output_tracks": 0,
                "average_track_length": None,
                "skipped": True,
                "reason": "missing_detections",
            }
            continue

        run_result = _run_jpdaf_on_source(
            raw_detections=raw_detections,
            video_info=video_info if isinstance(video_info, dict) else {},
            video_array=video_array if isinstance(video_array, np.ndarray) else None,
            width=width,
            height=height,
            source_name=source_name,
            T=T,
            sigma_n=sigma_n,
            q0=q0,
            gamma_p2=_GAMMA_P2,
            gamma_v=gamma_v,
            lambda_c=lambda_c,
            lambda_n=lambda_n,
            PD=_PD,
            pdelete=_PDELETE,
            pconfirm=_PCONFIRM,
            redundant_dist=redundant_dist,
            show_progress=show_progress,
            initial_frame=initial_frame,
        )

        if run_result is None:
            frame_range = _resolve_tracking_frame_range(
                video_info=video_info if isinstance(video_info, dict) else {},
                video_array=video_array if isinstance(video_array, np.ndarray) else None,
                raw_detections=raw_detections,
            )
            if (
                frame_range is not None
                and max(0, int(initial_frame)) >= frame_range[1]
            ):
                reason_text = "initial_frame_exceeds_video_length"
            else:
                reason_text = "invalid_frame_range"
            jpdaf_root[source_name] = {}
            per_source[source_name] = {
                "input_frames": 0,
                "output_tracks": 0,
                "average_track_length": None,
                "skipped": True,
                "reason": reason_text,
            }
            continue

        globalised, n_frames = run_result

        jpdaf_root[source_name] = globalised
        processed_sources.append(source_name)
        output_tracks = len(globalised)
        total_output_tracks += output_tracks
        total_input_frames += n_frames

        avg_len: float | None = (
            float(np.mean([len(v) for v in globalised.values() if isinstance(v, dict)]))
            if globalised else None
        )
        per_source[source_name] = {
            "input_frames": n_frames,
            "output_tracks": output_tracks,
            "average_track_length": avg_len,
            "skipped": False,
        }

    primary_source: str | None = None
    if active_predicted_method is not None and active_predicted_method in jpdaf_root:
        primary_source = active_predicted_method
    elif "groundtruth" in jpdaf_root:
        primary_source = "groundtruth"
    elif jpdaf_root:
        primary_source = sorted(jpdaf_root.keys())[0]

    all_skipped = (
        all(bool(s.get("skipped", True)) for s in per_source.values())
        if per_source else True
    )

    if verbose:
        print(
            f"JPDAF summary: sources_processed={processed_sources}, "
            f"input_frames={total_input_frames}, "
            f"output_tracks={total_output_tracks}"
        )
        for sn in processed_sources:
            si = per_source.get(sn, {})
            avg = si.get("average_track_length")
            avg_txt = f"{float(avg):.2f}" if isinstance(avg, (int, float)) else "None"
            print(f"- {sn}: tracks={si.get('output_tracks')}, average_track_length={avg_txt}")

    casa["meta"]["last_tracking"] = {
        "backend": "jpdaf",
        "detection_method": primary_source,
        "sources_requested": list(source_order),
        "sources_processed": processed_sources,
        "per_source": per_source,
        "skip_gt": bool(skip_gt),
        "frame_rate": float(frame_rate),
        "sigma_n": float(sigma_n),
        "gamma_v": float(gamma_v),
        "um_per_px": um_per_px,
        "initial_frame": int(max(0, initial_frame)),
        "input_frames": int(total_input_frames),
        "output_tracks": int(total_output_tracks),
        "skipped": bool(all_skipped),
        "reason": "all_sources_skipped" if all_skipped else None,
    }
    return casa
