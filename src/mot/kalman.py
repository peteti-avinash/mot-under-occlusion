"""Kalman filter for the SORT-family trackers.

State is the SORT state ``[cx, cy, s, r, vx, vy, vs, vr]`` where ``(cx, cy)``
is the bbox centre in pixels, ``s`` the area, ``r`` the aspect ratio and
``(vx, vy, vs, vr)`` the per-frame velocities. Motion is approximated as linear
(constant velocity): cheap, and good enough that the Hungarian assignment can
operate on the resulting Mahalanobis distances.

The covariance after a single update is known to under-estimate uncertainty,
which causes over-confident predictions during occlusion. The process noise is
therefore scaled by the target's own height and floored, so a coasting track's
gate widens exactly when it should, and the filter behaves the same for a 40 px
child and a 400 px adult.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

#: 95% chi-square bound for 4 degrees of freedom (used by the association gate).
CHI2_95_DF4 = 9.4877


class KalmanFilter:
    """Constant-velocity Kalman filter over bbox state."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Process noise. Values follow the SORT paper: spatial position noise is
        # large, velocity/appearance noise much smaller.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._min_position_noise = 0.5
        self._min_velocity_noise = 0.05

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def box_height(area, aspect):
        """Pixel height implied by a Kalman ``(area, aspect)`` pair.

        The noise model is scaled by the target's height rather than by its
        area. Scaling by area (state index 2) is a trap: a pedestrian of
        88x262 px has an area of ~23000, so an ``area/20`` noise term produces a
        velocity variance around 1e6 and makes the filter effectively blind.
        """
        area = max(float(area), 1e-6)
        aspect = max(float(aspect), 1e-6)
        return float(np.sqrt(area / aspect))

    def measurement_std(self, mean):
        """Measurement noise for ``[cx, cy, area, aspect]`` from a state mean.

        These are *relative* standard deviations. An absolute noise floor would be
        meaningless here: an absolute 0.1 px tolerance on the area of a
        23000 px^2 pedestrian is far tighter than any detector, so a Mahalanobis
        gate built on it rejects every candidate. Scaling each component by its
        own magnitude keeps the gate dimensionless.
        """
        h = self.box_height(mean[2, 0], mean[3, 0])
        area = max(abs(float(mean[2, 0])), 1.0)
        aspect = max(abs(float(mean[3, 0])), 1e-3)
        return np.array([max(0.05 * h, 1.0),
                         max(0.05 * h, 1.0),
                         max(0.10 * area, 1.0),
                         max(0.10 * aspect, 1e-3)],
                        dtype=np.float32)

    def initiate(self, measurement):
        mean_pos = np.asarray(measurement, dtype=np.float32).reshape(4)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = self.box_height(mean_pos[2], mean_pos[3])
        pos_std = max(self._min_position_noise, self._std_weight_position * h)
        vel_std = max(self._min_velocity_noise, self._std_weight_velocity * h)
        std = [
            2 * pos_std,           # cx
            2 * pos_std,           # cy
            1e-2,                  # area (fractional; see measurement_std)
            pos_std,               # aspect
            10 * vel_std,          # vx
            10 * vel_std,          # vy
            1e-4,                  # vs
            10 * vel_std,          # vr
        ]
        return mean.reshape(8, 1), np.diag(np.square(np.asarray(std, dtype=np.float32)))

    def predict(self, mean, covariance):
        # Process noise scaled by the target's own height, and floored so that a
        # momentarily constant aspect ratio cannot collapse the velocity
        # variance to zero (which makes the filter too stiff to follow motion).
        # These weights are the ones SORT uses for its F matrices.
        h = self.box_height(mean[2, 0], mean[3, 0])
        pos_std = max(self._min_position_noise, self._std_weight_position * h)
        vel_std = max(self._min_velocity_noise, self._std_weight_velocity * h)
        std = np.r_[
            np.full(2, pos_std),          # cx, cy
            np.full(2, 1e-2),             # area, aspect (relative)
            np.full(2, 10.0 * vel_std),   # vx, vy
            10.0 * 1e-2,                 # vs
            10.0 * vel_std,              # vr
        ].astype(np.float32)
        motion_cov = np.diag(np.square(std))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(self, mean, covariance):
        if np.asarray(mean).ndim == 3:
            out_m, out_c = [], []
            for i in range(np.asarray(mean).shape[0]):
                m, c = self.predict(mean[i], covariance[i])
                out_m.append(m)
                out_c.append(c)
            return np.asarray(out_m), np.asarray(out_c)
        return self.predict(mean, covariance)

    def project(self, mean, covariance):
        innovation_cov = np.diag(np.square(self.measurement_std(mean)))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4, 1)
        innovation_cov = np.diag(np.square(self.measurement_std(mean)))
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = _cholesky_factor(projected_cov, sigma=1e-4)
        kalman_gain = solve_triangular(
            chol_factor.T,
            solve_triangular(
                chol_factor,
                (covariance @ self._update_mat.T).T,
                lower=False,
                trans="T",
            ),
            lower=False,
            trans="T",
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        # Joseph form rather than ``P - K S K^T``. The short form subtracts a
        # large matrix from a smaller one and is not guaranteed to stay
        # positive semi-definite; once it does not, every later Cholesky of the
        # projected covariance raises ``LinAlgError`` and the tracker dies
        # mid-sequence.
        eye_minus_kh = np.eye(covariance.shape[0]) - kalman_gain @ self._update_mat
        new_covariance = (eye_minus_kh @ covariance @ eye_minus_kh.T
                          + kalman_gain @ innovation_cov @ kalman_gain.T)
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements):
        mean, covariance = self.project(mean, covariance)
        measurements = np.asarray(measurements, dtype=np.float32).reshape(-1, 4)
        if measurements.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        chol_factor = _cholesky_factor(covariance, sigma=1e-4)
        d = measurements - mean.T
        z = solve_triangular(chol_factor, d.T, lower=False, trans="T")
        return np.sum(z * z, axis=0)

    def velocity_covariance(self, covariance):
        """Velocity block of the covariance, used for adaptive gating."""
        return np.asarray(covariance)[4:, 4:]


def self_std(weight, x):
    return weight * x


def _cholesky_factor(matrix, sigma=0.1, tries=4):
    """Cholesky with an escalating jitter fallback for near-singular matrices.

    Sequence-dependent covariance matrices occasionally drift to
    non-positive-definite after many missed frames; adding jitter is cheaper
    and safer than silently returning garbage distances. Jitter only repairs
    matrices that are *positive semi*-definite, so the last resort is an
    eigenvalue clip, which is guaranteed to yield a factor.

    Returns the upper-triangular factor ``L`` with ``P ~= L.T @ L``. There is no
    ``lower`` return value: every path goes through ``np.linalg.cholesky``, which
    is upper triangular by definition, so callers always use ``lower=False``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    scale = float(np.mean(np.diag(matrix)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    eye = np.eye(matrix.shape[0])
    result = matrix
    for i in range(tries):
        try:
            return np.linalg.cholesky(result).T
        except np.linalg.LinAlgError:
            sigma_i = sigma ** (i + 1) * scale
            result = result + sigma_i * eye
    values, vectors = np.linalg.eigh(result)
    values = np.clip(values, sigma * scale, None)
    repaired = vectors @ np.diag(values) @ vectors.T
    return np.linalg.cholesky(repaired).T


def convert_bbox_to_z(bbox):
    """``[x, y, w, h]`` -> Kalman measurement ``[cx, cy, area, aspect]``."""
    w = max(float(bbox[2]), 1e-6)
    h = max(float(bbox[3]), 1e-6)
    return np.array(
        [float(bbox[0]) + w / 2.0, float(bbox[1]) + h / 2.0, w * h, w / h],
        dtype=np.float32,
    )


def convert_z_to_bbox(z):
    """Kalman measurement ``[cx, cy, area, aspect]`` -> ``[x, y, w, h]``."""
    z = np.asarray(z, dtype=np.float32).reshape(4)
    area = max(float(z[2]), 1e-9)
    aspect = max(float(z[3]), 1e-9)
    w = float(np.sqrt(area * aspect))
    h = float(area / max(w, 1e-9))
    return np.array([float(z[0]) - w / 2.0, float(z[1]) - h / 2.0, w, h],
                    dtype=np.float32)