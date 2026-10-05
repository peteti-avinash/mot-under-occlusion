"""Cost matrices, Mahalanobis gating and Hungarian assignment.

Gating matters more than the solver choice. ``linear_sum_assignment`` always
returns a complete pairing, so without a gate the tracker happily hands a track
to a detection 300 px away and instantly manufactures an ID switch. Every
assignment in this project is therefore filtered twice: once by the track's own
prediction covariance (Mahalanobis / chi-square) and once by a fused
motion+appearance cost threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import CHI2_95_DF4


def iou_matrix(track_boxes, det_boxes):
    """Pairwise IoU between two ``(N, 4)`` / ``(M, 4)`` arrays of ``[x, y, w, h]``."""
    a = np.asarray(track_boxes, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inside = (tl < br).all(axis=2)
    inter = np.prod(np.clip(br - tl, 0.0, None), axis=2) * inside
    area_a = np.prod(a[:, 2:], axis=1)
    area_b = np.prod(b[:, 2:], axis=1)
    denom = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(denom, 1e-6, None)).astype(np.float32)


def iou_cost(track_boxes, det_boxes):
    """``1 - IoU``: a minimisation-friendly cost."""
    return 1.0 - iou_matrix(track_boxes, det_boxes)


def fuse_score(cost, det_scores, alpha=1.0):
    """Bias the cost against low-confidence detections.

    ``cost' = cost + alpha * (1 - score)``. Motivation: when two pedestrians
    overlap, the detector's confidence often alternates between them frame to
    frame. Folding confidence into the cost stops the tracker from bouncing the
    track between the two candidates just because confidence flipped.
    """
    if cost.size == 0:
        return cost
    scores = np.asarray(det_scores, dtype=np.float32).reshape(1, -1)
    return cost + alpha * (1.0 - scores)


def mahalanobis_distance(kf, track_states, track_covs, det_measurements):
    """Per-track Mahalanobis distance from predicted mean to each detection.

    Returns ``(D, gating)`` where ``D`` is the raw distance and ``gating`` is the
    distance normalised by the 95% chi-square bound (values > 1 are rejected).

    Computing this per track (rather than from one pooled covariance over all
    tracks) is what makes the gate meaningful: two tracks have very different
    uncertainties depending on their recent motion history.

    The track states are the full ``(8, 1)`` SORT states; projection into
    measurement space happens here through the filter, which is why ``kf`` is a
    required argument rather than being inferred.
    """
    z = np.asarray(det_measurements, dtype=np.float32).reshape(-1, 4)
    n = len(track_states)
    m = z.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32), np.zeros((n, m), dtype=np.float32)

    bound = float(CHI2_95_DF4)
    D = np.zeros((n, m), dtype=np.float64)
    for i, (state, cov) in enumerate(zip(track_states, track_covs)):
        d2 = kf.gating_distance(np.asarray(state).reshape(8, 1),
                                np.asarray(cov).reshape(8, 8), z)
        D[i] = np.sqrt(np.clip(np.asarray(d2, dtype=np.float64), 0.0, None))
    gating = D / np.sqrt(bound)
    return D.astype(np.float32), gating.astype(np.float32)


def linear_assignment(cost, max_distance):
    """Gated Hungarian assignment.

    Returns ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is a
    list of ``[row, col]`` index pairs.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        return [], np.array([], dtype=int), np.array([], dtype=int)
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], np.arange(n), np.arange(m)

    INF = 1e5
    gated = np.where(cost <= max_distance, cost, INF)
    # Rows/columns with no admissible pair would otherwise be arbitrarily paired
    # with an INF cost and could shadow a valid pair on the other axis.
    feasible = gated < INF
    gated = np.where(feasible.any(axis=1, keepdims=True), gated, gated + INF)
    gated = np.where(feasible.any(axis=0, keepdims=True), gated, gated + INF)

    row_ind, col_ind = linear_sum_assignment(gated)

    matches = []
    used_rows, used_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if not feasible[r, c]:
            continue
        matches.append([int(r), int(c)])
        used_rows.add(int(r))
        used_cols.add(int(c))

    unmatched_a = np.array([i for i in range(n) if i not in used_rows], dtype=int)
    unmatched_b = np.array([j for j in range(m) if j not in used_cols], dtype=int)
    return matches, unmatched_a, unmatched_b


def fuse_cost(iou_cost_mat, appearance_cost=None, appearance_weight=0.0):
    """Blend IoU and appearance costs into the final matching cost.

    ``cost = appearance_weight * cos_dist + (1 - appearance_weight) * iou_cost``

    Both inputs must be *costs* (lower is better). A Mahalanobis distance is not
    a similarity and must not be passed here: subtracting it from 1 as an earlier
    version of this function did inverts its meaning and makes implausible pairs
    look cheap.

    With ``appearance_weight = 0`` this is pure IoU (motion only, i.e. SORT).
    Raising it lets appearance dominate once tracks are long-lived, which is
    where pure IoU breaks down: two people who swap screen position during an
    overlap still overlap well with each other's old boxes.
    """
    iou_cost_mat = np.asarray(iou_cost_mat, dtype=np.float32)
    if iou_cost_mat.size == 0:
        return iou_cost_mat
    a = float(appearance_weight)
    a = min(max(a, 0.0), 1.0)
    if appearance_cost is None or a == 0.0:
        return iou_cost_mat
    appearance_cost = np.asarray(appearance_cost, dtype=np.float32)
    if appearance_cost.shape != iou_cost_mat.shape:
        raise ValueError(
            f"appearance cost shape {appearance_cost.shape} does not match "
            f"IoU cost shape {iou_cost_mat.shape}"
        )
    return a * appearance_cost + (1.0 - a) * iou_cost_mat


def cosine_distance_matrix(a, b):
    """``(1 - cosine_similarity)`` between two batches of L2-normalised vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim == 2 else 0,
                         b.shape[0] if b.ndim == 2 else 0), dtype=np.float32)
    return 1.0 - (a @ b.T).astype(np.float32)