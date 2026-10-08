"""Appearance-based tracker with explicit occlusion recovery.

Two ideas separate this from plain SORT, and both target the specific failure
the assignment asks about (people crossing paths in a retail ceiling view):

1. **Cascade matching by age.** Tracks are matched oldest-first, and the cost
   blends IoU with appearance using a weight that grows with the track's
   uncertainty. Matching the most certain tracks first prevents them from
   stealing a detection that a more-certain track deserves — the classic cause
   of "identity theft" in crowded scenes.

2. **A two-tier re-identification stage.** After the normal cascade, any
   detection still unmatched is offered to tracks that have been *coasting
   through an occlusion* (tracks in ``Lost`` state). Such a match must clear a
   tight appearance threshold plus a looser spatial gate that widens with the
   coasting duration. This is the part that lets a person who disappeared for
   20 frames behind another person keep their ID instead of spawning a new one.

A per-identity gallery of recent embeddings (rather than only the latest) is
maintained because a person changes appearance as they turn; matching against
the best-matching recent pose is more reliable than the most recent frame.
"""

from __future__ import annotations

import numpy as np

from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment

from .associate import (
    cosine_distance_matrix,
    fuse_cost,
    iou_cost,
)
from .kalman import KalmanFilter, convert_bbox_to_z
from .track import Detection, Track, TrackState


class OcclusionAwareTracker:
    """DeepSORT-style tracker with gallery-based occlusion recovery."""

    def __init__(self,
                 max_age=30,
                 min_hits=3,
                 det_conf_threshold=0.4,
                 n_init=3,
                 appearance_weight=0.6,
                 iou_threshold=0.7,
                 max_iou_distance=0.9,
                 max_appearance_distance=0.35,
                 gate_multiplier=1.6,
                 reid_max_age=45,
                 reid_appearance_threshold=0.32,
                 reid_gate_slope=0.02,
                 reid_max_gate=12.0,
                 match_gate=9.0,
                 reid_min_iou=0.0,
                 gallery_size=30,
                 frame_rate=30,
                 enable_reid=True,
                 cascade=True):
        self.max_age = max_age
        self.min_hits = min_hits
        self.n_init = n_init
        self.det_conf_threshold = det_conf_threshold
        # Ablation switches. ``enable_reid`` drops the Lost-track recovery
        # stage. ``cascade=False`` replaces the oldest-first sequential
        # association with one global Hungarian assignment over every track at
        # once, keeping the same costs and gates, so the ablation isolates the
        # cascade *ordering* rather than the cost function.
        self.enable_reid = enable_reid
        self.cascade = cascade

        # Matching weights.
        self.appearance_weight = appearance_weight
        self.iou_threshold = iou_threshold
        self.max_iou_distance = max_iou_distance
        self.max_appearance_distance = max_appearance_distance

        # Adaptive gating: the distance threshold grows with the track's own
        # velocity covariance, so a moving track tolerates a bigger jump.
        self.gate_multiplier = gate_multiplier

        # Re-identification of coasting (Lost) tracks.
        self.reid_max_age = reid_max_age
        self.reid_appearance_threshold = reid_appearance_threshold
        self.reid_gate_slope = reid_gate_slope
        self.reid_max_gate = reid_max_gate
        self.match_gate = match_gate
        # Overlap is deliberately *not* required here. Someone reappearing after a
        # long occlusion frequently has zero IoU with their coasted box, so an
        # IoU floor would veto precisely the matches this stage exists to make.
        # The Mahalanobis gate below is the spatial constraint, and it widens
        # with the coasted duration. Kept as a parameter so a specific sequence
        # can demand overlap if that turns out to help.
        self.reid_min_iou = reid_min_iou

        self.gallery_size = gallery_size
        self.frame_rate = frame_rate

        self.tracks = []
        self.frame_id = 0
        self.kf = KalmanFilter()

        # Diagnostics consumed by the failure-analysis tooling.
        self.events = []

    # ------------------------------------------------------------------ public
    def reset(self):
        """Clear state so the tracker can be reused on the next sequence."""
        self.tracks = []
        self.frame_id = 0
        self.events = []
        Track.reset_id_counter(1)
        Detection.reset_id_counter(1)

    def update(self, detections, frame_id=None):
        self.frame_id = self.frame_id + 1 if frame_id is None else frame_id

        keep = [d for d in detections if d.score >= self.det_conf_threshold]
        low = [d for d in detections if d.score < self.det_conf_threshold]

        for t in self.tracks:
            # Cleared up front: it describes *this* frame's match, and leaving a
            # stale True on a cascade-matched track would overstate how often
            # re-identification actually fired.
            t.is_recovered = False
            t.predict(self.kf)

        matches, unmatched_tracks, unmatched_dets = self._cascade_match(keep)
        if self.enable_reid:
            reid_pairs, unmatched_tracks, unmatched_dets = self._reid_match(
                keep, unmatched_tracks, unmatched_dets
            )
        else:
            reid_pairs = []

        for ti, di in matches:
            self.tracks[ti].update(
                self.kf, keep[di].tlwh, keep[di].score, self.frame_id,
                features=keep[di].features, min_hits=self.min_hits,
            )
        for ti, di in reid_pairs:
            t = self.tracks[ti]
            t.time_since_update_before = t.time_since_update
            t.update(
                self.kf, keep[di].tlwh, keep[di].score, self.frame_id,
                features=keep[di].features, min_hits=self.min_hits,
            )
            t.recovered += 1
            t.is_recovered = True
            self.events.append({
                "frame": self.frame_id,
                "type": "reid_recovery",
                "track_id": t.track_id,
                "coasted": t.time_since_update_before,
            })

        for ti in unmatched_tracks.tolist():
            self.tracks[ti].is_recovered = False
            self.tracks[ti].mark_missed()

        for di in unmatched_dets.tolist():
            d = keep[di]
            self.tracks.append(Track(d.tlwh, d.score, d.features, self.frame_id,
                                     gallery_size=self.gallery_size,
                                     min_hits=self.min_hits, kf=self.kf))

        # Low-confidence detections still spawn tracks; SORT does this too and
        # it materially helps recall for heavily occluded people.
        for d in low:
            self.tracks.append(Track(d.tlwh, d.score, d.features, self.frame_id,
                                     gallery_size=self.gallery_size,
                                     min_hits=self.min_hits, kf=self.kf))

        self._prune()

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_reportable()]

    # ----------------------------------------------------------------- private
    def _cascade_match(self, detections):
        """Oldest-first association; unmatched detections can seed new tracks.

        Each track is resolved against the still-available detections in turn,
        oldest first. Because a single row of an assignment problem is solved by
        its own argmin, this is equivalent to running the Hungarian solver on the
        reduced problem after every accepted pair, but it is far cheaper and
        makes the cascade semantics explicit.
        """
        if not self.cascade:
            return self._global_match(detections)

        available = list(range(len(detections)))
        matches = []

        # Longest-lived first: a track that has survived many frames has the
        # strongest claim to the detection it predicted.
        order = np.argsort([-t.age for t in self.tracks], kind="stable")

        for ti in order.tolist():
            t = self.tracks[ti]
            if not available:
                break
            if t.state == TrackState.Deleted:
                continue

            det_boxes = np.array([detections[j].tlwh for j in available],
                                 dtype=np.float32)
            ious = iou_cost(np.array([t.bbox], dtype=np.float32), det_boxes)[0]
            gate = self._gate(t)

            costs = ious.reshape(1, -1)

            # Appearance is only usable once the track actually has a history.
            ap = self._appearance_distance(t, detections, available)
            if ap is not None:
                costs = fuse_cost(costs, ap.reshape(1, -1), self.appearance_weight)

            # Reject pairs that are spatially implausible no matter how good the
            # appearance looks: a person cannot teleport 300 px in one frame.
            #
            # ``match_gate`` is a raw Mahalanobis radius in projected
            # measurement space. It is deliberately independent of the IoU gate
            # (which is a *cost* in [0, 1]) rather than being derived from it via
            # ``max(reid_max_gate, ...)``: that expression is pinned at
            # ``reid_max_gate`` for every realistic parameter set, which quietly
            # turned the adaptive gate into a constant.
            mahalanobis_gate = max(self.match_gate,
                                   gate * self.gate_multiplier + 2.0)
            admissible = (ious <= gate) & _mahalanobis_ok(
                t, det_boxes, mahalanobis_gate, kf=self.kf
            )
            if not admissible.any():
                continue

            masked = np.where(admissible, costs, np.inf)
            j = int(np.argmin(masked))
            if not np.isfinite(masked[0, j]):
                continue
            chosen = available.pop(j)
            matches.append([ti, chosen])

        matched_tracks = {ti for ti, _ in matches}
        unmatched_tracks = [i for i in range(len(self.tracks))
                            if i not in matched_tracks]
        return matches, np.array(unmatched_tracks, dtype=int), \
            np.array(available, dtype=int)

    def _global_match(self, detections):
        """One-shot Hungarian assignment over every track at once.

        This is SORT's association shape, kept on the same cost and gates as
        the cascade so the only difference in an ablation is the matching
        *order*: sequential oldest-first vs one global optimum. Pairs whose
        combined cost/gate are inadmissible get a large finite cost and are
        rejected after the assignment, since ``linear_sum_assignment`` cannot
        be given ``inf`` entries.
        """
        n_t, n_d = len(self.tracks), len(detections)
        if n_t == 0 or n_d == 0:
            return [], np.arange(n_t, dtype=int), np.arange(n_d, dtype=int)

        det_boxes = np.array([d.tlwh for d in detections], dtype=np.float32)
        t_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        costs = iou_cost(t_boxes, det_boxes)
        for i, t in enumerate(self.tracks):
            ap = self._appearance_distance(t, detections, list(range(n_d)))
            if ap is not None:
                costs[i:i + 1] = fuse_cost(costs[i:i + 1],
                                           ap.reshape(1, -1),
                                           self.appearance_weight)

        masked = np.full((n_t, n_d), 1e5, dtype=np.float64)
        for i, t in enumerate(self.tracks):
            if t.state == TrackState.Deleted:
                continue
            iou_gate = self._gate(t)
            mahalanobis_gate = max(self.match_gate,
                                   iou_gate * self.gate_multiplier + 2.0)
            admissible = (costs[i] <= iou_gate) & _mahalanobis_ok(
                t, det_boxes, mahalanobis_gate, kf=self.kf
            )
            masked[i, admissible] = costs[i, admissible]

        rows, cols = linear_sum_assignment(masked)
        matches = [[int(i), int(j)] for i, j in zip(rows, cols)
                   if masked[i, j] < 1e4]
        matched_tracks = {ti for ti, _ in matches}
        used_dets = {di for _, di in matches}
        unmatched_tracks = [i for i in range(n_t) if i not in matched_tracks]
        unmatched_dets = [di for di in range(n_d) if di not in used_dets]
        return matches, np.array(unmatched_tracks, dtype=int), \
            np.array(unmatched_dets, dtype=int)

    def _gate(self, track):
        """Adaptive IoU-cost threshold based on the track's own uncertainty.

        ``gate`` is a *cost* ceiling, so larger means more permissive. It starts
        at ``max_iou_distance`` (i.e. a minimum acceptable IoU of
        ``1 - max_iou_distance``) and widens with the magnitude of the velocity
        covariance, which grows automatically on every coasted frame.
        """
        vel_cov = np.asarray(track.P)[4:, 4:]
        trace = float(np.trace(vel_cov))
        gate = self.max_iou_distance + self.gate_multiplier * trace
        return float(np.clip(gate, self.max_iou_distance, 0.99))

    def _appearance_distance(self, track, detections, det_indices):
        """Min cosine distance over the track's appearance gallery."""
        gallery = track.gallery_matrix()
        if gallery.size == 0:
            return None
        det_feats = np.array([detections[j].features for j in det_indices],
                             dtype=np.float32)
        if det_feats.shape[0] == 0 or det_feats.ndim != 2:
            return None
        if det_feats.shape[1] != gallery.shape[1]:
            return None
        # (n_gallery, n_dets)
        d = cosine_distance_matrix(gallery, det_feats)
        return d.min(axis=0)

    def _reid_match(self, detections, unmatched_tracks, unmatched_dets):
        """Try to re-associate detections with tracks coasting through occlusion.

        Only tracks in ``Lost`` state and older than ``min_hits`` qualify — this
        is what distinguishes genuine occlusion (the track has a history, it just
        has no detection) from a detection that never belonged to anyone.
        """
        eligible = [
            ti for ti in unmatched_tracks.tolist()
            if self.tracks[ti].state == TrackState.Lost
            and self.tracks[ti].time_since_update <= self.reid_max_age
            and self.tracks[ti].hits >= self.min_hits
        ]
        if not eligible or len(unmatched_dets) == 0:
            return [], np.asarray(unmatched_tracks, dtype=int), np.asarray(unmatched_dets, dtype=int)

        pairs = []
        used_dets = set()
        matched_tracks = set()

        # Deterministic order: recover the longest-missing tracks first, since
        # they are closest to deletion and to expiring their gallery usefulness.
        eligible.sort(key=lambda i: -self.tracks[i].time_since_update)

        for ti in eligible:
            t = self.tracks[ti]
            t.time_since_update_before = t.time_since_update
            gallery = t.gallery_matrix()
            if gallery.size == 0:
                continue

            cand = [j for j in unmatched_dets.tolist() if j not in used_dets]
            if not cand:
                break
            det_feats = np.array([detections[j].features for j in cand],
                                 dtype=np.float32)
            if det_feats.ndim != 2 or det_feats.shape[1] != gallery.shape[1]:
                continue

            # Appearance: best over the whole gallery and over all candidates.
            cos = cosine_distance_matrix(gallery, det_feats).min(axis=0)

            # Spatial: Mahalanobis distance with a gate that widens with the
            # number of frames coasted, capped so it cannot pair distant people.
            track_boxes = np.array([t.bbox], dtype=np.float32)
            det_boxes = np.array([detections[j].tlwh for j in cand],
                                 dtype=np.float32)
            ious = iou_cost(track_boxes, det_boxes)[0]
            gate = min(self.reid_max_gate,
                       self.reid_gate_slope * t.time_since_update + 2.0)

            ok = (cos <= self.reid_appearance_threshold) & (ious <= 1.0 - self.reid_min_iou)
            ok &= _mahalanobis_ok(t, det_boxes, gate, kf=self.kf)
            if not ok.any():
                continue

            scores = np.where(ok, cos + 0.5 * ious, 1e5)
            best = int(np.argmin(scores))
            pairs.append([ti, cand[best]])
            used_dets.add(cand[best])
            matched_tracks.add(ti)

        remaining_tracks = [ti for ti in unmatched_tracks.tolist()
                            if ti not in matched_tracks]
        remaining_dets = [di for di in unmatched_dets.tolist()
                          if di not in used_dets]
        return pairs, np.array(remaining_tracks, dtype=int), \
            np.array(remaining_dets, dtype=int)

    def _prune(self):
        still = []
        for t in self.tracks:
            if t.time_since_update > max(self.max_age, self.reid_max_age):
                continue
            if t.state == TrackState.Tentative and t.time_since_update > 3:
                continue
            still.append(t)
        self.tracks = still


def _mahalanobis_ok(track, det_boxes, gate, kf=None):
    """Spatial plausibility test for a track/detection pairing.

    Returns a boolean mask over ``det_boxes`` (``(N, 4)`` in ``[x, y, w, h]``).

    The distance is computed in *projected* measurement space, i.e. after pushing
    the track's 8-D state through the filter's observation model and adding the
    detector's measurement noise. Testing the raw ``P[:4, :4]`` state block
    instead is a real trap: that block describes how well the *state* is known,
    not how well a *box* is known, and its area component carries a relative
    std of ~0.01. Against a ~23000 px^2 pedestrian that is a tolerance far
    tighter than any detector, so every candidate is rejected and the tracker
    silently fragments into a new track per detection per frame.
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    if det_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if kf is None:
        from .kalman import KalmanFilter
        kf = KalmanFilter()

    z = np.array([convert_bbox_to_z(b) for b in det_boxes], dtype=np.float64)
    mean = np.asarray(track.x, dtype=np.float64).reshape(8, 1)
    cov = np.asarray(track.P, dtype=np.float64).reshape(8, 8)

    projected_mean, projected_cov = kf.project(mean, cov)
    # Coasting covariances drift towards non-positive-definite; jitter keeps the
    # factorisation alive instead of returning garbage distances.
    chol = _cholesky(projected_cov)
    d = z - projected_mean.reshape(1, 4)  # (N, 4)
    whitened = solve_triangular(chol, d.T, lower=False)
    d2 = np.sum(whitened ** 2, axis=0)
    return np.sqrt(np.clip(d2, 0.0, None)) <= float(gate)


def _cholesky(matrix, jitter=1e-6, tries=4):
    """Upper-triangular Cholesky factor of a symmetric matrix.

    A track that has coasted for a while accumulates a covariance that is
    positive *semi*-definite at best, and after the state is augmented the
    projection can be numerically indefinite, so plain ``cholesky`` may fail.
    Escalating jitter handles the mild cases; the eigenvalue clip is the
    guaranteed fallback, since it can only make the matrix more positive
    definite and therefore cannot lose a factor that jitter would have found.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    scale = float(np.mean(np.diag(matrix)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    eye = np.eye(matrix.shape[0])
    for i in range(tries):
        try:
            return np.linalg.cholesky(matrix).T
        except np.linalg.LinAlgError:
            matrix = matrix + (jitter * (10.0 ** i) * scale) * eye
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, jitter * scale, None)
    return np.linalg.cholesky(vectors @ np.diag(values) @ vectors.T).T