"""Baseline SORT tracker: motion-only association (Kalman + Hungarian IoU).

This is the reference implementation from Bewley et al. (2016), kept deliberately
faithful and dependency-light so that any improvement in :mod:`mot.occlusion`
is attributable to appearance and occlusion handling rather than to a
re-tuned baseline. It is one of the two systems compared in the ablation.
"""

from __future__ import annotations

import numpy as np

from .associate import iou_cost, linear_assignment
from .kalman import KalmanFilter
from .track import Detection, Track, TrackState


class SortTracker:
    """Classic SORT."""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3,
                 det_conf_threshold=0.4, frame_rate=30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_conf_threshold = det_conf_threshold
        self.frame_rate = frame_rate
        self.tracks = []
        self.frame_id = 0
        self.kf = KalmanFilter()

    # ------------------------------------------------------------------ public
    def reset(self):
        """Clear state so the tracker can be reused on the next sequence.

        Track ids must restart at 1 per sequence: MOTChallenge treats ids as
        sequence-local, and a globally increasing counter would make two runs
        produce incomparable result files.
        """
        self.tracks = []
        self.frame_id = 0
        Track.reset_id_counter(1)
        Detection.reset_id_counter(1)

    def update(self, detections, frame_id=None):
        self.frame_id = self.frame_id + 1 if frame_id is None else frame_id

        # SORT predicts *before* associating. Without this step the tracker
        # matches detections against last frame's boxes, which is what makes
        # fast-moving people appear to have no velocity at all.
        for t in self.tracks:
            t.predict(self.kf)

        matches, unmatched_dets, unmatched_tracks = self._match(detections)

        for ti, di in matches:
            self.tracks[ti].update(
                self.kf, detections[di].tlwh, detections[di].score,
                self.frame_id, features=detections[di].features,
                min_hits=self.min_hits,
            )
        for ti in unmatched_tracks.tolist():
            self.tracks[ti].mark_missed()

        for di in unmatched_dets.tolist():
            det = detections[di]
            self.tracks.append(
                Track(det.tlwh, det.score, det.features, self.frame_id,
                      min_hits=self.min_hits, kf=self.kf)
            )

        self._prune()

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_reportable()]

    def predict_only(self):
        for t in self.tracks:
            t.predict(self.kf)

    # ----------------------------------------------------------------- private
    def _prune(self):
        still = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue
            if t.state == TrackState.Tentative and t.time_since_update > 1:
                # A tentative track that immediately lost its detection was a
                # detection artefact, not a person.
                continue
            if t.state == TrackState.Deleted:
                continue
            still.append(t)
        self.tracks = still

    def _match(self, detections):
        # Partition by confidence once, keeping original indices so the caller's
        # index space is preserved for the track-creation step.
        keep_idx = [i for i, d in enumerate(detections)
                    if d.score >= self.det_conf_threshold]
        low_idx = [i for i, d in enumerate(detections)
                   if d.score < self.det_conf_threshold]

        if len(self.tracks) == 0 or len(keep_idx) == 0:
            # With nothing to match against, *every* track is unmatched. Returning
            # an empty unmatched set here would leave tracks in the Confirmed
            # state forever instead of aging them into Lost, which breaks the
            # re-identification gallery the occlusion tracker keys on.
            unmatched = (np.arange(len(self.tracks), dtype=int)
                         if len(keep_idx) == 0 else np.array([], dtype=int))
            return [], np.array(range(len(detections)), dtype=int), unmatched

        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = np.array([detections[i].tlwh for i in keep_idx],
                             dtype=np.float32)
        # SORT's published cost is pure IoU; detection confidence is used only as
        # a pre-filter, not folded into the cost, so the baseline stays faithful.
        cost = iou_cost(track_boxes, det_boxes)

        matches, unmatched_track_local, unmatched_det_local = linear_assignment(
            cost, self.iou_threshold
        )
        # Both index spaces are local: map track and detection indices back into
        # the caller's lists before returning, or the track-creation step below
        # would index the wrong detections.
        matches_global = [[ti, keep_idx[di]] for ti, di in matches]
        unmatched_tracks = np.array(unmatched_track_local, dtype=int)
        unmatched_dets = np.array(
            [keep_idx[j] for j in unmatched_det_local.tolist()] + low_idx,
            dtype=int,
        )
        return matches_global, unmatched_dets, unmatched_tracks


def convert_detections(det_boxes, scores, features=None, min_score=0.0):
    """Convenience constructor used by the pipeline."""
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    feats = None
    if features is not None:
        feats = np.asarray(features, dtype=np.float32).reshape(len(det_boxes), -1)
    out = []
    for i in range(len(det_boxes)):
        if scores[i] < min_score:
            continue
        f = feats[i] if feats is not None else None
        out.append(Detection(det_boxes[i], scores[i], f))
    return out