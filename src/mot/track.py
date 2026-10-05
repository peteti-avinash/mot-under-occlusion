"""Track and Detection objects shared by all trackers in this package."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .kalman import convert_bbox_to_z, convert_z_to_bbox


class TrackState(Enum):
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Lost = 4  # predicted through occlusion, waiting for re-association


class Track:
    """A single tracked person.

    Holds the Kalman state, the appearance gallery used for re-identification,
    and the bookkeeping counters the tracker needs to decide when a track is
    genuinely gone versus temporarily occluded.

    The Kalman filter is owned by the tracker and passed in, so a track never
    carries a private copy of the filter and every track in a frame advances
    through exactly the same state transition.
    """

    _next_id = 1

    def __init__(self, bbox, score, det_features=None, frame_id=0,
                 gallery_size=30, min_hits=3, kf=None):
        self.frame_id = frame_id
        self.track_id = Track._next_id
        Track._next_id += 1
        self.state = TrackState.Tentative

        self.score = float(score)
        self.bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

        self.features = det_features
        self.gallery = []
        if det_features is not None:
            self.gallery.append(np.asarray(det_features, dtype=np.float32))
        self.gallery_size = gallery_size

        # The filter owns the transition and measurement matrices, so the state
        # is initialised from it rather than from an ad-hoc identity scaling.
        self.kf = kf
        z = convert_bbox_to_z(self.bbox)
        if kf is not None:
            self.x, self.P = kf.initiate(z)
        else:
            self.x = np.zeros((8, 1), dtype=np.float32)
            self.x[:4, 0] = z
            self.P = np.eye(8, dtype=np.float32) * 10.0
        self.std_pos = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.min_hits = min_hits
        self.fragments = 0          # number of times this track has been re-created
        self.recovered = 0          # number of successful occlusions survived
        self.gt_hint = -1           # filled in by the failure-analysis tooling
        self.is_recovered = False   # matched on this frame via the lost-track gallery
        self.time_since_update_before = 0

    # ---------------------------------------------------------------- id helpers
    @staticmethod
    def next_id():
        return Track._next_id

    @staticmethod
    def reset_id_counter(value=1):
        Track._next_id = value

    # ------------------------------------------------------------ frame updates
    def predict(self, kf=None):
        """Advance the state one frame without a measurement.

        This does *not* change the track state: whether an unmatched track is
        ``Lost`` or ``Deleted`` is the tracker's policy decision, made in
        :meth:`mark_missed` / :meth:`mark_occluded` after association.
        """
        kf = kf or self.kf
        if kf is None:
            self.time_since_update += 1
            self.age += 1
            return
        self.x, self.P = kf.predict(self.x, self.P)
        self.std_pos = np.sqrt(np.maximum(np.diag(self.P)[:4], 0)).astype(np.float32)
        self.age += 1
        self.time_since_update += 1
        self.bbox = convert_z_to_bbox(self.x[:4, 0])

    def update(self, kf, bbox, score, frame_id, features=None, min_hits=3):
        """Fuse a matched detection into the state.

        The posterior mean is produced by the Kalman gain; overwriting it with
        the raw measurement (as a naive implementation often does) silently
        disables velocity estimation, which is the whole point of running a
        filter here.
        """
        self.frame_id = frame_id
        self.score = float(score)
        self.time_since_update = 0
        self.time_since_update_before = 0
        self.hits += 1
        self.hit_streak += 1
        # ``age`` is deliberately not bumped here. Every track is advanced by
        # ``predict`` exactly once per frame, so incrementing in both places
        # would age a matched track twice as fast as a coasting one. That bias
        # corrupts the cascade ordering in the occlusion tracker, which ranks
        # tracks by age so the longest-lived claims its detection first: a
        # freshly matched track would outrank an older one that is still
        # recovering from an occlusion.
        # Promotion, not blanket confirmation. A track must accumulate
        # ``min_hits`` observations before it is allowed to be reported;
        # confirming on the first match makes every isolated detector
        # blip a one-frame "identity", which costs a false positive and a
        # fragment each. ``hits`` and ``hit_streak`` agree for any track that
        # survives, because a tentative track that loses its detection is
        # deleted rather than coasted.
        if self.state == TrackState.Tentative:
            if self.hits >= min_hits:
                self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Confirmed

        self.bbox = np.asarray(bbox, dtype=np.float32).copy()
        z = convert_bbox_to_z(self.bbox)
        self.x, self.P = kf.update(self.x, self.P, z)

        if features is not None:
            self.features = np.asarray(features, dtype=np.float32)
            self.gallery.append(self.features)
            if len(self.gallery) > self.gallery_size:
                self.gallery.pop(0)

    # --------------------------------------------------------- state transitions
    def mark_missed(self):
        """Called when no detection was associated this frame."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.state != TrackState.Deleted:
            self.state = TrackState.Lost

    def mark_occluded(self):
        """Kept alive but explicitly flagged as coasting through an occlusion."""
        if self.state != TrackState.Deleted:
            self.state = TrackState.Lost

    # ----------------------------------------------------------------- features
    def gallery_matrix(self):
        """All stored appearance features for this identity, as ``(n, d)``."""
        if not self.gallery:
            if self.features is None:
                return np.zeros((0, 0), dtype=np.float32)
            return np.asarray([self.features], dtype=np.float32)
        return np.asarray(self.gallery, dtype=np.float32)

    def mean_feature(self):
        g = self.gallery_matrix()
        if g.size == 0:
            return None
        return g.mean(axis=0)

    # ------------------------------------------------------------------- output
    def to_tlwh(self):
        x, y, w, h = self.bbox
        return (x, y, w, h)

    def to_xyxy(self):
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def is_reportable(self):
        """Whether this track may be written to a MOT result file.

        Two rules apply. A track is only emitted once it has been confirmed;
        emitting tentative tracks is what makes a tracker look good on the
        first few frames of a sequence and then fall apart. And a track that
        was *not* matched this frame is not emitted at all: a ``Lost`` track is
        a coasted Kalman prediction kept alive to bridge an occlusion, and
        writing it out duplicates the box of whatever detection now covers
        that space, which shows up as a false positive per missing frame.
        """
        return self.state == TrackState.Confirmed and self.time_since_update == 0


class Detection:
    """A single detector output, optionally carrying an appearance embedding."""

    _next_id = 1

    def __init__(self, bbox, score, features=None):
        self.tlwh = np.asarray(bbox, dtype=np.float32).copy()
        self.score = float(score)
        self.features = None if features is None else np.asarray(features, np.float32)
        self.track_id = 0
        self.is_occluded = False
        self.detection_id = Detection._next_id
        Detection._next_id += 1

    @staticmethod
    def reset_id_counter(value=1):
        Detection._next_id = value

    def to_xyxy(self):
        x, y, w, h = self.tlwh
        return (x, y, x + w, y + h)

    def to_tlbr(self):
        return self.to_xyxy()

    def to_measurement(self):
        return convert_bbox_to_z(self.tlwh)

    def __repr__(self):
        return f"Detection({self.tlwh.round(1).tolist()}, score={self.score:.2f})"