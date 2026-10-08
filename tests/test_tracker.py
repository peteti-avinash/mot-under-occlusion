"""Synthetic tracker tests: constant motion, crossing, and occlusion recovery.

These are the failure modes the project is about, so they are tested on exact
ground truth rather than only on MOT17 frames (which need a detector, and thus
cannot isolate tracker bugs from detector noise).

Run: ``python -m tests.test_tracker``
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from _runner import run_tests  # noqa: E402
from mot.kalman import KalmanFilter, _cholesky_factor  # noqa: E402
from mot.occlusion import OcclusionAwareTracker, _cholesky  # noqa: E402
from mot.sort import SortTracker  # noqa: E402
from mot.track import Detection, Track, TrackState  # noqa: E402


def _box(cx, cy, w=60, h=160):
    return [cx - w / 2.0, cy - h / 2.0, float(w), float(h)]


def _feat(seed):
    """Deterministic, well-separated appearance vectors."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=32).astype(np.float32)
    return v / np.linalg.norm(v)


def _run(tracker, frames):
    """Feed frames of ``[(gt_id, Detection)]`` and return id -> gt_id votes."""
    votes = {}
    for frame in frames:
        tracker.update([d for _, d in frame])
        for t in tracker.get_active_tracks():
            # Associate output to ground truth purely by position for the test.
            for gt_id, det in frame:
                if _iou(t.bbox, det.tlwh) > 0.5:
                    votes.setdefault(t.track_id, {}).setdefault(gt_id, 0)
                    votes[t.track_id][gt_id] += 1
    return votes


def _iou(a, b):
    tl = np.maximum(a[:2], b[:2])
    br = np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    inter = float(np.prod(np.clip(br - tl, 0, None)))
    ua = float(np.prod(a[2:])) + float(np.prod(b[2:])) - inter
    return inter / max(ua, 1e-6)


# --------------------------------------------------------------------- tests
def test_constant_motion_single_identity():
    """One object moving steadily must keep one ID for its whole life."""
    frames = []
    for i in range(40):
        cx = 100 + 8 * i
        frames.append([(1, Detection(_box(cx, 400), 0.9, _feat(1)))])

    for name, tracker in (("SORT", SortTracker(min_hits=1)),
                          ("occlusion", OcclusionAwareTracker(min_hits=1))):
        tracker.reset()
        ids = set()
        for f in frames:
            tracker.update([d for _, d in f])
            for t in tracker.get_active_tracks():
                if _iou(t.bbox, f[0][1].tlwh) > 0.5:
                    ids.add(t.track_id)
        assert len(ids) == 1, f"{name}: expected 1 id, got {sorted(ids)}"


def test_two_objects_stay_separate():
    """Two objects moving in parallel must never share or swap an ID."""
    frames = []
    for i in range(40):
        frames.append([
            (1, Detection(_box(200 + 6 * i, 300), 0.9, _feat(1))),
            (2, Detection(_box(200 + 6 * i, 600), 0.9, _feat(2))),
        ])

    for name, tracker in (("SORT", SortTracker(min_hits=1)),
                          ("occlusion", OcclusionAwareTracker(min_hits=1))):
        tracker.reset()
        votes = _run(tracker, frames)
        assert len(votes) == 2, f"{name}: expected 2 ids, got {len(votes)}"
        for tid, gt in votes.items():
            best = max(gt, key=gt.get)
            purity = gt[best] / sum(gt.values())
            assert purity > 0.9, f"{name}: id {tid} impure: {gt}"


def test_occlusion_longer_than_max_age():
    """A person hidden longer than SORT's max_age is recovered by appearance.

    The occlusion is deliberately longer than SORT's ``max_age`` so the baseline
    is forced to drop the track and mint a new id on reappearance. The occlusion
    tracker keeps the id alive by coasting and re-associating through the
    appearance gallery, which is the capability the project claims.
    """
    # Occluded for 36 frames: longer than SORT's max_age=30, shorter than the
    # occlusion tracker's reid_max_age=45. The person also keeps moving while
    # hidden, so by reappearance the coasted prediction has drifted; a lateral
    # jump puts the true position out of IoU reach of the prediction so the
    # appearance gallery is genuinely required.
    hidden = set(range(15, 51))
    JUMP = 150.0
    frames = []
    for i in range(80):
        dets = [(1, Detection(_box(60 + 6 * i, 300), 0.9, _feat(1)))]
        if i not in hidden:
            cx = 600 - 6 * i + (0.0 if i <= 14 else JUMP)
            dets.append((2, Detection(_box(cx, 520), 0.9, _feat(2))))
        frames.append(dets)

    results = {}
    for name, tracker in (("SORT", SortTracker(min_hits=1, max_age=30)),
                          ("occlusion",
                           OcclusionAwareTracker(min_hits=1, max_age=30,
                                                 reid_max_age=45))):
        tracker.reset()
        seen, recoveries = {}, 0
        for f in frames:
            tracker.update([d for _, d in f])
            for t in tracker.get_active_tracks():
                if t.is_recovered:
                    recoveries += 1
                for gt_id, det in f:
                    if _iou(t.bbox, det.tlwh) > 0.5:
                        seen.setdefault(gt_id, set()).add(t.track_id)
        results[name] = (seen, recoveries)

    sort_ids = results["SORT"][0].get(2, set())
    occ_seen, recoveries = results["occlusion"]
    occ_ids = occ_seen.get(2, set())

    print(f"  SORT gt2 ids={sorted(sort_ids)}  "
          f"occlusion gt2 ids={sorted(occ_ids)} recoveries={recoveries}")

    assert recoveries > 0, "occlusion tracker never used the reid stage"
    assert len(occ_ids) == 1, \
        f"occlusion: gt 2 fragmented into ids {sorted(occ_ids)}"


def test_crossing_paths_do_not_swap_ids():
    """Two people walking through each other's box keep their identities."""
    frames = []
    for i in range(50):
        t = i
        frames.append([
            (1, Detection(_box(150 + 8 * t, 300 + 6 * t), 0.9, _feat(1))),
            (2, Detection(_box(700 - 8 * t, 700 - 6 * t), 0.9, _feat(2))),
        ])

    for name, tracker in (("SORT", SortTracker(min_hits=1)),
                          ("occlusion", OcclusionAwareTracker(min_hits=1))):
        tracker.reset()
        votes = _run(tracker, frames)
        assert len(votes) == 2, f"{name}: expected 2 ids, got {sorted(votes)}"
        for tid, gt in votes.items():
            best = max(gt, key=gt.get)
            purity = gt[best] / sum(gt.values())
            assert purity > 0.9, f"{name}: id {tid} swapped identities: {gt}"


def test_tentative_track_is_not_reported_before_min_hits():
    """A one-frame detection must not become a reported identity.

    Without this rule every spurious box spawns a track that is written to the
    result file, which costs a false positive and a fragmentation each.
    """
    for name, tracker in (("SORT", SortTracker(min_hits=3)),
                          ("occlusion", OcclusionAwareTracker(min_hits=3))):
        tracker.reset()
        reported = []
        for i in range(3):
            tracker.update([Detection(_box(100 + 5 * i, 300), 0.9, _feat(1))])
            reported.append(len(tracker.get_active_tracks()))
        # Frames 1 and 2 are below min_hits; frame 3 is the first reportable one.
        assert reported[0] == 0 and reported[1] == 0, \
            f"{name}: reported before min_hits: {reported}"
        assert reported[2] == 1, \
            f"{name}: never confirmed after min_hits: {reported}"


def test_lost_track_is_not_written_to_output():
    """A coasted prediction is kept alive internally but never written out.

    Writing it duplicates whatever detection now covers that space, so each
    coasted frame becomes a false positive.
    """
    tracker = SortTracker(min_hits=1, max_age=10)
    tracker.reset()
    for i in range(5):
        tracker.update([Detection(_box(100 + 5 * i, 300), 0.9, _feat(1))])
    assert len(tracker.get_active_tracks()) == 1
    # The object is now undetected: the track coasts but must not be reported.
    for _ in range(5):
        tracker.update([])
        assert tracker.get_active_tracks() == [], \
            "coasted track was written to the result file"
    # It must survive internally, otherwise occlusion recovery is impossible.
    assert len(tracker.tracks) == 1 and tracker.tracks[0].state == TrackState.Lost


def test_kalman_covariance_stays_positive_definite():
    """The Joseph form keeps P usable when a match has a very different shape.

    Re-identification deliberately associates a track with a detection whose
    aspect ratio may differ by more than half. The short form ``P - K S K^T``
    then drives the aspect variance negative on that frame, and the next
    Cholesky of the projected covariance raises and the tracker dies.
    """
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([300.0, 200.0, 40.0 * 160.0, 40.0 / 160.0]))
    aspects = [0.25, 0.25, 0.37, 0.18, 0.41, 0.22, 0.35, 0.19, 0.44, 0.24]
    for i, aspect in enumerate(aspects):
        width = 40.0
        measurement = np.array([300.0 + 2 * i, 200.0 + i,
                                width * width / aspect, aspect])
        mean, cov = kf.update(mean, cov, measurement)
        eigenvalues = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
        assert eigenvalues.min() > -1e-6, \
            f"covariance lost positive semi-definiteness on update {i + 1}: " \
            f"{eigenvalues.min():.4g}"


def test_kalman_update_recovers_from_indefinite_covariance():
    """Recovers a state that is *already* indefinite, as happens in the field.

    The recorded state below is a real failure captured from MOT17-02 frame
    119 of the occlusion tracker: the covariance's smallest eigenvalue was
    -6.5e4 on the aspect dimension, so the Cholesky of the projected
    covariance raised and the tracker died. The short covariance form had
    silently produced that negative variance on an earlier frame, where a
    re-identification match paired the track with a much smaller, differently
    shaped detection.

    Neither float64 nor the Joseph form repairs a broken input covariance --
    Joseph can only avoid creating one. Recovery therefore depends on the
    Cholesky fallback in ``_cholesky_factor``.
    """
    kf = KalmanFilter()
    mean = np.array([
        1004.4188232421875, 441.04437255859375, 5329.6201171875,
        0.23306581377983093, 10.218745231628418, -44.05565643310547,
        -0.10339685529470444, -0.1622822880744934,
    ], dtype=np.float32).reshape(8, 1)
    covariance = np.array([
        np.array([12210626.0, 0.0, 0.0, 0.0, 12210560.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 12210626.0, 0.0, 0.0, 0.0, 12210560.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 189.9685821533203, 0.0, 0.0, 0.0, 7.401383399963379, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, -32624.0, 0.0, 0.0, 0.0, -32688.0], dtype=np.float32),
        np.array([12210048.0, 0.0, 0.0, 0.0, 12210085.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 12210048.0, 0.0, 0.0, 0.0, 12210085.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 7.401383399963379, 0.0, 0.0, 0.0, 0.3896704912185669, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, -32688.0, 0.0, 0.0, 0.0, -32667.33984375], dtype=np.float32),
    ], dtype=np.float32)
    measurement = np.array([
        993.6000366210938, 485.70001220703125, 1002.2376708984375,
        0.3678150773048401,
    ], dtype=np.float32)

    assert np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64)).min() < -1e-3, \
        "fixture no longer reproduces an indefinite covariance"

    # The tracker must survive the frame: this is the call that raised
    # ``LinAlgError`` and killed the run in the field.
    mean, covariance = kf.update(mean, covariance, measurement)
    assert np.isfinite(np.asarray(covariance, dtype=np.float64)).all()

    # A broken input covariance stays indefinite through the Joseph form --
    # that is a congruence transform of an indefinite matrix -- so the returned
    # covariance may legitimately still be indefinite. What must hold is that
    # the *next* frame can be factored, which is what `_cholesky_factor` is for.
    _, projected = kf.project(mean, covariance)
    factor = _cholesky_factor(projected)
    assert np.allclose(factor, np.triu(factor))


def test_cholesky_fallback_handles_indefinite_input():
    """Jitter cannot repair a matrix with large negative eigenvalues."""
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1 and 3
    factor = _cholesky_factor(matrix)
    assert np.allclose(factor, np.triu(factor)), "factor is not upper triangular"
    rebuilt = factor.T @ factor
    assert np.allclose(rebuilt, rebuilt.T, atol=1e-9), "rebuilt matrix is asymmetric"
    assert np.linalg.eigvalsh(rebuilt).min() >= -1e-9, \
        "rebuilt matrix is not positive semi-definite"
    # The repair must stay close to the original, i.e. only the offending
    # eigenvalue is lifted rather than the whole matrix being replaced.
    assert np.allclose(rebuilt, matrix, atol=1.0), \
        f"rebuilt matrix drifted too far: {rebuilt}"


def test_occlusion_cholesky_helper_survives_indefinite_input():
    matrix = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
    factor = _cholesky(matrix)
    rebuilt = factor.T @ factor
    assert np.allclose(factor, np.triu(factor)), "factor is not upper triangular"
    assert np.linalg.eigvalsh(rebuilt).min() >= -1e-9


def test_id_reset_between_sequences():
    """Ids restart at 1 so two sequences produce comparable result files."""
    for tracker in (SortTracker(), OcclusionAwareTracker()):
        tracker.reset()
        for i in range(10):
            tracker.update([Detection(_box(100 + 5 * i, 300), 0.9, _feat(1))])
        first = {t.track_id for t in tracker.tracks}
        tracker.reset()
        for i in range(10):
            tracker.update([Detection(_box(100 + 5 * i, 300), 0.9, _feat(1))])
        second = {t.track_id for t in tracker.tracks}
        assert min(second) == min(first) == 1, \
            f"{type(tracker).__name__}: ids did not reset ({first} vs {second})"


def test_track_age_counts_frames_not_observations():
    """``age`` must advance at the same rate whether a track is matched or coasting.

    The occlusion tracker ranks its cascade by ``-age`` so the longest-lived
    track claims a contested detection first. If matched tracks aged twice as
    fast, a new arrival would outrank an older track recovering from an
    occlusion -- exactly the case the ordering exists to get right.
    """
    tr = SortTracker(min_hits=1, max_age=10)
    tr.reset()
    for _ in range(4):
        tr.update([Detection([70.0, 220.0, 60.0, 160.0], 0.9)])
    track = tr.tracks[0]
    assert track.age == 4, f"matched track age after 4 frames: {track.age}, want 4"

    # Now coast for two frames without any detection.
    for _ in range(2):
        tr.update([])
    assert track.age == 6, f"coasting track age after 2 more frames: {track.age}, want 6"


def test_occlusion_cascade_prefers_the_older_track():
    """An older coasting track must outrank a newer, actively matched one.

    Scenario chosen so the ``age`` bias actually flips the order: the older
    track coasts for 17 frames while the newer one is matched for 13. Counting an
    age step on ``update`` as well as ``predict`` would give the newer track 11
    against the older track's 10, inverting the ranking.
    """
    tracker = OcclusionAwareTracker(min_hits=1, max_age=30)
    tracker.reset()
    for _ in range(4):
        tracker.update([Detection([70.0, 220.0, 60.0, 160.0], 0.9)])
    old = tracker.tracks[0]
    for _ in range(4):
        tracker.update([])          # the old track now coasts and ages 1x/frame

    fresh = Track([620.0, 220.0, 60.0, 160.0], 0.9, frame_id=tracker.frame_id + 1)
    tracker.tracks.append(fresh)
    for _ in range(13):
        tracker.update([Detection([620.0, 220.0, 60.0, 160.0], 0.9)])

    assert old.age > fresh.age, (
        f"precondition failed: old={old.age} fresh={fresh.age}")
    order = [tracker.tracks[i] for i in np.argsort([-t.age for t in tracker.tracks],
                                                   kind="stable")]
    assert order[0] is old, (
        f"cascade ranked by age put the newer track first "
        f"(old={old.age}, fresh={fresh.age})")


def main():
    return run_tests(dict(globals()), "tracker")


def test_global_hungarian_matches_one_to_one():
    """cascade=False must produce the same one-to-one pairing as the cascade.

    Two mature tracks side by side, two detections obviously above them: both
    association shapes must match both tracks and leave nothing unmatched.
    """
    det1 = Detection(_box(101, 180), 0.9)
    det2 = Detection(_box(301, 180), 0.9)
    pairs = []
    for use_cascade in (True, False):
        tr = OcclusionAwareTracker(min_hits=1, enable_reid=False,
                                   cascade=use_cascade)
        tr.reset()
        t1 = Track(_box(100, 180), 0.9, frame_id=1, min_hits=1, kf=tr.kf)
        t2 = Track(_box(300, 180), 0.9, frame_id=1, min_hits=1, kf=tr.kf)
        tr.tracks = [t1, t2]
        for t in tr.tracks:
            t.hits = 10
        m, unmatched_t, unmatched_d = tr._cascade_match([det1, det2])
        assert len(m) == 2, f"cascade={use_cascade}: matches {m}"
        assert len(unmatched_t) == 0 and len(unmatched_d) == 0, (
            f"cascade={use_cascade}: unmatched tracks {unmatched_t}, "
            f"dets {unmatched_d} -- matched dets must not respawn as new tracks")
        pairs.append(sorted((ti, di) for ti, di in m))
    assert pairs[0] == pairs[1], f"pairings disagree: {pairs}"


def test_global_hungarian_never_double_reports_dets():
    """A matched detection must not also come back as unmatched.

    The first cut of ``_global_match`` returned ``np.arange(n_d)``, so every
    matched detection spawned a duplicate track and the ablation collapsed
    (IDS 409, recall 17% on the 60-frame smoke). This pins the contract.
    """
    tr = OcclusionAwareTracker(min_hits=1, enable_reid=False, cascade=False)
    tr.reset()
    t1 = Track(_box(100, 180), 0.9, frame_id=1, min_hits=1, kf=tr.kf)
    tr.tracks = [t1]
    t1.hits = 10
    det = Detection(_box(101, 180), 0.9)
    m, unmatched_t, unmatched_d = tr._cascade_match([det])
    assert m == [[0, 0]], f"matches {m}"
    assert len(unmatched_d) == 0, (
        f"matched det returned as unmatched: {unmatched_d}")


def test_global_hungarian_solves_the_crossed_case():
    """One global assignment must swap detections the greedy order cannot.

    Track 0 sits at x=100, track 1 at x=300. The detection nearer track 1
    arrives first; a global optimum pairs each track with its own detection.
    """
    tr = OcclusionAwareTracker(min_hits=1, enable_reid=False, cascade=False)
    tr.reset()
    t1 = Track(_box(100, 180), 0.9, frame_id=1, min_hits=1, kf=tr.kf)
    t2 = Track(_box(300, 180), 0.9, frame_id=1, min_hits=1, kf=tr.kf)
    tr.tracks = [t1, t2]
    for t in tr.tracks:
        t.hits = 10
    det_near_t1 = Detection(_box(302, 180), 0.9)   # x=302: nearer track 1
    det_near_t0 = Detection(_box(99, 180), 0.9)    # x=99:  nearer track 0
    m, _, _ = tr._cascade_match([det_near_t1, det_near_t0])
    got = {ti: di for ti, di in m}
    # track 0 -> det 1 (x=99), track 1 -> det 0 (x=302)
    assert got == {0: 1, 1: 0}, f"crossed pairing came out as {m}"


def test_no_reid_disables_the_recovery_stage():
    """enable_reid=False must never emit reid_recovery events or recovered hits."""
    tracker = OcclusionAwareTracker(min_hits=1, max_age=5, enable_reid=False)
    tracker.reset()
    for _ in range(3):
        tracker.update([Detection([70.0, 220.0, 60.0, 160.0], 0.9, _feat(1))])
    track = tracker.tracks[0]
    # Coast far past the reid window with a near-identical appearance
    # detection available: with reid enabled this would recover the id.
    for _ in range(8):
        tracker.update([Detection([74.0, 220.0, 60.0, 160.0], 0.9, _feat(1))])
    assert tracker.enable_reid is False
    assert all(e["type"] != "reid_recovery" for e in tracker.events), (
        "reid events emitted while enable_reid=False")
    assert track.recovered == 0, f"recovered hits with reid disabled: {track.recovered}"

if __name__ == "__main__":
    raise SystemExit(main())
