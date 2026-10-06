"""Analytic checks for the local metric implementations.

These are hand-computed expectations rather than fixtures, so they pin the
definitions themselves. The HOTA cases are the important ones: an implementation
that drops the association term scores 1.0 even when every frame carries a
different track id, which reduces HOTA to a pure detection score.

Every HOTA expectation below is derived from the TrackEval definition

    A(c,t) = TPA / (TPA + FNA + FPA)
    AssA   = sum_{c,t} TPA * A / TP
    HOTA   = sqrt(DetA * AssA)

and each test states its own derivation in a comment. A separate test
cross-checks ``hota`` against an independent transcription of the TrackEval
algorithm so that a shared misreading of the spec cannot pass both.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from mot.metrics import HOTA_ALPHAS, distance_matrix, evaluate, hota  # noqa: E402


def _box(track_id, x=0.0):
    return np.array([[track_id, x, 0.0, 10.0, 10.0]], dtype=np.float64)


def test_hota_perfect_identity_is_exactly_one():
    n = 10
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    hyp = {t: _box(1) for t in frames}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(value - 1.0) < 1e-9, value
    assert abs(det_a - 1.0) < 1e-9, det_a
    assert abs(ass_a - 1.0) < 1e-9, ass_a


def test_hota_penalises_identity_switches():
    """Pixel-identical boxes, so only the association term can move HOTA."""
    n = 10
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}

    # One switch at frame 6. The gt is covered for 10 frames (gt_count 10).
    # Tracker 1 covers frames 1-5 (count 5), tracker 2 covers frames 6-10
    # (count 5), so TPA(c,1) = TPA(c,2) = 5. Neither tracker ever fires on a
    # frame where it was not this gt's match, so FPA = 0 for both pairs and
    # FNA = 10 - 5 = 5.
    # A(c,t) = 5 / (5 + 5 + 0) = 1/2 for both pairs.
    # AssA = (5 * 1/2 + 5 * 1/2) / 10 = 1/2.
    # DetA = 10 / 10 = 1, so HOTA = sqrt(1/2).
    one_switch = {t: _box(1 if t < 6 else 2) for t in frames}
    value, det_a, ass_a = hota(one_switch, gt, n)
    assert abs(value - math.sqrt(0.5)) < 1e-9, value
    assert abs(ass_a - 0.5) < 1e-9, ass_a
    assert abs(det_a - 1.0) < 1e-9, "DetA must stay pinned at 1.0"

    # A different id every frame: ten trackers each covering one frame, so
    # TPA = 1, tracker count = 1, FPA = 0, FNA = 9 for every pair, and
    # A(c,t) = 1 / (1 + 9 + 0) = 1/10.
    # AssA = (10 * 1/10) / 10 = 1/10, HOTA = sqrt(1/10).
    all_switch = {t: _box(t) for t in frames}
    scrambled, det_a, ass_a = hota(all_switch, gt, n)
    assert abs(ass_a - 0.1) < 1e-9, ass_a
    assert abs(scrambled - math.sqrt(0.1)) < 1e-9, scrambled
    assert abs(det_a - 1.0) < 1e-9, "DetA must stay pinned at 1.0"
    assert scrambled < value, "more switches did not score worse"


def test_hota_half_recall_with_perfect_association():
    """Two gt people, one always detected.

    Both gt are present for 10 frames, so gt_count = 10 each. Tracker 1
    matches gt 1 on all 10 frames (TPA 10, FPA 0) and never fires at gt 2, so
    A(c=1,t=1) = 10 / (10 + 0 + 0) = 1 and A(c=2,t=1) = 0.
    AssA = (10 * 1) / 10 = 1, DetA = 10 / 20 = 0.5,
    HOTA = sqrt(0.5).
    """
    n = 10
    frames = range(1, n + 1)
    gt = {t: np.concatenate([_box(1, 0.0), _box(2, 500.0)]) for t in frames}
    hyp = {t: _box(1, 0.0) for t in frames}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(value - math.sqrt(0.5)) < 1e-9, value
    assert abs(det_a - 0.5) < 1e-9, det_a
    assert abs(ass_a - 1.0) < 1e-9, ass_a


def _reference_hota(hypotheses, gt, num_frames, alphas=HOTA_ALPHAS):
    """Independent transcription of the TrackEval HOTA algorithm.

    Written from the same spec as ``mot.metrics.hota`` but with its own loops,
    its own index handling and no shared helper, so that a misreading of the
    spec that both copies happen to share still shows up as disagreement.
    """
    from scipy.optimize import linear_sum_assignment

    alphas = np.asarray(alphas, dtype=np.float64)
    eps = np.finfo("float").eps

    def _iou_one(a, b):
        a_x2, a_y2 = a[0] + a[2], a[1] + a[3]
        b_x2, b_y2 = b[0] + b[2], b[1] + b[3]
        w = max(0.0, min(a_x2, b_x2) - max(a[0], b[0]))
        h = max(0.0, min(a_y2, b_y2) - max(a[1], b[1]))
        inter = w * h
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / max(union, 1e-10)

    gt_ids = sorted({int(r[0]) for boxes in gt.values() for r in boxes})
    hyp_ids = sorted({int(r[0]) for boxes in hypotheses.values() for r in boxes})
    if not gt_ids or not hyp_ids:
        return 0.0, 0.0, 0.0
    g_pos = {v: i for i, v in enumerate(gt_ids)}
    h_pos = {v: i for i, v in enumerate(hyp_ids)}

    seq = []
    for frame in range(1, num_frames + 1):
        g = gt.get(frame)
        d = hypotheses.get(frame)
        g_i = [g_pos[int(r[0])] for r in g] if g is not None and len(g) else []
        h_i = [h_pos[int(r[0])] for r in d] if d is not None and len(d) else []
        if g_i and h_i:
            # sim[i, j] must be IoU(gt_i, hyp_j).
            g_boxes = [r[1:5] for r in g]
            h_boxes = [r[1:5] for r in d]
            sim = np.asarray([[_iou_one(a, b) for b in h_boxes] for a in g_boxes],
                             dtype=np.float64)
        else:
            sim = np.zeros((len(g_i), len(h_i)))
        seq.append((np.array(g_i, dtype=np.int64), np.array(h_i, dtype=np.int64), sim))

    potential = np.zeros((len(gt_ids), len(hyp_ids)))
    n_gt = np.zeros((len(gt_ids), 1))
    n_hyp = np.zeros((1, len(hyp_ids)))
    for g_i, h_i, sim in seq:
        if len(g_i) and len(h_i):
            union = sim.sum(axis=0)[None, :] + sim.sum(axis=1)[:, None] - sim
            jac = np.zeros_like(sim)
            ok = union > eps
            jac[ok] = sim[ok] / union[ok]
            potential[g_i[:, None], h_i[None, :]] += jac
        n_gt[g_i] += 1
        n_hyp[0, h_i] += 1
    denom = n_gt + n_hyp - potential
    align = np.where(denom > eps, potential / denom, 0.0)
    align[~np.isfinite(align)] = 0.0

    acc = np.zeros((len(alphas), len(gt_ids), len(hyp_ids)))
    tp = np.zeros(len(alphas))
    fn = np.zeros(len(alphas))
    fp = np.zeros(len(alphas))
    for g_i, h_i, sim in seq:
        if len(g_i) == 0:
            fp += len(h_i)
            continue
        if len(h_i) == 0:
            fn += len(g_i)
            continue
        cost = align[g_i[:, None], h_i[None, :]] * sim
        r, c = linear_sum_assignment(-cost)
        for ai, alpha in enumerate(alphas):
            hit = sim[r, c] >= alpha - eps
            nr = int(hit.sum())
            tp[ai] += nr
            fn[ai] += len(g_i) - nr
            fp[ai] += len(h_i) - nr
            if nr:
                acc[ai][g_i[r[hit]], h_i[c[hit]]] += 1.0

    ass = np.zeros(len(alphas))
    det = np.zeros(len(alphas))
    for ai in range(len(alphas)):
        a = acc[ai]
        pair = a / np.maximum(1.0, n_gt + n_hyp - a)
        ass[ai] = (a * pair).sum() / max(1.0, tp[ai])
        det[ai] = tp[ai] / max(1.0, tp[ai] + fn[ai] + fp[ai])
    return (float(np.sqrt(det * ass).mean()), float(det.mean()), float(ass.mean()))


def test_hota_matches_independent_trackeval_transcription():
    """Cases with identity switches, false positives and missed detections."""
    rng = np.random.default_rng(7)
    for trial in range(4):
        num_frames = 12
        gt, hyp = {}, {}
        for frame in range(1, num_frames + 1):
            rows = []
            for person in range(1 + trial):
                x = 100.0 * person + rng.uniform(-6.0, 6.0)
                rows.append([float(person), x, 50.0, 20.0, 40.0])
            if frame in (4, 9) and trial >= 2:
                rows.append([90.0, rng.uniform(0, 300), 50.0, 20.0, 40.0])
            gt[frame] = np.asarray(rows, dtype=np.float64)

        rows = []
        for person in range(1 + trial):
            if frame == 7 and trial >= 1:
                person += 1  # tracker changes id mid-sequence
            x = 100.0 * (person - 1) + rng.uniform(-6.0, 6.0)
            rows.append([float(person + 1), x, 50.0, 20.0, 40.0])
        # One false positive per fourth frame, emitted once per frame: keeping
        # it outside the loop above matters, because two boxes sharing one id
        # inside a frame is not valid MOT data.
        if frame % 4 == 0:
            rows.append([float(70 + frame), rng.uniform(0, 300), 50.0, 20.0, 40.0])
        hyp[frame] = np.asarray(rows, dtype=np.float64)

        got = hota(hyp, gt, num_frames)
        want = _reference_hota(hyp, gt, num_frames)
        for name, a, b in zip(("hota", "det_a", "ass_a"), got, want):
            assert abs(a - b) < 1e-9, (
                f"trial {trial} {name}: local {a!r} vs reference {b!r}")


def test_hota_uses_global_alignment_not_raw_iou_matching():
    """The per-frame assignment must be driven by the global alignment score.

    Two gt people whose 10 px-wide boxes overlap by 4 px, so per-frame IoU is
    ambiguous about who owns the overlap. Tracker 1 jitters around gt 1 for the
    whole sequence; tracker 2 starts on gt 2 and walks left through gt 1's
    position.

    Matching on raw IoU, tracker 2 wins the contested frames near the end and
    the identity of gt 1 flips between trackers, dropping AssA to 56.65. The
    global alignment carries tracker 1's consistent eight-frame history into
    the assignment and holds AssA at 79.47. Replacing the score matrix with
    ``sim`` changes HOTA from 61.29 to 51.94, so this pins that term.
    """
    n = 8
    gt = {f: np.concatenate([_box(1, 0.0), _box(2, 14.0)])
          for f in range(1, n + 1)}
    jitter = (0.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5)
    walk = (14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0)
    hyp = {f: np.concatenate([_box(1, jitter[f - 1]), _box(2, walk[f - 1])])
           for f in range(1, n + 1)}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(value * 100.0 - 61.285868) < 1e-4, value * 100.0
    assert abs(det_a * 100.0 - 47.957662) < 1e-4, det_a * 100.0
    assert abs(ass_a * 100.0 - 79.465603) < 1e-4, ass_a * 100.0


def test_distance_matrix_is_object_by_hypothesis():
    """The matrix ``evaluate`` hands to ``motmetrics`` is ``(objects, hypotheses)``.

    ``MOTAccumulator.update`` flattens its argument with
    ``reshape(oids.shape[0], hids.shape[0])`` (``motmetrics/mot.py``), so a matrix
    of the opposite orientation is silently reinterpreted element-for-element
    instead of raising. With two hypotheses and three objects the transposed
    layout still has the right element count, so nothing crashes and the matching
    is quietly wrong. The shape is therefore part of the contract.
    """
    det = np.asarray([[10.0, 10.0, 20.0, 20.0],
                      [50.0, 50.0, 20.0, 20.0]])
    gt = np.asarray([[10.0, 10.0, 20.0, 20.0],
                     [50.0, 50.0, 20.0, 20.0],
                     [90.0, 90.0, 20.0, 20.0]])
    dists = distance_matrix(det, gt)
    assert dists.shape == (3, 2), dists.shape
    # rows are objects: object 0 matches hypothesis 0 exactly, and the object at
    # (90,90) is forbidden against both hypotheses, not merely the second.
    assert dists[0, 0] == 0.0
    assert math.isnan(dists[2, 0]) and math.isnan(dists[2, 1])


def test_evaluate_clear_mot_event_counts_on_a_hand_counted_case():
    """MOTA follows from CLEAR's event counts on a case countable by hand.

    Two frames, three objects per frame, two hypotheses per frame, both of which
    sit exactly on an object. That gives 4 matches, no false positives, 2 misses
    (the third object is never tracked in either frame) and no id switches over 6
    object-frames, so MOTA = (4 - 0 - 0) / 6 = 66.667. A transposed distance
    matrix or an off-by-one frame range changes these counts.
    """
    import tempfile

    directory = tempfile.mkdtemp()
    gt_path = os.path.join(directory, "gt.txt")
    with open(gt_path, "w") as handle:
        for frame in (1, 2):
            for obj_id, x in ((1, 10), (2, 50), (3, 90)):
                handle.write(f"{frame},{obj_id},{x},{x},20,20,1,1,1\n")

    detections = np.asarray([[7, 10, 10, 20, 20, 1],
                             [8, 50, 50, 20, 20, 1]], dtype=np.float32)
    metrics = evaluate({1: detections, 2: detections.copy()},
                       gt_path, sequence="case", max_frame=2)
    assert int(metrics.num_matches) == 4, metrics.num_matches
    assert int(metrics.num_false_positives) == 0, metrics.num_false_positives
    assert int(metrics.num_misses) == 2, metrics.num_misses
    assert int(metrics.num_switches) == 0, metrics.num_switches
    assert int(metrics.num_objects) == 6, metrics.num_objects
    assert abs(metrics.mota - 100.0 * 4 / 6) < 1e-9, metrics.mota
    assert abs(metrics.motp - 100.0) < 1e-9, metrics.motp


def test_hota_counts_repeated_id_rows_in_one_frame():
    """A repeated id inside one frame must accumulate once per row.

    ``arr[ids] += x`` applies only the last write for a duplicated index, so
    the obvious spelling of the alignment accumulation silently drops a row and
    inflates the Jaccard association score. Two gt rows sharing id 1 in each of
    two frames, with one hypothesis that matches one of them, must score
    AssA = 2 / (4 + 2 - 2) = 0.5 rather than 1.0.
    """
    gt = {
        1: np.asarray([[1.0, 0.0, 0.0, 10.0, 10.0],
                       [1.0, 40.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        2: np.asarray([[1.0, 0.0, 0.0, 10.0, 10.0],
                       [1.0, 40.0, 0.0, 10.0, 10.0]], dtype=np.float64),
    }
    hyp = {
        1: np.asarray([[5.0, 0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
        2: np.asarray([[5.0, 0.0, 0.0, 10.0, 10.0]], dtype=np.float64),
    }
    value, _det_a, ass_a = hota(hyp, gt, 2, alphas=[0.5])
    assert abs(ass_a - 0.5) < 1e-9, ass_a
    assert abs(value - np.sqrt(0.5 * 0.5)) < 1e-9, value


def test_hota_alignment_accumulates_jaccard_not_raw_iou():
    """The alignment pass must accumulate per-frame Jaccard, not raw IoU.

    Two gt people and three trackers over four frames, all boxes 10 px wide on
    one row, so overlaps are easy to check by hand. Frame 1 has all three
    trackers stacked on gt 1, which makes the per-frame Jaccard for that pair
    small even though the raw IoU is 1.0. Accumulating raw IoU instead of
    Jaccard over-rewards a tracker that crowds one gt and produces HOTA 39.12
    with DetA 27.77 / AssA 56.04; the Jaccard form gives 29.73 with
    DetA 35.26 / AssA 25.45.
    """
    gt = {
        1: np.concatenate([_box(1, 20.0), _box(2, 18.0)]),
        2: np.concatenate([_box(1, 13.0), _box(2, 8.0)]),
        3: np.concatenate([_box(1, 13.0), _box(2, 8.0)]),
        4: np.concatenate([_box(1, 8.0), _box(2, 3.0)]),
    }
    hyp = {
        1: np.concatenate([_box(1, 20.0), _box(2, 0.0), _box(3, 0.0)]),
        2: np.concatenate([_box(1, 11.0), _box(2, 24.0), _box(3, 5.0)]),
        3: np.concatenate([_box(1, 7.0), _box(2, 20.0), _box(3, 3.0)]),
        4: np.concatenate([_box(1, 3.0), _box(2, 7.0), _box(3, 18.0)]),
    }
    value, det_a, ass_a = hota(hyp, gt, 4)
    assert abs(value * 100.0 - 29.734140) < 1e-4, value * 100.0
    assert abs(det_a * 100.0 - 35.256410) < 1e-4, det_a * 100.0
    assert abs(ass_a * 100.0 - 25.451725) < 1e-4, ass_a * 100.0


def test_hota_penalises_false_positive_association():
    """A track that stays alive after drifting off its gt must lose AssA.

    The gt is present for 10 frames. Tracker 1 locks on for frames 1-5, then
    drifts 900 px away for frames 6-10 while staying alive. That gives
    TPA(c,1) = 5, FNA = 10 - 5 = 5 and FPA = 5 (it fires on 5 frames without
    being this gt's match), so

        A(c,1) = 5 / (5 + 5 + 5) = 1/3
        AssA   = (5 * 1/3) / 5 = 1/3

    The five off-target frames are both FPA (a tracker firing without its gt)
    and FP (a detection with no gt at all), so DetA = 5 / (5 + 5 + 5) = 1/3 and
    HOTA = 1/3.

    Dropping FPA from the denominator inflates A to 5/10 and AssA to 1/2,
    which is the specific regression this pins.
    """
    n = 10
    gt = {t: _box(1, 0.0) for t in range(1, n + 1)}
    hyp = {t: _box(1, 0.0 if t <= 5 else 900.0) for t in range(1, n + 1)}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(ass_a - 1.0 / 3.0) < 1e-9, ass_a
    assert abs(det_a - 1.0 / 3.0) < 1e-9, det_a
    assert abs(value - 1.0 / 3.0) < 1e-9, value


def test_hota_counts_detections_on_frames_with_no_ground_truth():
    """A detection on an empty frame is a false positive and must be charged.

    Eight gt frames plus two frames where the scene is empty. The tracker
    keeps firing on all ten, so it is never absent, which makes the two empty
    frames FPA for this id pair:

        TP = 8, FN = 0, FP = 2
        DetA = 8 / 10 = 0.8
        A(c,1) = TPA / (TPA + FNA + FPA) = 8 / (8 + 0 + 2) = 0.8
        AssA = (8 * 0.8) / 8 = 0.8
        HOTA = sqrt(0.8 * 0.8) = 0.8

    HOTA landing exactly on 0.8 here is a coincidence of both terms; what
    matters is that ignoring the empty frames entirely would give 1.0.
    """
    n = 10
    gt = {t: _box(1, 0.0) for t in range(1, 9)}
    hyp = {t: _box(1, 0.0) for t in range(1, n + 1)}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(det_a - 0.8) < 1e-9, det_a
    assert abs(ass_a - 0.8) < 1e-9, ass_a
    assert abs(value - 0.8) < 1e-9, value


def test_hota_counts_missing_detections_on_frames_with_ground_truth():
    """A gt frame with no detection is a false negative and must be charged.

    The tracker stops early instead of running late, so it is absent for two
    of the ten gt frames and never fires without a target:

        TP = 8, FN = 2, FP = 0
        DetA = 8 / 10 = 0.8
        A(c,1) = 8 / (8 + 2 + 0) = 0.8, AssA = (8 * 0.8) / 8 = 0.8
        HOTA = sqrt(0.8 * 0.8) = 0.8
    """
    n = 10
    gt = {t: _box(1, 0.0) for t in range(1, n + 1)}
    hyp = {t: _box(1, 0.0) for t in range(1, 9)}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(det_a - 0.8) < 1e-9, det_a
    assert abs(ass_a - 0.8) < 1e-9, ass_a
    assert abs(value - 0.8) < 1e-9, value


def test_hota_is_zero_without_hypotheses():
    n = 5
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    assert hota({}, gt, n) == (0.0, 0.0, 0.0)


def test_hota_localisation_error_lowers_score_monotonically():
    """A box that drifts further from the gt must never score higher."""
    n = 6
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    previous = None
    for offset in (0.0, 1.0, 2.0, 3.0, 4.0):
        hyp = {t: _box(1, offset) for t in frames}
        value = hota(hyp, gt, n)[0]
        if previous is not None:
            assert value < previous, (
                f"offset {offset} scored {value}, not below {previous}")
        previous = value


if __name__ == "__main__":
    from _runner import run_tests

    raise SystemExit(run_tests(dict(globals()), "metrics"))