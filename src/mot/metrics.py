"""CLEAR-MOT and HOTA evaluation for MOT-format tracking results.

Three details here differ from a naive ``motmetrics`` call and each one matters
for honest numbers on MOT17:

1. **Ignore regions.** The MOTChallenge devkit does not count a false positive
   that lands inside an annotated distractor/ignored region. ``motmetrics`` has
   no concept of that, so detections that mostly overlap an ignore region are
   dropped from the hypotheses before scoring. Without this, MOTA is
   systematically pessimistic — MOT17 contains thousands of such regions.
2. **Class filter.** The released ground truth uses ``conf == 1`` only for plain
   pedestrians; crowds, riders and static people carry ``conf == 0``. Filtering
   on confidence therefore silently drops most targets. We filter on the MOT17
   "consider" class list instead (see :mod:`mot.dataset`).
3. **Aggregation.** Two schemes, and conflating them is a silent bug. The
   MOTChallenge rates are *pooled*: MOTA/precision/recall are recomputed once from
   the summed event counts, MOTP is weighted by matches, IDF1 is recomputed from
   pooled ``idtp``/``idfp``/``idfn``, and count metrics are summed. Averaging MOTA
   over sequences instead weights a 525-frame sequence equally with a 3750-frame
   one and can even disagree in sign. HOTA and its components have no count to
   pool, so they are an unweighted sequence mean -- which is *not* TrackEval's
   ``COMBINED_SEQ``, which pools per-alpha TP counts and association sums and
   recomputes. So a pooled ``OVERALL`` MOTA is a valid TrackEval-style combined
   score while ``OVERALL`` HOTA is a different quantity; do not quote it as one.

``evaluate_identity_check`` is the guard rail: feeding ground truth back in as
hypotheses must return MOTA 1.0, HOTA 1.0 and zero ID switches. Any deviation
means the loader or the scoring path is broken, not the tracker.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

from .dataset import read_gt, read_ignore_regions

#: Metrics pulled from the motmetrics accumulator.
_MM_METRICS = [
    "mota", "motp", "idf1", "idtp", "idfp", "idfn", "num_switches",
    "num_fragmentations", "num_misses", "num_false_positives", "num_objects",
    "num_matches", "mostly_tracked", "mostly_lost", "num_frames",
]

#: HOTA is averaged over these IoU thresholds, matching TrackEval's default.
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)


@dataclass
class TrackMetrics:
    """Flat MOT metrics for one sequence or one aggregate experiment."""

    sequence: str = ""
    mota: float = 0.0
    motp: float = 0.0
    hota: float = 0.0
    hota_det: float = 0.0
    hota_ass: float = 0.0
    idf1: float = 0.0
    idtp: float = 0.0
    idfp: float = 0.0
    idfn: float = 0.0
    num_switches: float = 0.0
    num_fragmentations: float = 0.0
    mostly_tracked: float = 0.0
    mostly_lost: float = 0.0
    num_matches: float = 0.0
    num_misses: float = 0.0
    num_false_positives: float = 0.0
    num_objects: float = 0.0
    num_frames: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# box helpers
# --------------------------------------------------------------------------

def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def iou_matrix(det_boxes_xywh, gt_boxes_xywh):
    """IoU between ``(N,4)`` and ``(M,4)`` xywh boxes."""
    d = np.asarray(det_boxes_xywh, dtype=np.float32).reshape(-1, 4)
    g = np.asarray(gt_boxes_xywh, dtype=np.float32).reshape(-1, 4)
    if d.shape[0] == 0 or g.shape[0] == 0:
        return np.zeros((d.shape[0], g.shape[0]), dtype=np.float32)
    dxy = xywh_to_xyxy(d)
    gxy = xywh_to_xyxy(g)
    area_d = np.clip(dxy[:, 2] - dxy[:, 0], 1e-6, None) * \
        np.clip(dxy[:, 3] - dxy[:, 1], 1e-6, None)
    area_g = np.clip(gxy[:, 2] - gxy[:, 0], 1e-6, None) * \
        np.clip(gxy[:, 3] - gxy[:, 1], 1e-6, None)
    lt = np.maximum(dxy[:, None, :2], gxy[None, :, :2])
    rb = np.minimum(dxy[:, None, 2:], gxy[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_d[:, None] + area_g[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def inside_fraction(boxes_xywh, region_xywh):
    """Fraction of each box's area contained in ``region_xywh``."""
    x, y, w, h = [float(v) for v in region_xywh]
    if w <= 0 or h <= 0:
        return np.zeros(len(boxes_xywh), dtype=np.float32)
    x1 = np.clip(boxes_xywh[:, 0], x, x + w)
    y1 = np.clip(boxes_xywh[:, 1], y, y + h)
    x2 = np.clip(boxes_xywh[:, 0] + boxes_xywh[:, 2], x, x + w)
    y2 = np.clip(boxes_xywh[:, 1] + boxes_xywh[:, 3], y, y + h)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes_xywh[:, 2], 1e-6, None) * \
        np.clip(boxes_xywh[:, 3], 1e-6, None)
    return (inter / area).astype(np.float32)


def filter_ignored(hypotheses, ignore_regions, frac_thresh=0.5):
    """Drop hypothesis rows whose box mostly overlaps an ignore region.

    Only the frame-local regions are consulted, which keeps this linear in the
    number of annotated regions rather than quadratic in detections.
    """
    if not ignore_regions:
        return hypotheses
    out = {}
    for frame_id, boxes in hypotheses.items():
        regions = ignore_regions.get(frame_id)
        if regions is None or boxes.shape[0] == 0:
            out[frame_id] = boxes
            continue
        keep = np.ones(boxes.shape[0], dtype=bool)
        for region in regions:
            keep &= inside_fraction(boxes[:, 1:5], region[1:5]) <= frac_thresh
        out[frame_id] = boxes[keep]
    return out


# --------------------------------------------------------------------------
# HOTA
# --------------------------------------------------------------------------

def hota(hypotheses, gt, num_frames, alphas=HOTA_ALPHAS):
    """Single-class HOTA, DetA and AssA averaged over ``alphas``.

    This is the TrackEval formulation, transcribed from
    ``trackeval/metrics/hota.py`` and validated two ways: against three
    hand-computed cases (perfect identity, one switch, a new id per frame) and
    against an independent transcription of the same algorithm kept in
    ``tests/test_metrics.py``.

    MOT17 is scored as one "person" class, so boxes carry only a track id in
    column 0 and the class sums collapse. For IoU threshold ``a``:

        TPA(c,t,a) = frames where gt ``c`` was matched to tracker ``t``
        FNA(c,a)   = frames where gt ``c`` was present but unmatched
        FPA(c,t,a) = frames where tracker ``t`` fired but not onto gt ``c``
        A(c,t,a)   = TPA / (TPA + FNA + FPA)          # Jaccard, per id pair
        AssA(a)    = sum_{c,t} TPA * A / TP
        DetA(a)    = TP / (TP + FN + FP)
        HOTA(a)    = sqrt(DetA * AssA)

    ``A`` is the whole point of HOTA over DetA: a tracker that finds every box
    but hands each gt a different id scores AssA -> 1/TP, not 1.0.

    Two implementation details are load-bearing and easy to get backwards.

    First, ``FPA`` is *not* a count of false positives -- it counts a tracker's
    detections on frames where the gt it had claimed was not its match. The
    Jaccard denominator ``gt_id_count + tracker_id_count - matches`` expresses
    that directly, since ``gt_id_count`` is ``TPA + FNA`` and
    ``tracker_id_count`` is ``TPA + FPA``.

    Second, the per-frame Hungarian assignment is driven by
    ``global_alignment_score * IoU``, not by IoU alone. The global alignment is
    a first pass over the whole sequence that accumulates a Jaccard-ish
    preference for each (gt id, tracker id) pair, so a track that has been
    consistently matching one person wins ties against a track that happens to
    overlap better on one frame. Matching on IoU alone lets identity flip
    between two nearby people frame to frame.

    Similarity is indexed as ``(num_gt_dets, num_tracker_dets)``, matching
    TrackEval; transposing it silently produces a plausible wrong answer.
    """
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    if alphas.size == 0:
        raise ValueError("alphas must contain at least one threshold")
    n_alpha = alphas.size
    eps = np.finfo("float").eps

    gt_index = {}
    for boxes in gt.values():
        for row in boxes:
            gt_index.setdefault(int(row[0]), len(gt_index))
    hyp_index = {}
    for boxes in hypotheses.values():
        for row in boxes:
            hyp_index.setdefault(int(row[0]), len(hyp_index))
    num_gt_ids = len(gt_index)
    num_hyp_ids = len(hyp_index)
    if num_gt_ids == 0 or num_hyp_ids == 0:
        return 0.0, 0.0, 0.0

    # Per-frame id indices plus IoU, in one pass, shared by both passes below.
    frames = []
    for frame_id in range(1, num_frames + 1):
        g = gt.get(frame_id)
        d = hypotheses.get(frame_id)
        g_rows = np.zeros((0, 4), dtype=np.float32)
        g_ids = np.zeros(0, dtype=np.int64)
        if g is not None and g.shape[0]:
            g_ids = np.fromiter((gt_index[int(v)] for v in g[:, 0]),
                                dtype=np.int64, count=g.shape[0])
            g_rows = g[:, 1:5]
        h_ids = np.zeros(0, dtype=np.int64)
        h_rows = np.zeros((0, 4), dtype=np.float32)
        if d is not None and d.shape[0]:
            h_ids = np.fromiter((hyp_index[int(v)] for v in d[:, 0]),
                                dtype=np.int64, count=d.shape[0])
            h_rows = d[:, 1:5]
        if g_ids.size and h_ids.size:
            # iou_matrix(a, b) is (len(a), len(b)), so this is already
            # (num_gt_dets, num_tracker_dets) as TrackEval indexes it.
            # NaN/inf coordinates would poison the whole assignment below
            # (scipy raises on non-finite entries), and an unmatchable box
            # scoring zero is the same convention distance_matrix uses.
            sim = np.nan_to_num(iou_matrix(g_rows, h_rows).astype(np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)
        else:
            sim = np.zeros((g_ids.size, h_ids.size), dtype=np.float64)
        frames.append((g_ids, h_ids, sim))

    # Pass 1: accumulate the global alignment score over the whole sequence.
    potential = np.zeros((num_gt_ids, num_hyp_ids), dtype=np.float64)
    gt_id_count = np.zeros((num_gt_ids, 1), dtype=np.float64)
    hyp_id_count = np.zeros((1, num_hyp_ids), dtype=np.float64)
    for g_ids, h_ids, sim in frames:
        if g_ids.size and h_ids.size:
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_jaccard = np.zeros_like(sim)
            mask = denom > eps
            sim_jaccard[mask] = sim[mask] / denom[mask]
            # np.add.at, not fancy-index +=: a repeated id inside one frame
            # must accumulate once per row, not once per unique index.
            # TrackEval assumes unique ids per frame and gets this for free.
            np.add.at(potential, (g_ids[:, None], h_ids[None, :]), sim_jaccard)
        np.add.at(gt_id_count, (g_ids, np.zeros_like(g_ids)), 1.0)
        np.add.at(hyp_id_count, (np.zeros_like(h_ids), h_ids), 1.0)
    align_denom = gt_id_count + hyp_id_count - potential
    with np.errstate(divide="ignore", invalid="ignore"):
        global_align = np.where(align_denom > eps, potential / align_denom, 0.0)
    global_align[~np.isfinite(global_align)] = 0.0

    # Pass 2: match each frame, then score. ``tp_count`` is shared across
    # alphas because the assignment is driven by the alignment score and does
    # not depend on the threshold -- only the thresholding of its output does.
    tp = np.zeros(n_alpha)
    fn = np.zeros(n_alpha)
    fp = np.zeros(n_alpha)
    matches_count = [np.zeros((num_gt_ids, num_hyp_ids), dtype=np.float64)
                     for _ in range(n_alpha)]

    for g_ids, h_ids, sim in frames:
        if not g_ids.size:
            fp += len(h_ids)
            continue
        if not h_ids.size:
            fn += len(g_ids)
            continue
        score = global_align[g_ids[:, None], h_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for ai, alpha in enumerate(alphas):
            keep = sim[rows, cols] >= alpha - eps
            n_match = int(keep.sum())
            tp[ai] += n_match
            fn[ai] += len(g_ids) - n_match
            fp[ai] += len(h_ids) - n_match
            if n_match:
                matches_count[ai][g_ids[rows[keep]], h_ids[cols[keep]]] += 1.0

    ass_a = np.zeros(n_alpha)
    det_a = np.zeros(n_alpha)
    for ai in range(n_alpha):
        m = matches_count[ai]
        pair_score = m / np.maximum(1.0, gt_id_count + hyp_id_count - m)
        ass_a[ai] = np.sum(m * pair_score) / max(1.0, tp[ai])
        det_a[ai] = tp[ai] / max(1.0, tp[ai] + fn[ai] + fp[ai])
    hota_vals = np.sqrt(det_a * ass_a)
    return (float(hota_vals.mean()), float(det_a.mean()), float(ass_a.mean()))


def distance_matrix(det_boxes_xywh, gt_boxes_xywh, distth=0.5):
    """CLEAR-MOT distance matrix, ``(num_gt, num_det)``, ``NaN`` = do-not-pair.

    ``motmetrics.distances.iou_matrix`` cannot be used here because it calls
    ``np.asfarray``, removed in NumPy 2.0, and this project runs NumPy 2 with a
    CUDA 12/13 build of PyTorch. Computing the matrix locally keeps the exact
    semantics ``MOTAccumulator.update`` documents: finite entries are candidate
    pairs, ``NaN`` entries are forbidden pairs.
    """
    ious = iou_matrix(det_boxes_xywh, gt_boxes_xywh).T  # -> (num_gt, num_det)
    if ious.size == 0:
        return np.zeros((len(np.atleast_2d(gt_boxes_xywh)),
                         len(np.atleast_2d(det_boxes_xywh))), dtype=np.float64)
    dists = 1.0 - ious.astype(np.float64)
    dists[ious < distth] = np.nan
    return dists


def _num_id(boxes_by_frame):
    """Number of distinct track ids across all frames."""
    ids = set()
    for boxes in boxes_by_frame.values():
        if boxes.shape[0]:
            ids.update(int(v) for v in boxes[:, 0])
    return len(ids)


def _max_id(*dicts):
    """Largest track id seen in any of the given ``frame -> boxes`` dicts."""
    best = 0
    for d in dicts:
        for boxes in d.values():
            if boxes is not None and boxes.shape[0]:
                best = max(best, int(np.max(boxes[:, 0])))
    return best


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------

def evaluate(hypotheses, gt_path, sequence="", distth=0.5, use_ignore=True,
             name=None, with_hota=True, max_frame=None):
    """Score ``hypotheses`` against one sequence's ground truth.

    Args:
        hypotheses: ``frame_id -> (N, 6)`` array of ``[id, x, y, w, h, score]``.
        gt_path: path to ``gt.txt``.
        distth: IoU distance threshold, i.e. 0.5 means 50% overlap required.
        use_ignore: suppress detections inside annotated ignore regions.
        max_frame: last frame to score. Defaults to the last frame present in
            either hypotheses or the ground truth. Pass it explicitly when
            scoring a truncated run, otherwise every untracked trailing frame
            is charged as a miss and recall collapses.
    """
    gt = read_gt(gt_path)
    if use_ignore:
        hypotheses = filter_ignored(hypotheses, read_ignore_regions(gt_path))

    if max_frame is None:
        max_frame = max(list(hypotheses.keys()) + list(gt.keys()) + [1])
    max_frame = int(max_frame)
    acc = mm.MOTAccumulator(auto_id=False)
    empty = np.zeros((0, 6), dtype=np.float32)
    for frame_id in range(1, max_frame + 1):
        d = hypotheses.get(frame_id, empty)
        g = gt.get(frame_id, empty)
        dists = distance_matrix(d[:, 1:5], g[:, 1:5], distth=distth)
        acc.update(g[:, 0].astype(int), d[:, 0].astype(int), dists, frameid=frame_id)

    host = mm.metrics.create()
    summary = host.compute(acc, metrics=_MM_METRICS, name=name or sequence or "seq",
                           return_dataframe=False)

    out = TrackMetrics(sequence=sequence)
    out.mota = float(summary["mota"]) * 100.0
    # ``motmetrics`` reports MOTP as the *mean matching distance* (lower is
    # better). MOTChallenge and every leaderboard publish ``1 - distance`` as a
    # percentage, so convert rather than mislabel a distance as precision.
    out.motp = (1.0 - float(summary["motp"])) * 100.0
    out.idf1 = float(summary["idf1"])
    for key in ("num_switches", "num_fragmentations", "num_misses",
                "num_false_positives", "num_matches", "mostly_tracked",
                "mostly_lost", "num_objects", "num_frames",
                "idtp", "idfp", "idfn"):
        out.__dict__[key] = float(summary[key])

    # Precision and recall are recomputed here instead of taken from the
    # library: its ``recall`` divides "matches + switches" by its own
    # ``num_objects``, which disagrees with the ``num_objects`` it reports, so
    # the two numbers cannot both be right. CLEAR defines them directly from
    # the event counts, which is what we want and what is checkable.
    matches = out.num_matches
    denominator = matches + out.num_false_positives
    out.precision = 100.0 * matches / denominator if denominator else 0.0
    denominator = matches + out.num_misses
    out.recall = 100.0 * matches / denominator if denominator else 0.0

    if with_hota:
        # Stored as percentages to match MOTA/MOTP, so every rate in this
        # dataclass is on the same 0-100 scale except IDF1 which is 0-1.
        h, det, ass = hota(hypotheses, gt, max_frame)
        out.hota, out.hota_det, out.hota_ass = h * 100.0, det * 100.0, ass * 100.0
    return out


def aggregate(results):
    """Combine per-sequence metrics into one OVERALL row.

    Two different aggregations are used, and conflating them is a silent bug:

    **Pooled** (single global confusion matrix over every sequence) for the
    MOTChallenge rates and counts. These are ratios of summed event counts, so
    they must be recomputed from the pooled counts, not averaged. Averaging
    MOTA over sequences implicitly weights a 525-frame sequence equally with a
    1050-frame one, and it is why the mean can even disagree in *sign* with the
    pooled value. ``motp`` is pooled weighted by matches, because it is a mean
    localization cost over matched boxes. ``idf1`` likewise comes from pooled
    ``idtp``/``idfp``/``idfn``.

    **Mean** for the scores with no event count to pool: ``hota`` and its
    components, ``mostly_tracked``, ``mostly_lost``. ``num_objects`` and
    ``num_frames`` are summed.
    """
    if not results:
        return TrackMetrics(sequence="OVERALL")

    averaged = ("hota", "hota_det", "hota_ass", "mostly_tracked", "mostly_lost")
    summed = ("num_switches", "num_fragmentations", "num_misses",
              "num_false_positives", "num_matches", "num_objects",
              "num_frames", "idtp", "idfp", "idfn")

    out = TrackMetrics(sequence="OVERALL")
    for name in averaged:
        setattr(out, name, float(np.mean([getattr(r, name) for r in results])))
    for name in summed:
        setattr(out, name, float(np.sum([getattr(r, name) for r in results])))
    out.extra["num_sequences"] = len(results)

    # Pooled MOTChallenge rates. Each mirrors the per-sequence formula in
    # evaluate(), applied once to the summed counts.
    errors = out.num_misses + out.num_false_positives + out.num_switches
    out.mota = 100.0 * (1.0 - errors / out.num_objects) if out.num_objects else 0.0
    denominator = out.num_matches + out.num_false_positives
    out.precision = (100.0 * out.num_matches / denominator
                     if denominator else 0.0)
    denominator = out.num_matches + out.num_misses
    out.recall = 100.0 * out.num_matches / denominator if denominator else 0.0
    # MOTP is a mean over matched boxes, so weight by matches.
    weights = np.asarray([r.num_matches for r in results], dtype=np.float64)
    if weights.sum() > 0:
        out.motp = float(np.average([r.motp for r in results], weights=weights))
    else:
        out.motp = float(np.mean([r.motp for r in results]))
    denom = 2 * out.idtp + out.idfp + out.idfn
    if denom > 0:
        out.idf1 = float(2 * out.idtp / denom)
    return out


def format_table(results, include_overall=None):
    """Render a compact per-sequence metric table.

    The OVERALL row is appended automatically unless ``include_overall`` says
    otherwise; it defaults to on only when there is more than one sequence, so
    a single-sequence run does not print its own row twice (once from the loop,
    once from ``aggregate`` of a one-element list) and the final summary print
    can skip it.
    """
    if include_overall is None:
        include_overall = len(results) > 1
    header = (f"{'sequence':<20}{'MOTA':>8}{'MOTP':>8}{'HOTA':>8}{'IDF1':>8}"
              f"{'IDS':>7}{'Frag':>7}{'MT':>6}{'ML':>6}{'Rec':>7}{'Prec':>7}")
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.sequence:<20}{r.mota:>8.2f}{r.motp:>8.2f}{r.hota:>8.2f}"
            f"{r.idf1:>8.3f}{r.num_switches:>7.0f}{r.num_fragmentations:>7.0f}"
            f"{r.mostly_tracked:>6.0f}{r.mostly_lost:>6.0f}"
            f"{r.recall:>7.2f}{r.precision:>7.2f}"
        )
    if include_overall:
        agg = aggregate(results)
        lines.append("-" * len(header))
        lines.append(
            f"{'OVERALL':<20}{agg.mota:>8.2f}{agg.motp:>8.2f}{agg.hota:>8.2f}"
            f"{agg.idf1:>8.3f}{agg.num_switches:>7.0f}{agg.num_fragmentations:>7.0f}"
            f"{agg.mostly_tracked:>6.0f}{agg.mostly_lost:>6.0f}"
            f"{agg.recall:>7.2f}{agg.precision:>7.2f}"
        )
    return "\n".join(lines)


def save_json(results, path, config=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": asdict(aggregate(results)),
        "sequences": [asdict(r) for r in results],
        "config": config or {},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def evaluate_identity_check(gt_path):
    """Ground truth as hypotheses must score perfectly; anything else is a bug.

    Returns ``(metrics, ok)``.
    """
    gt = read_gt(gt_path)
    hypotheses = {k: v.copy() for k, v in gt.items()}
    metrics = evaluate(hypotheses, gt_path, sequence="identity", use_ignore=False)
    ok = (metrics.mota > 99.9 and metrics.num_switches == 0
          and metrics.hota > 99.0 and metrics.idf1 > 0.999)
    return metrics, ok