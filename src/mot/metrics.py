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
3. **Aggregation.** MOTA/MOTP/HOTA are averaged over sequences, the way the
   leaderboard does it, while ID switches and fragmentations are summed because
   they are counts.

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

    Written out from the HOTA paper so the definition is auditable. MOT17 is
    scored as one "person" class, so boxes carry only a track id in column 0
    and the class sums collapse. For IoU threshold ``a``:

        DetA(t,a) = TP / (TP + FN + FP)
        AssA(c,t,a) = TP / (TP + H + M)        # for each matched gt c
        A(a)      = sum_t sum_c DetA(t,a) * AssA(c,t,a)
        HOTA(a)   = sqrt(A(a)) / (sqrt(TP) + sqrt(FN) + sqrt(FP))
        DetA(a)   = TP / (TP + FN + FP)
        AssA(a)   = A(a) / TP

    ``H`` counts identity switches: gt ``c`` being matched to a track id
    different from the one it was matched to previously. ``M`` counts a gt
    matched to two detections in the same frame, which 1-to-1 Hungarian
    matching makes impossible, so ``M`` is always 0. ``H`` is the entire point
    of HOTA over DetA -- a tracker that finds every box but scrambles
    identities must not score 1.0 -- so it must not be dropped.

    Note this is the paper's single-class form. TrackEval aggregates the
    association term slightly differently (a log-Jaccard over global id pairs),
    so absolute HOTA will differ by a small margin from published TrackEval
    numbers while ranking order is preserved.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    n_alpha = alphas.size
    num_gt_ids = _max_id(gt, {}) + 1
    if num_gt_ids == 0:
        return 0.0, 0.0, 0.0

    # Presence masks, so we can ask "was gt c annotated on frame t".
    gt_present = np.zeros((num_gt_ids, num_frames), dtype=bool)
    for frame_id, boxes in gt.items():
        if frame_id > num_frames or boxes.shape[0] == 0:
            continue
        for row in boxes:
            gt_present[int(row[0]), frame_id - 1] = True
    

    hota_vals = np.zeros(n_alpha, dtype=np.float64)
    det_a_vals = np.zeros(n_alpha, dtype=np.float64)
    ass_a_vals = np.zeros(n_alpha, dtype=np.float64)

    for ai, alpha in enumerate(alphas):
        tp_total = 0
        fn_total = 0
        fp_total = 0
        a_total = 0.0
        # gt id -> track id it was last matched to, for the H (identity switch)
        # penalty. Reset per alpha: a different threshold is a different
        # matching, so switches are recounted for each one.
        last_track_for_gt = {}

        for frame_id in range(1, num_frames + 1):
            g = gt.get(frame_id)
            d = hypotheses.get(frame_id)
            ng = 0 if g is None else g.shape[0]
            nd = 0 if d is None else d.shape[0]
            n_gts = int(gt_present[:, frame_id - 1].sum())

            pairs = []
            switch = {}
            if ng and nd:
                ious = iou_matrix(d[:, 1:5], g[:, 1:5])
                viable = ious >= alpha
                # Maximise the number of viable pairs first, then IoU: pairs
                # below the threshold cost 0 so they are never preferred, and
                # viable pairs cost strictly negative.
                cost = np.where(viable, -(ious - alpha + 1.0), 0.0)
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if not viable[r, c]:
                        continue
                    gt_id = int(g[c, 0])
                    track_id = int(d[r, 0])
                    previous = last_track_for_gt.get(gt_id)
                    switch[gt_id] = (previous is not None and previous != track_id)
                    last_track_for_gt[gt_id] = track_id
                    pairs.append(gt_id)

            tp = len(pairs)
            fn = n_gts - tp
            fp = nd - tp
            tp_total += tp
            fn_total += fn
            fp_total += fp
            if tp:
                det_a = tp / (tp + fn + fp)
                for gt_id in pairs:
                    # M == 0 under 1-to-1 matching, so AssA = 1 / (1 + H).
                    a_total += det_a / (1.0 + float(switch[gt_id]))

        denom = np.sqrt(tp_total) + np.sqrt(fn_total) + np.sqrt(fp_total)
        hota_vals[ai] = np.sqrt(a_total) / denom if denom > 0 else 0.0
        dsum = tp_total + fn_total + fp_total
        det_a_vals[ai] = tp_total / dsum if dsum > 0 else 0.0
        ass_a_vals[ai] = a_total / tp_total if tp_total > 0 else 0.0

    return (float(hota_vals.mean()), float(det_a_vals.mean()),
            float(ass_a_vals.mean()))


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