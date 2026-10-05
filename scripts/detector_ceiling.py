"""Detector ceiling: how good can detection get, independent of the tracker?

Run this before trusting any tracker conclusion. If recall is the binding
constraint, then differences between trackers are mostly differences in how
they cope with missed detections, and the write-up should say so.

    PYTHONPATH=src ./venv/bin/python scripts/detector_ceiling.py \
        --sequence 02 --frames 60

Greedy one-to-one matching at IoU >= 0.5 against MOT17 ground truth, using the
same class filtering and ignore-region suppression as the evaluator, so the
numbers here are comparable with the pipeline's evaluation.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from mot.dataset import (find_dataset_root, list_sequences, read_frame,  # noqa: E402
                         read_gt, read_ignore_regions, frame_files)
from mot.detector import YoloDetector, xyxy_to_tlwh  # noqa: E402
from mot.metrics import inside_fraction  # noqa: E402

IOU_THRESHOLD = 0.5


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _corners(tlwh):
    """``[x, y, w, h]`` -> ``(x1, y1, x2, y2)``."""
    x, y, w, h = [float(v) for v in tlwh[:4]]
    return (x, y, x + w, y + h)


def _corners_xyxy(box):
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _greedy_match(dets_xyxy, gts_tlwh, iou_threshold=IOU_THRESHOLD):
    """Greedy highest-IoU-first one-to-one matching. Returns (tp, n_dets, n_gts)."""
    gts = [_corners_xyxy(_corners(g)) for g in gts_tlwh]
    pairs = []
    for di, det in enumerate(dets_xyxy):
        for gi, gt in enumerate(gts):
            value = _iou(_corners_xyxy(det), gt)
            if value >= iou_threshold:
                pairs.append((value, di, gi))
    pairs.sort(reverse=True)
    used_det, used_gt, tp = set(), set(), 0
    for _, di, gi in pairs:
        if di in used_det or gi in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gi)
        tp += 1
    return tp, len(dets_xyxy), len(gts_tlwh)


def _suppressed(dets_tlwh, regions):
    """True where a detection sits inside an annotated ignore region.

    ``inside_fraction`` takes a single region, so the per-region fractions are
    combined with a max.
    """
    if regions is None or len(regions) == 0 or len(dets_tlwh) == 0:
        return np.zeros(len(dets_tlwh), dtype=bool)
    best = np.zeros(len(dets_tlwh), dtype=np.float32)
    for row in regions:
        best = np.maximum(best, inside_fraction(dets_tlwh, row[1:5]))
    return best >= 0.5


def evaluate_config(detector, seq, frames, gt, ignore, batch_size=8):
    cache, images, ids = {}, [], []
    for frame_id in frames:
        images.append(read_frame(seq, frame_id, cache))
        ids.append(frame_id)

    tp = n_det = n_gt = 0
    start = 0
    while start < len(images):
        chunk = images[start:start + batch_size]
        results = detector.detect_batch(chunk)
        for offset, (xyxy, scores, classes) in enumerate(results):
            frame_id = ids[start + offset]
            # COCO class 0 is "person", which is what all four MOT17 evaluation
            # classes map onto.
            keep = np.asarray(classes) == 0 if len(xyxy) else np.zeros(0, bool)
            xyxy, scores = np.asarray(xyxy)[keep], np.asarray(scores)[keep]
            dets = xyxy_to_tlwh(xyxy) if len(xyxy) else np.zeros((0, 4), np.float32)
            # Suppressed detections must be dropped from *both* the numerator's
            # candidate set and the denominator's detection count, so the same
            # mask has to reach _greedy_match -- not just the tlwh copy.
            if len(dets):
                live_mask = ~_suppressed(dets, ignore.get(frame_id))
            else:
                live_mask = np.zeros(0, bool)
            live_xyxy = xyxy[live_mask]
            gt_boxes = gt.get(frame_id)
            gts = gt_boxes[:, 1:5] if gt_boxes is not None and len(gt_boxes) \
                else np.zeros((0, 4))
            a, b, c = _greedy_match(live_xyxy, gts)
            tp += a
            n_det += b
            n_gt += c
        start += batch_size

    recall = tp / n_gt if n_gt else 0.0
    precision = tp / n_det if n_det else 0.0
    return recall, precision, tp, n_det, n_gt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="02")
    ap.add_argument("--frames", type=int, default=60,
                    help="evaluate the first N frames only (debug speed)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--configs", default="default",
                    help="'default', 'size-sweep', 'conf-sweep', or comma-separated "
                         "weights:imgsz:conf triples")
    args = ap.parse_args()

    root = find_dataset_root(args.dataset_root)
    sequences = {s.name: s for s in list_sequences(root, detector="FRCNN")}
    if args.sequence not in sequences:
        raise SystemExit(f"sequence {args.sequence} not found under {root}; "
                         f"have {sorted(sequences)}")
    seq = sequences[args.sequence]

    total = min(args.frames, seq.length)
    frames = list(range(1, total + 1))
    gt = read_gt(seq.gt_file)
    ignore = read_ignore_regions(seq.gt_file)
    print(f"sequence {seq.name}: {total} frames, {seq.width}x{seq.height}\n")
    print(f"{'weights':14s} {'imgsz':>6s} {'conf':>6s} "
          f"{'recall':>8s} {'prec':>8s} {'tp':>6s} {'det':>6s} {'gt':>6s}")

    if args.configs == "default":
        configs = [("yolo11x.pt", 1600, 0.25)]
    elif args.configs == "size-sweep":
        configs = [("yolo11n.pt", 1280, 0.25), ("yolo11m.pt", 1280, 0.25),
                   ("yolo11l.pt", 1280, 0.25), ("yolo11x.pt", 1600, 0.25)]
    elif args.configs == "conf-sweep":
        configs = [("yolo11x.pt", 1600, c) for c in (0.20, 0.25, 0.30, 0.35, 0.40)]
    else:
        configs = []
        for item in args.configs.split(","):
            weights, imgsz, conf = item.split(":")
            configs.append((weights, int(imgsz), float(conf)))

    for weights, imgsz, conf in configs:
        if not os.path.exists(weights):
            print(f"  {weights:14s} SKIPPED (missing weights; see README)")
            continue
        # imgsz has to be passed explicitly: the sweep varies it, and leaving it
        # at YoloDetector's default would silently run every config at 1600.
        detector = YoloDetector(weights=weights, device=args.device, conf=conf,
                                imgsz=imgsz)
        recall, precision, tp, n_det, n_gt = evaluate_config(
            detector, seq, frames, gt, ignore, args.batch_size)
        print(f"{weights:14s} {imgsz:6d} {conf:6.2f} "
              f"{recall * 100:7.2f}% {precision * 100:7.2f}% "
              f"{tp:6d} {n_det:6d} {n_gt:6d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())