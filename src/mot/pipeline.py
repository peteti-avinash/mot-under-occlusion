"""End-to-end MOT17/MOT20 benchmark pipeline.

Runs a pretrained detector, a tracker, and the evaluator over one or more
sequences, and writes MOT-format result files plus a metrics summary.

The tracker is deliberately swappable so the ablation in the write-up can hold
the detector fixed and vary only the tracker:

    python -m mot.pipeline --sequences all --tracker occlusion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python src/mot/pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.appearance import AppearanceEncoder
from mot.dataset import (
    MOT17_EVAL_CLASSES,
    find_dataset_root,
    list_sequences,
    load_dets_cache,
    read_frame,
    save_dets_cache,
    write_results,
)
from mot.detector import YoloDetector, clip_boxes, xyxy_to_tlwh
from mot.metrics import aggregate, evaluate, format_table, save_json
from mot.occlusion import OcclusionAwareTracker
from mot.sort import SortTracker
from mot.track import Detection

TRACKERS = {"sort": SortTracker, "occlusion": OcclusionAwareTracker}


def build_tracker(name, seq, args):
    """Construct a tracker with per-sequence settings."""
    cls = TRACKERS[name]
    kwargs = dict(
        max_age=args.max_age,
        min_hits=args.min_hits,
        det_conf_threshold=args.det_conf_threshold,
        frame_rate=seq.fps,
    )
    if name == "sort":
        kwargs["iou_threshold"] = args.sort_iou_threshold
    else:
        kwargs.update(
            appearance_weight=args.appearance_weight,
            gallery_size=args.gallery_size,
            reid_max_age=args.reid_max_age,
            enable_reid=not args.no_reid,
            cascade=not args.no_cascade,
        )
    return cls(**kwargs)


def run_sequence(seq, args, detector, encoder, tracker_name, cache_dir=None):
    """Track one sequence.

    Returns ``(rows, hypotheses, wall_seconds)`` where ``rows`` are MOT-format
    output rows and ``hypotheses`` is the ``frame_id -> (N, 6)`` array layout the
    evaluator expects.

    When ``cache_dir`` is set, per-frame detections are read from (and written
    to) ``<cache_dir>/<seq.name>.npz`` so every tracker config runs on
    byte-identical detector output; a cache that covers fewer frames than the
    sequence is treated as stale and ignored.
    """
    cache = {}
    dets_cache = load_dets_cache(cache_dir, seq.name, seq.length)
    fresh_dets = []
    tracker = build_tracker(tracker_name, seq, args)
    tracker.reset()

    rows = []
    hypotheses = {}
    t0 = time.perf_counter()
    report_every = max(1, seq.length // 5)
    for frame_id in range(1, seq.length + 1):
        if dets_cache is not None and frame_id in dets_cache:
            boxes = dets_cache[frame_id][:, :4]
            scores = dets_cache[frame_id][:, 4]
        else:
            image = read_frame(seq, frame_id, cache)
            boxes_xyxy, scores, _ = detector.detect(image)
            boxes = clip_boxes(xyxy_to_tlwh(boxes_xyxy),
                               image.shape[1], image.shape[0])
            if cache_dir is not None:
                fresh_dets.append((frame_id, boxes, scores))

        features = None
        if encoder is not None and boxes.shape[0]:
            image = read_frame(seq, frame_id, cache)
            features = encoder.encode_batch(image, boxes)

        detections = [
            Detection(boxes[i], float(scores[i]),
                      None if features is None else features[i])
            for i in range(boxes.shape[0])
        ]

        tracker.update(detections, frame_id=frame_id)

        frame_rows = []
        for t in tracker.get_active_tracks():
            if t.hits < tracker.min_hits:
                continue
            x, y, w, h = (float(v) for v in t.to_tlwh())
            frame_rows.append((t.track_id, x, y, w, h, 1.0))
            # write_results takes the 6 MOT columns (frame,id,x,y,w,h) and
            # appends the confidence plus the two trailing placeholder columns.
            rows.append((frame_id, t.track_id, x, y, w, h))
        if frame_rows:
            hypotheses[frame_id] = np.asarray(frame_rows, dtype=np.float32)

        if frame_id % report_every == 0:
            done = time.perf_counter() - t0
            print(f"  {frame_id}/{seq.length} frames  "
                  f"{done:.0f}s  ({frame_id / done:.1f} fps)  "
                  f"{len(detections)} dets  {len(tracker.tracks)} tracks",
                  flush=True)
    if cache_dir is not None and fresh_dets:
        save_dets_cache(cache_dir, seq.name, fresh_dets, seq.length)
    return rows, hypotheses, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--sequences", default="all",
                    help="'all', a comma-separated list, or a single name")
    ap.add_argument("--detector", default="FRCNN", choices=["FRCNN", "DPM", "ALL"])
    ap.add_argument("--tracker", default="occlusion", choices=sorted(TRACKERS))
    # Detector operating point measured on MOT17-02 (60 frames, IoU>=0.5,
    # MOT17 eval classes): yolo11x @ 1600 reaches 45.5% recall / 68.0%
    # precision at conf 0.15, versus 29.4% / 72.9% for yolo11n @ 960. MOT17
    # pedestrians are small, so input height buys more recall than model size
    # does, and the whole benchmark is only ~8k frames, which the larger model
    # can afford.
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--device", default=0)
    # Only used for *starting* new tracks; lower-score detections still
    # sustain existing ones, so keep this below the detector threshold.
    ap.add_argument("--det-conf-threshold", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--sort-iou-threshold", type=float, default=0.3)
    ap.add_argument("--appearance-weight", type=float, default=0.6)
    ap.add_argument("--gallery-size", type=int, default=30)
    ap.add_argument("--reid-max-age", type=int, default=45)
    ap.add_argument("--no-appearance", action="store_true",
                    help="ablation: disable the appearance cue")
    ap.add_argument("--no-reid", action="store_true",
                    help="ablation: disable the Lost-track re-identification stage")
    ap.add_argument("--no-cascade", action="store_true",
                    help="ablation: replace the age-ordered cascade with one "
                         "global Hungarian assignment")
    ap.add_argument("--dets-cache", default=None,
                    help="directory of cached detection .npz files; cached "
                         "frames replace live detection so every tracker "
                         "config sees byte-identical input")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default=None,
                    help="output subdirectory; defaults to the tracker name")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="debug: stop after N frames per sequence")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    root = find_dataset_root(args.dataset_root)
    detector_key = "FRCNN" if args.detector == "ALL" else args.detector
    seqs = list_sequences(root, detector=detector_key)
    if args.sequences != "all":
        wanted = {w.strip() for w in args.sequences.split(",") if w.strip()}
        def matches(seq):
            if seq.name in wanted:
                return True
            # Accept the short form ("MOT17-05"), a bare index ("05"), and the
            # full folder name, so `--sequences 05` behaves as expected.
            short = seq.name.split("-")[1] if "-" in seq.name else seq.name
            return short in wanted or seq.name.split("-")[-1] in wanted
        seqs = [s for s in seqs if matches(s)]
    if not seqs:
        print("no sequences matched", file=sys.stderr)
        return 2
    if args.max_frames:
        for s in seqs:
            s.length = min(s.length, args.max_frames)

    tag = args.tag or args.tracker
    out_root = Path(args.out_dir) / tag
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"dataset {root}")
    print(f"{len(seqs)} sequence(s), detector {args.weights}, tracker {args.tracker}")
    print(f"eval classes {MOT17_EVAL_CLASSES}")

    detector = YoloDetector(weights=args.weights, device=args.device,
                            conf=args.conf, imgsz=args.imgsz)
    # SORT never looks at features, so the baseline must not pay for the CNN or
    # inherit any of its nondeterminism: the ablation is only clean if the
    # detector is the single shared component.
    use_appearance = not args.no_appearance and args.tracker == "occlusion"
    encoder = AppearanceEncoder(device="cuda") if use_appearance else None
    print(f"appearance={use_appearance}")
    if encoder is not None:
        print(f"appearance dim {encoder.dim}")

    results = []
    for seq in seqs:
        print(f"\n=== {seq.name} ({seq.length} frames) ===", flush=True)
        rows, hyp, secs = run_sequence(seq, args, detector, encoder,
                                       args.tracker, cache_dir=args.dets_cache)
        fps = seq.length / secs if secs else 0.0
        timing_label = ("detector+tracker" if encoder is None
                        else "detector+tracker+appearance")
        print(f"tracked in {secs:.1f}s ({fps:.1f} fps, {timing_label})")

        metrics = evaluate(hyp, seq.gt_file, sequence=seq.name, max_frame=seq.length)
        print(format_table([metrics]))
        results.append(metrics)

        if not args.no_write:
            path = out_root / f"{seq.name}.txt"
            write_results(path, rows)
            print(f"wrote {len(rows)} rows -> {path}")

    overall = aggregate(results)
    print("\n=== OVERALL ===")
    print(format_table([overall], include_overall=False))

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    config["appearance_dim"] = 0 if encoder is None else encoder.dim
    path = save_json(results, out_root / "metrics.json", config=config)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
