"""Run optional external trackers on identical cached detections.

This script runs the BoxMOT implementations of ByteTrack, OC-SORT and BoT-SORT
with their default parameters on the same yolo11x detections this repository's
trackers see (shared ``--dets-cache``), then scores every tracker with this
repository's evaluator:

    PYTHONPATH=src ./venv/bin/python scripts/run_baselines.py \
        --sequences all --dets-cache results/dets_cache --out results/sota

Using the vendors' defaults is deliberate: tuned baselines invite the
suspicion of a strawman, and defaults are what a reader can reproduce from
the boxmot docs. BoT-SORT runs without its ReID model (no embeddings are
passed), so it is BoT-SORT's camera-motion-compensation variant.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.dataset import (
    find_dataset_root,
    list_sequences,
    load_dets_cache,
    read_frame,
    save_dets_cache,
    write_results,
)
from mot.detector import YoloDetector, clip_boxes, xyxy_to_tlwh
from mot.metrics import aggregate, evaluate, format_table

BOXMOT_TRACKERS = ("bytetrack", "ocsort", "botsort")


def load_boxmot():
    """Import the optional baseline package with an actionable error."""
    try:
        from boxmot import BotSort, ByteTrack, OcSort
    except ImportError as exc:
        raise SystemExit(
            "BoxMOT is optional and is not installed. Run `pip install boxmot` "
            "before using scripts/run_baselines.py."
        ) from exc
    return BotSort, ByteTrack, OcSort


def make_boxmot(name, seq):
    """Build a boxmot tracker with vendor defaults at the sequence's frame rate."""
    BotSort, ByteTrack, OcSort = load_boxmot()
    if name == "bytetrack":
        return ByteTrack(frame_rate=seq.fps)
    if name == "ocsort":
        return OcSort(frame_rate=seq.fps)
    if name == "botsort":
        return BotSort(frame_rate=seq.fps)
    raise ValueError(name)


def ensure_cache(root, seqs, cache_dir, detector):
    """Live-detect any sequence whose cache is missing or stale."""
    for seq in seqs:
        if load_dets_cache(cache_dir, seq.name, seq.length) is not None:
            print(f"  {seq.name}: cached", flush=True)
            continue
        print(f"  {seq.name}: detecting {seq.length} frames ...", flush=True)
        cache = {}
        fresh = []
        for frame_id in range(1, seq.length + 1):
            image = read_frame(seq, frame_id, cache)
            boxes_xyxy, scores, _ = detector.detect(image)
            boxes = clip_boxes(xyxy_to_tlwh(boxes_xyxy),
                               image.shape[1], image.shape[0])
            fresh.append((frame_id, boxes, scores))
            if frame_id % 500 == 0:
                print(f"    {frame_id}/{seq.length}", flush=True)
        save_dets_cache(cache_dir, seq.name, fresh, seq.length)


def tlwh_to_xyxy(frame_dets):
    arr = np.empty((frame_dets.shape[0], 6), dtype=np.float32)
    arr[:, 0] = frame_dets[:, 0]
    arr[:, 1] = frame_dets[:, 1]
    arr[:, 2] = frame_dets[:, 0] + frame_dets[:, 2]
    arr[:, 3] = frame_dets[:, 1] + frame_dets[:, 3]
    arr[:, 4] = frame_dets[:, 4]
    arr[:, 5] = 0.0
    return arr


def run_tracker(name, seq, dets):
    """Track one sequence from cached detections; returns rows + hypotheses."""
    tracker = make_boxmot(name, seq)
    cache = {}
    rows = []
    hypotheses = {}
    empty = np.zeros((0, 6), dtype=np.float32)
    t0 = time.perf_counter()
    for frame_id in range(1, seq.length + 1):
        frame_dets = dets.get(frame_id)
        arr = empty if frame_dets is None or frame_dets.shape[0] == 0 \
            else tlwh_to_xyxy(frame_dets)
        image = read_frame(seq, frame_id, cache)
        out = tracker.update(arr, image)
        if out is None or len(out) == 0:
            continue
        out = np.asarray(out)
        frame_rows = []
        for r in out:
            x1, y1, x2, y2 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            tid = int(r[4])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            frame_rows.append((tid, x1, y1, w, h, 1.0))
            rows.append((frame_id, tid, x1, y1, w, h))
        if frame_rows:
            hypotheses[frame_id] = np.asarray(frame_rows, dtype=np.float32)
    return rows, hypotheses, time.perf_counter() - t0


def vars_to_dict(m):
    """TrackMetrics -> plain dict (JSON-safe)."""
    out = {"sequence": m.sequence, "extra": dict(m.extra)}
    for k, v in vars(m).items():
        if k in ("sequence", "extra"):
            continue
        out[k] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--sequences", default="all")
    ap.add_argument("--dets-cache", default="results/dets_cache")
    ap.add_argument("--out", default="results/sota")
    ap.add_argument("--trackers", default=",".join(BOXMOT_TRACKERS))
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--device", default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
    for t in trackers:
        if t not in BOXMOT_TRACKERS:
            print(f"unknown tracker {t}; have {BOXMOT_TRACKERS}", file=sys.stderr)
            return 2
    load_boxmot()

    root = find_dataset_root(args.dataset_root)
    seqs = list_sequences(root, detector="FRCNN")
    if args.sequences != "all":
        wanted = {w.strip() for w in args.sequences.split(",") if w.strip()}
        seqs = [s for s in seqs
                if s.name in wanted or s.name.split("-")[1] in wanted]
    if args.max_frames:
        for s in seqs:
            s.length = min(s.length, args.max_frames)
    if not seqs:
        print("no sequences matched", file=sys.stderr)
        return 2

    print(f"dataset {root}; caching detections to {args.dets_cache}", flush=True)
    ensure_cache(root, seqs, args.dets_cache,
                 YoloDetector(weights=args.weights, device=args.device,
                              conf=args.conf, imgsz=args.imgsz))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in trackers:
        per_seq = []
        print(f"\n=== {name} ===", flush=True)
        for seq in seqs:
            dets = load_dets_cache(args.dets_cache, seq.name, seq.length)
            if dets is None:
                print(f"  {seq.name}: no usable cache; skipping", flush=True)
                continue
            rows, hyp, secs = run_tracker(name, seq, dets)
            fps = seq.length / secs if secs else 0.0
            metrics = evaluate(hyp, seq.gt_file, sequence=seq.name,
                               max_frame=seq.length)
            per_seq.append(metrics)
            print(f"  {seq.name}: {secs:.1f}s ({fps:.1f} fps)  "
                  f"MOTA {metrics.mota:6.2f}  HOTA {metrics.hota:6.2f}  "
                  f"IDF1 {metrics.idf1:.3f}", flush=True)
            write_results(out_root / name / f"{seq.name}.txt", rows)
        if not per_seq:
            continue
        overall = aggregate(per_seq)
        summary[name] = {
            "sequences": [vars_to_dict(m) for m in per_seq],
            "aggregate": vars_to_dict(overall),
        }
        print(format_table([overall], include_overall=False))

    # Include this repo's own trackers for the comparison table, when present.
    for label, path in (("sort_ours", "results/full_sort/sort/metrics.json"),
                        ("occlusion_ours", "results/full_occlusion/occlusion/metrics.json")):
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            summary[label] = {"sequences": d["sequences"],
                              "aggregate": d["aggregate"]}

    payload = {"config": vars(args), "trackers": summary}
    path = out_root / "metrics.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {path}")

    print("\n=== External tracker comparison (OVERALL) ===")
    print(f"{'tracker':16s} {'MOTA':>7s} {'MOTP':>7s} {'HOTA':>7s} {'IDF1':>6s} {'IDS':>6s}")
    for name, d in summary.items():
        a = d["aggregate"]
        print(f"{name:16s} {a['mota']:7.2f} {a['motp']:7.2f} {a['hota']:7.2f} "
              f"{a['idf1']:6.3f} {int(a['num_switches']):6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
