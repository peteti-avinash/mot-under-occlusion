"""Render MOT-format tracker output over an MOTChallenge image sequence.

The renderer is deliberately independent of detector and tracker inference: it
uses the ``MOT17-XX-FRCNN.txt`` file already produced by ``mot.pipeline``. This
makes a visualisation reproducible without rerunning a GPU job.

Example:
    PYTHONPATH=src python scripts/visualize.py \
        --dataset-root data/MOT17 --sequence 04 --tracker occlusion \
        --results-dir results \
        --output results/visualizations/MOT17-04-occlusion.mp4
"""

from __future__ import annotations

import argparse
import configparser
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mot.dataset import find_dataset_root


def canonical_sequence_name(sequence, detector="FRCNN"):
    """Return an MOT17 directory name for a short or full sequence selector."""
    sequence = str(sequence).strip()
    if sequence.startswith("MOT17-"):
        return sequence
    try:
        sequence = f"{int(sequence):02d}"
    except ValueError:
        pass
    return f"MOT17-{sequence}-{detector}"


def resolve_sequence_dir(dataset_root, sequence, detector="FRCNN", sequence_dir=None):
    """Find a sequence directory with an ``img1`` folder."""
    if sequence_dir is not None:
        path = Path(sequence_dir)
        if not (path / "img1").is_dir():
            raise FileNotFoundError(f"missing img1 directory in {path}")
        return path

    root = Path(dataset_root)
    name = canonical_sequence_name(sequence, detector)
    for path in (root / "train" / name, root / name):
        if (path / "img1").is_dir():
            return path
    raise FileNotFoundError(
        f"could not find {name}/img1 under {root}; pass --sequence-dir to use "
        "a sequence directory directly"
    )


def resolve_results_file(results, results_dir, tracker, sequence_name):
    """Resolve an explicit results file or the pipeline's standard location."""
    if results:
        path = Path(results)
    else:
        path = Path(results_dir) / tracker / f"{sequence_name}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing tracker output {path}; run mot.pipeline first or pass --results"
        )
    return path


def read_tracks(path):
    """Read MOT rows into ``frame_id -> [(id, x, y, w, h)]``."""
    tracks = defaultdict(list)
    with open(path) as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            values = line.replace(" ", "").split(",")
            if len(values) < 6:
                raise ValueError(f"{path}:{lineno}: expected at least 6 MOT columns")
            try:
                frame_id = int(float(values[0]))
                track_id = int(float(values[1]))
                x, y, width, height = (float(value) for value in values[2:6])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-numeric MOT row") from exc
            if width > 0 and height > 0:
                tracks[frame_id].append((track_id, x, y, width, height))
    return tracks


def sequence_fps(sequence_dir, fallback=30.0):
    """Read frame rate from MOTChallenge metadata, with a portable fallback."""
    path = Path(sequence_dir) / "seqinfo.ini"
    if not path.is_file():
        return fallback
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.sections():
        return fallback
    section = parser[parser.sections()[0]]
    for key in ("seqFPS", "sequence_frame_rate", "frameRate"):
        if key in section:
            return float(section[key])
    return fallback


def track_color(track_id):
    """Stable, high-contrast BGR colour for a positive integer track ID."""
    hue = (int(track_id) * 47) % 180
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def image_files(sequence_dir):
    """Sorted image frames from an MOTChallenge sequence."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(path for path in (Path(sequence_dir) / "img1").iterdir()
                   if path.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"no image frames found in {Path(sequence_dir) / 'img1'}")
    return files


def draw_frame(image, rows, trails, history):
    """Draw boxes, IDs, and recent centre-point trajectories onto one frame."""
    for track_id, x, y, width, height in rows:
        color = track_color(track_id)
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + width)), int(round(y + height))
        centre = (int(round(x + width / 2)), int(round(y + height / 2)))
        trails[track_id].append(centre)

        points = list(trails[track_id])
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {track_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, y1 - label_height - baseline - 4)
        cv2.rectangle(image, (x1, label_top),
                      (x1 + label_width + 6, label_top + label_height + baseline + 4),
                      color, thickness=-1)
        cv2.putText(image, label, (x1 + 3, label_top + label_height + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def render(sequence_dir, tracks, output, history=30, start_frame=1, end_frame=None,
           fps=None):
    """Write annotated frames to ``output`` and return the frame count written."""
    if history < 1:
        raise ValueError("history must be at least 1")
    files = image_files(sequence_dir)
    start_frame = max(1, int(start_frame))
    final_frame = len(files) if end_frame is None else min(len(files), int(end_frame))
    if final_frame < start_frame:
        raise ValueError("end frame precedes start frame")

    first = cv2.imread(str(files[start_frame - 1]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(files[start_frame - 1])
    height, width = first.shape[:2]
    rate = float(fps) if fps is not None else sequence_fps(sequence_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    trails = defaultdict(lambda: deque(maxlen=history))
    written = 0
    try:
        for frame_id in range(start_frame, final_frame + 1):
            image = cv2.imread(str(files[frame_id - 1]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(files[frame_id - 1])
            image = draw_frame(image, tracks.get(frame_id, ()), trails, history)
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", help="MOT17 root containing train/")
    source.add_argument("--sequence-dir", help="direct path to MOT17-XX-FRCNN")
    parser.add_argument("--sequence", default=None,
                        help="short ID (for example 04) or full MOT17 sequence name")
    parser.add_argument("--detector", default="FRCNN", help="MOT17 detector suffix")
    parser.add_argument("--tracker", default="occlusion",
                        help="pipeline output subdirectory when --results is omitted")
    parser.add_argument("--results-dir", default="results",
                        help="parent directory containing <tracker>/MOT17-XX-*.txt")
    parser.add_argument("--results", help="explicit MOT-format tracker-output file")
    parser.add_argument("--output", required=True, help="output .mp4 file")
    parser.add_argument("--history", type=int, default=30,
                        help="number of recent centre points to draw per ID")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None,
                        help="override seqinfo.ini's frame rate")
    args = parser.parse_args(argv)

    if args.sequence_dir:
        sequence_dir = resolve_sequence_dir(None, args.sequence or "", args.detector,
                                            args.sequence_dir)
        sequence_name = sequence_dir.name
    else:
        if not args.sequence:
            parser.error("--sequence is required unless --sequence-dir is provided")
        dataset_root = args.dataset_root or find_dataset_root()
        sequence_dir = resolve_sequence_dir(dataset_root, args.sequence, args.detector)
        sequence_name = sequence_dir.name
    results = resolve_results_file(args.results, args.results_dir, args.tracker,
                                   sequence_name)
    count = render(sequence_dir, read_tracks(results), args.output, args.history,
                   args.start_frame, args.end_frame, args.fps)
    print(f"wrote {count} annotated frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
