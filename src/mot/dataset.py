"""MOTChallenge dataset access: sequence enumeration, frames, ground truth."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: MOT17 ground-truth column layout (1-indexed in the raw files).
GT_COLUMNS = [
    "frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility",
]

#: MOT17 "consider" classes: pedestrian, person on vehicle, person on bike,
#: static person. Class 8 (distractor) and class 9 (ignored region) exist only to
#: suppress false positives, so they must never appear in our target set.
MOT17_EVAL_CLASSES = (1, 2, 4, 7)


@dataclass
class SequenceInfo:
    name: str
    path: Path
    length: int
    width: int
    height: int
    fps: float

    @property
    def img_dir(self):
        return self.path / "img1"

    @property
    def gt_file(self):
        return self.path / "gt" / "gt.txt"

    @property
    def det_file(self):
        return self.path / "det" / "det.txt"


def find_dataset_root(start=None, name="MOT17"):
    """Locate an extracted MOT dataset directory."""
    candidates = []
    if start:
        candidates.append(Path(start))
    here = Path(__file__).resolve().parents[2]
    candidates += [
        here / "data" / name,
        here.parent / "data" / name,
        Path.cwd() / "data" / name,
        Path.cwd() / name,
        Path("/data") / name,
        Path.home() / "datasets" / name,
    ]
    env = os.environ.get("MOT_DATA_ROOT")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if (c / "train").is_dir():
            return c
    raise FileNotFoundError(
        f"could not find {name}; set MOT_DATA_ROOT or pass --data-root"
    )


def list_sequences(root, split="train", detector=None, min_length=1):
    """All sequences under ``root/split``.

    MOT17 stores three variants of every train sequence (DPM/FRCNN/SDP) which
    share the same ground truth. Passing ``detector='FRCNN'`` returns the 7
    canonical sequences instead of all 21 variants.
    """
    out = []
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if detector and not name.endswith(detector):
            continue
        meta = read_seqinfo(d)
        if meta["length"] < min_length:
            continue
        out.append(SequenceInfo(name=name, path=d, **meta))
    return out


def read_seqinfo(seq_dir):
    """Parse ``seqinfo.ini``.

    MOT17 calls the section ``[Sequence]`` with keys ``seqLength``, ``seqWidth``,
    ``seqHeight``, ``seqFPS``; MOT20 uses the same file but a section named
    ``[sequence]``. Both spellings are accepted, along with the plural keys used
    by a few other MOT releases, so the loader works on either dataset.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str  # keep key case as written
    path = seq_dir / "seqinfo.ini"
    if not path.exists():
        raise FileNotFoundError(path)
    parser.read(path)
    if not parser.sections():
        raise ValueError(f"no section found in {path}")
    sec = dict(parser[parser.sections()[0]])

    def pick(*keys):
        for k in keys:
            if k in sec:
                return sec[k]
        raise KeyError(f"none of {keys} present in {path}")

    length = int(pick("seqLength", "sequence_length", "numFrames"))
    width = int(pick("seqWidth", "sequence_width", "imWidth"))
    height = int(pick("seqHeight", "sequence_height", "imHeight"))
    fps = float(pick("seqFPS", "sequence_frame_rate", "frameRate"))
    return {"length": length, "width": width, "height": height, "fps": fps}


def frame_files(seq):
    files = sorted(os.listdir(seq.img_dir))
    return [seq.img_dir / f for f in files]


def read_frame(seq, frame_id, cache):
    """Read a frame as RGB ``uint8``. Results are cached per sequence."""
    if seq.name not in cache:
        cache.clear()
        cache[seq.name] = frame_files(seq)
    files = cache[seq.name]
    path = files[frame_id - 1]
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_boxes(path, distractor_classes=(0,), keep_classes=None,
               min_confidence=0.0):
    """Parse a MOT-format txt (ground truth or detections).

    Returns a dict ``frame_id -> (N, 6) array`` of
    ``[track_id, x, y, w, h, score]``.
    """
    if not os.path.exists(path):
        return {}
    frames = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = [float(v) for v in line.replace(" ", "").split(",")]
            frame_id = int(p[0])
            if distractor_classes and len(p) >= 8 and int(p[7]) in distractor_classes:
                continue
            if keep_classes is not None and len(p) >= 8:
                if int(p[7]) not in keep_classes:
                    continue
            score = p[6] if len(p) > 6 else 1.0
            if score < min_confidence:
                continue
            frames.setdefault(frame_id, []).append(
                [int(p[1]), p[2], p[3], p[4], p[5], score]
            )
    return {k: np.array(v, dtype=np.float32) for k, v in frames.items()}


def read_gt(path):
    """Ground truth for evaluation: MOT17 "consider" classes only.

    MOTChallenge scores detections against class 1 (pedestrian), 2 (person on
    vehicle), 4 (person on bike) and 7 (static person). Class 8 (distractor) and
    class 9 (ignored region) are not targets — see :func:`read_ignore_regions`.

    Note on the ``conf`` column: in the released MOT17 ground truth it is 1 only
    for class 1. Filtering on ``conf == 1`` therefore silently drops crowds,
    riders and static people, so the class list is the correct filter.
    """
    return read_boxes(path, distractor_classes=(), keep_classes=MOT17_EVAL_CLASSES)


def read_ignore_regions(path):
    """Boxes that must not generate false positives.

    The MOTChallenge devkit suppresses false positives inside annotated
    distractor / ignored regions. ``motmetrics`` has no notion of this, so we
    parse them separately and mark detections inside them as unmatchable in
    :mod:`mot.metrics`.
    """
    regions = {}
    if not os.path.exists(path):
        return regions
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = [float(v) for v in line.replace(" ", "").split(",")]
            cls = int(p[7]) if len(p) > 7 else 0
            if cls in MOT17_EVAL_CLASSES:
                continue
            regions.setdefault(int(p[0]), []).append(
                [p[1], p[2], p[3], p[4], p[5], 0.0, p[8] if len(p) > 8 else 1.0]
            )
    return {k: np.array(v, dtype=np.float32) for k, v in regions.items()}


def read_gt_with_visibility(path, min_visibility=0.0):
    """GT plus the visibility ratio, which the failure analysis uses.

    Visibility is the fraction of the annotated box that is actually visible, so
    a low value is a direct occlusion signal. Rows are kept regardless of the
    ``conf`` column (see :func:`read_gt`), and ``min_visibility`` optionally drops
    the heavily occluded boxes that MOTChallenge's own analysis uses to isolate
    occlusion failures.
    """
    frames = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = [float(v) for v in line.replace(" ", "").split(",")]
            frame_id = int(p[0])
            cls = int(p[7]) if len(p) > 7 else 1
            if cls not in MOT17_EVAL_CLASSES:
                continue
            visibility = p[8] if len(p) > 8 else 1.0
            if visibility < min_visibility:
                continue
            frames.setdefault(frame_id, []).append(
                [int(p[1]), p[2], p[3], p[4], p[5], 1.0, visibility]
            )
    return {k: np.array(v, dtype=np.float32) for k, v in frames.items()}


def gt_visibility(boxes):
    """Per-box visibility column from :func:`read_gt_with_visibility` output."""
    if boxes is None or boxes.shape[1] < 7:
        return np.ones(0, dtype=np.float32)
    return boxes[:, 6]


def write_results(path, rows, conf=1.0):
    """Write MOTChallenge-format results (frame,id,x,y,w,h,conf,-1,-1).

    ``rows`` holds the six MOT columns ``(frame, id, x, y, w, h)``. Confidence is
    uniform because MOTChallenge scores a tracking output, not a ranked detector,
    so it is written once here instead of being threaded through every caller.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    with open(path, "w") as fh:
        for frame, tid, x, y, w, h in rows:
            fh.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},"
                     f"{w:.2f},{h:.2f},{conf:.4f},-1,-1\n")