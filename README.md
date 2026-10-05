# MOT17 under occlusion: SORT vs. an appearance-assisted re-identification tracker

A from-scratch multi-object tracker evaluated on the MOT17 FRCNN sequences, built to
answer one question honestly:

> When a tracked pedestrian is occluded for a few seconds and re-emerges, can a tracker
> recover the **same identity** — and how much does that cost in false positives?

The project runs a **fixed, off-the-shelf detector** (`yolo11x`) over MOT17 and swaps the
tracker underneath, so every difference in the results table is attributable to
association logic rather than to perception quality. No detector is trained here.

## Results

See [`WRITEUP.md`](WRITEUP.md) for the full benchmark tables, the failure analysis, and
the negative results. Headline numbers are reproduced there.

## What is in here

| Path | What it does |
| --- | --- |
| `src/mot/detector.py` | `yolo11x` wrapper: batching, fp16, CUDA-OOM retry |
| `src/mot/dataset.py` | MOT17 reader, class filtering, ignore regions |
| `src/mot/appearance.py` | 1088-d appearance descriptor (MobileNetV3-small, HSV) |
| `src/mot/kalman.py` | Kalman filter/covariance update with a robust Cholesky fallback |
| `src/mot/track.py` | Track state machine (`Tentative`/`Confirmed`/`Lost`) |
| `src/mot/sort.py` | SORT baseline: IoU + Kalman only |
| `src/mot/occlusion.py` | Occlusion tracker: cascade association + re-ID |
| `src/mot/metrics.py` | MOT metrics + HOTA (self-contained; see *Metrics* below) |
| `src/mot/pipeline.py` | CLI that ties detector + tracker + evaluator together |
| `tests/` | 51 tests across five suites, no pytest required |
| `scripts/run_full_benchmark.sh` | Runs both trackers over all seven sequences |
| `scripts/detector_ceiling.py` | Measures detection ceiling (no tracker) for the write-up |
| `scripts/reevaluate.py` | Recomputes metrics from written `.txt` files, no GPU |
| `scripts/make_report.py` | Renders the write-up's tables from `metrics.json` |

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch, so it installs from
`download.pytorch.org` in addition to PyPI. `pip` handles that
automatically; with `uv` you must pass `--index-strategy unsafe-best-match`,
otherwise the resolver treats the PyTorch index as authoritative for unrelated
packages (it ships its own NumPy) and fails:

```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

Detector weights are **not** committed. Fetch the one the benchmark uses before
running:

```bash
# yolo11x.pt must sit in the repository root
curl -L -o yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

`scripts/detector_ceiling.py --configs size-sweep` additionally uses `yolo11n`,
`yolo11m` and `yolo11l`; it prints `SKIPPED` for any checkpoint you have not
fetched.

MOT17 must be extracted so that `data/MOT17/train/MOT17-XX-FRCNN/` each contain
`img1/` and `gt/gt.txt`. The reader also accepts a directory passed via
`--dataset-root`.

## Running

Single sequence, either tracker:

```bash
PYTHONPATH=src ./venv/bin/python -m mot.pipeline --sequences 02 --tracker occlusion
```

Both trackers over all seven FRCNN sequences, exactly as benchmarked:

```bash
./scripts/run_full_benchmark.sh
```

Useful flags (defaults in brackets):

- `--weights [yolo11x.pt]` — any Ultralytics checkpoint
- `--conf [0.25]`, `--imgsz [1600]` — detector inference settings
- `--det-conf-threshold [0.3]` — only used to *start* new tracks; lower-score
  detections still sustain existing ones
- `--min-hits [3]`, `--max-age [30]` — track lifecycle
- `--appearance-weight [0.6]`, `--gallery-size [30]`, `--reid-max-age [45]` — re-ID
- `--no-appearance` — ablation: run the occlusion tracker with the appearance cue off
- `--max-frames N` — debug: stop after N frames per sequence

Output is MOT-format `results/<tag>/MOT17-XX-FRCNN.txt` plus a `metrics.json` summary.

`metrics.json` is authoritative. Its `aggregate` row pools the MOTChallenge rates
from summed event counts; recomputing from the `.txt` files instead
(`scripts/reevaluate.py`) shifts a few borderline IoU matches, because results
are written at `%.2f` precision.

## Tests

```bash
PYTHONPATH=src ./venv/bin/python tests/run_all.py
```

51 tests, five suites (`tracker`, `detector`, `metrics`, `metrics aggregate`,
`detector_ceiling`). The suite runner is hand-rolled (`tests/_runner.py`) because
pytest is not available in this environment.

The tests are written as **regression tests with hand-computed or captured
expectations**, not smoke tests. Where a fix could be "verified", each test was
mutation-checked: reintroducing the original bug must turn it red. Notable cases:

- the MOT17-02 frame-119 Kalman covariance crash fixture
- `min_hits` promotion and empty-frame track aging
- detector OOM retry must not drop or duplicate frames
- HOTA must fall when identities are scrambled even if every box is perfect
- OVERALL MOTA/recall/precision must be pooled from summed event counts, not
  averaged across sequences, and `num_matches` must be summed at all

## Metrics

`src/mot/metrics.py` wraps `motmetrics` 1.4.0 for MOTA / MOTP / IDF1 and the event
counts, with two deliberate deviations:

- **The distance matrix is computed locally.** `motmetrics.distances.iou_matrix`
  calls `np.asfarray`, removed in NumPy 2.0, and this project runs NumPy 2 with a
  CUDA build of PyTorch. `distance_matrix()` reimplements it with the exact
  semantics `MOTAccumulator.update` documents (`NaN` = forbidden pair), so the
  accumulator still does the matching and scoring.
- **MOTP is converted to a percentage.** `motmetrics` reports the *mean matching
  distance* (lower is better); MOTChallenge and every leaderboard publish
  `1 - distance`. Without the conversion the column would be mislabelled.
- **Ignore regions are handled explicitly.** `motmetrics` has no notion of MOT17's
  annotated distractor / ignored boxes, so detections inside them are filtered
  before scoring.

HOTA is implemented from the paper in the same module so its definition is auditable
in one place. The association term carries an `H` penalty for identity switches —
dropping it is a silent bug that makes HOTA degenerate into a detection score which
returns 1.0 even when every frame has a different track id. `tests/test_metrics.py`
pins this.

This is the paper's single-class form; TrackEval aggregates the association term
differently (a log-Jaccard over global id pairs), so absolute HOTA differs from
published TrackEval numbers while preserving ranking order.

## Notes and limitations

- **Appearance is not person-ReID.** The descriptor is a generic ImageNet
  MobileNetV3-small embedding in HSV, L2-normalised. It separates pedestrians well enough
  to be useful but is not trained to distinguish *individuals*; a trained ReID checkpoint
  was not obtainable in this environment. This is the single biggest available
  improvement.
- Detection quality dominates the absolute numbers. See the detector sweep in the
  write-up; recall is the binding constraint.
- `bus.jpg` in the repository root is an Ultralytics demo asset, not project input.

## License

See repository metadata. MOT17 is distributed by its own authors under its own terms.