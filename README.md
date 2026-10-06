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
| `tests/` | 60 tests across five suites, no pytest required |
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
from summed event counts, weights MOTP by matches, and recomputes IDF1 from
pooled ID counts; only HOTA and its components are unweighted means over
sequences (see *Metrics* below). Recomputing
from the `.txt` files instead (`scripts/reevaluate.py`) reproduces those numbers
to within 0.002 HOTA / 0.015 MOTA per sequence — the residual is `%.2f`
quantisation of box coordinates shifting borderline IoU matches, not a
disagreement. It still cannot see detections that `min_hits` suppressed before
they were written, so it is a debugging tool rather than a substitute.

## Tests

```bash
PYTHONPATH=src ./venv/bin/python tests/run_all.py
```

60 tests, five suites (`tracker`, `detector`, `metrics`, `metrics aggregate`,
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
  before scoring. The regions are applied to *hypotheses only*: a ground-truth box
  whose centre falls inside an ignore region still counts as a missed target. That
  is the convention implemented here, and it is the only thing the phrase "ignore
  regions" leaves open, so it is worth quantifying. Dropping such ground-truth
  boxes as well was measured over the seven SORT sequences: HOTA moves by **+0.91 on
  average**, ranging from −0.14 (MOT17-09) to +2.16 (MOT17-04), and MOTA by **−0.83
  on average**, reaching −6.36 on MOT17-09. A +2.16 HOTA shift is ~10% of this
  project's 21.5 baseline, so neither metric should be compared across the two
  conventions without a note.

HOTA is implemented from the paper in the same module so its definition is auditable
in one place. Association is a sequence-wide Jaccard over global
`(gt id, tracker id)` pairs, so dropping the association term is a silent bug that
makes HOTA degenerate into a detection score and return 1.0 even when every frame
has a different track id. `tests/test_metrics.py` pins this.

This is the TrackEval *formulation*, but the per-sequence values here are **not**
directly comparable to published MOTChallenge HOTA numbers. This project evaluates
MOT17 classes 1/2/4/7 as targets; the official protocol scores class 1 only and
suppresses hypotheses matched to distractor classes. Measured on identical
hypothesis files, restricting ground truth to class 1 alone shifts per-sequence
HOTA by +0.08 to +2.38 (mean +0.91 over the seven sequences). Compare against
other work only after normalising the class convention. The per-frame Hungarian assignment is
driven by the global alignment score rather than raw IoU, which is what stops
identity flipping between nearby people.

Two deliberate differences from the TrackEval tool are worth stating plainly. The
MOTChallenge rates in the `OVERALL` row — MOTA, precision, recall, MOTP, IDF1 — are
pooled the way TrackEval's `COMBINED_SEQ` pools them, but **HOTA is not**: TrackEval
pools per-alpha TP counts and association sums across sequences and recomputes the
score, whereas `OVERALL` takes the unweighted mean of the per-sequence HOTA values.
Both are legitimate, but they are different quantities, so the HOTA `OVERALL` must not
be quoted as TrackEval's `COMBINED_SEQ` HOTA. On this dataset the distinction is not
cosmetic: pooled MOTA is 34.87 against a sequence mean of 32.07, because pooling
weights MOT17-02's 26,960 evaluated objects against much longer sequences
differently than averaging seven scores does.

And TrackEval's per-frame assignment uses buffered
`+=` indexing, which keeps only the last write for a duplicated index; this
implementation uses `np.add.at`, which sums every row. The two agree exactly on
valid MOT data, where an id appears at most once per frame, and differ only on
input that is not valid MOT data.

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

MIT — see [LICENSE](LICENSE), which also carries the TrackEval notice for the
transcribed HOTA metric in `src/mot/metrics.py`.

The HOTA metric in `src/mot/metrics.py` was transcribed from TrackEval
(https://github.com/JonathonLuiten/TrackEval, © Jonathon Luiten). TrackEval is
MIT-licensed and carries its own copyright notice; the attribution above was
written from the upstream project without re-fetching its license text, since
this environment has no outbound network access — verify against upstream before
redistribution if that matters for your compliance process.

MOT17 is distributed by its own authors under its own terms.