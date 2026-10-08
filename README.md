# Multi-Object Tracking Under Occlusion

This project compares a classic SORT baseline with a custom
**appearance-assisted occlusion-aware tracker** on the MOT17 FRCNN training
sequences. It uses a fixed, COCO-pretrained YOLO11 detector—no detector is
trained from scratch—and asks whether retaining lost tracks and matching them
with appearance information preserves identity through brief occlusions.

> This is not DeepSORT. The occlusion-aware tracker uses a generic
> ImageNet-pretrained MobileNetV3-small descriptor plus HSV colour features;
> it is not a person-ReID model trained on a ReID benchmark.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the 5.5 GB MOT17 archive and extract it to data/MOT17/.
mkdir -p data
curl -L -o /tmp/MOT17.zip https://motchallenge.net/data/MOT17.zip
unzip -q /tmp/MOT17.zip -d data

# Download the fixed pretrained detector used by the recorded benchmark.
curl -L -o yolo11x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt

# Run one sequence with the two trackers.
PYTHONPATH=src python -m mot.pipeline --dataset-root data/MOT17 \
  --sequences 04 --tracker sort --out-dir results --tag sort
PYTHONPATH=src python -m mot.pipeline --dataset-root data/MOT17 \
  --sequences 04 --tracker occlusion --out-dir results --tag occlusion

# Render the occlusion-aware tracks as an annotated MP4.
PYTHONPATH=src python scripts/visualize.py --dataset-root data/MOT17 \
  --sequence 04 --tracker occlusion --results-dir results \
  --output results/visualizations/MOT17-04-occlusion.mp4
```

The official [MOT17 archive](https://motchallenge.net/data/MOT17/) extracts
the expected `data/MOT17/train/MOT17-XX-FRCNN/{img1,gt}` layout. Data, model
weights, videos, caches, logs, and results are intentionally ignored by Git.

## Architecture

| System | Association and lifecycle |
| --- | --- |
| **SORT baseline** | Constant-velocity Kalman filter, IoU-cost Hungarian assignment, tentative/confirmed/lost lifecycle. |
| **Occlusion-aware tracker** | Kalman prediction, age-ordered cascade association, generic appearance and HSV descriptor, lost-track gallery, Mahalanobis gating, and re-identification recovery. |

The detector wrapper filters the COCO `person` class. The evaluator calculates
MOTA, MOTP, IDF1, ID switches, precision, recall, and a self-contained HOTA
implementation. MOTA, precision, recall, MOTP, and IDF1 are pooled from event
counts; HOTA is a per-sequence mean and is **not** TrackEval's combined HOTA.

## Installation and Hardware

`requirements.txt` pins the CUDA 13.0 PyTorch build used for the recorded
experiments. Install it with the commands above; the PyTorch wheel index is
already declared in that file. The recorded configuration is `yolo11x` at
`imgsz=1600`, which is demanding for an 8 GB laptop GPU. If it exhausts memory,
use a smaller checkpoint or image size for exploratory runs, but do not compare
those results directly with the recorded benchmark.

The detector weights are downloaded separately and are not committed. On first
appearance-encoder use, torchvision may also download its ImageNet pretrained
MobileNetV3-small weights.

## Running the Trackers

Run the complete seven-sequence FRCNN benchmark with the recorded configuration:

```bash
./scripts/run_full_benchmark.sh
```

It writes per-sequence MOT text files and `metrics.json` under
`results/full_sort/sort/` and `results/full_occlusion/occlusion/`. For smaller
or reproducible iterative runs, cache detector output so both trackers consume
the same detections:

```bash
PYTHONPATH=src python -m mot.pipeline --dataset-root data/MOT17 \
  --sequences 04 --tracker sort --dets-cache results/dets_cache \
  --out-dir results --tag sort
PYTHONPATH=src python -m mot.pipeline --dataset-root data/MOT17 \
  --sequences 04 --tracker occlusion --dets-cache results/dets_cache \
  --out-dir results --tag occlusion
```

Useful options include `--max-frames N` for smoke runs, `--no-appearance`,
`--no-reid`, and `--no-cascade` for ablations. Run `PYTHONPATH=src python -m
mot.pipeline --help` for the complete CLI.

## Evaluation

Evaluate existing MOT-format outputs without a GPU or detector rerun:

```bash
PYTHONPATH=src python scripts/reevaluate.py \
  --dataset-root data/MOT17 --results results/occlusion --tracker occlusion --write
```

Generate Markdown comparison tables from two `metrics.json` files:

```bash
PYTHONPATH=src python scripts/make_report.py \
  --sort results/full_sort/sort/metrics.json \
  --occlusion results/full_occlusion/occlusion/metrics.json
```

`metrics.json` emitted during a pipeline run is the authoritative record.
Re-evaluation from `.txt` output can differ near the IoU boundary because boxes
are serialized to two decimal places.

## Trajectory Visualisation

`scripts/visualize.py` reads frames plus an existing tracker-output file. It
draws a stable colour per ID, bounding box, ID label, and a configurable recent
centre-point trail. It does not rerun detection or tracking.

```bash
PYTHONPATH=src python scripts/visualize.py \
  --sequence-dir data/MOT17/train/MOT17-04-FRCNN \
  --results results/full_occlusion/occlusion/MOT17-04-FRCNN.txt \
  --history 45 --start-frame 1 --end-frame 300 \
  --output results/visualizations/MOT17-04-occlusion.mp4
```

The verified local run produced `MOT17-04-occlusion.mp4` with 1,050 annotated
frames. Benchmark assets and generated videos are intentionally not versioned;
the command above is the reproducible generation path.

## Results and Failure Analysis

The retained seven-sequence run uses `yolo11x.pt` and MOT17 FRCNN training
sequences 02, 04, 05, 09, 10, 11, and 13. The authoritative local artifacts are
`results/full_sort/sort/metrics.json` and
`results/full_occlusion/occlusion/metrics.json`; they are ignored by Git so that
the repository does not contain datasets, weights, or generated outputs.

| Tracker | MOTA | HOTA | IDF1 | ID Switches | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SORT | 34.87 | 21.51 | 0.267 | 3017 | 40.97 | 88.48 |
| Occlusion-aware | 39.66 | 32.01 | 0.379 | 2204 | 49.11 | 84.79 |

The occlusion-aware tracker improves MOTA by **4.79** points, HOTA by **10.50**
points, and IDF1 by **0.112**, while reducing ID switches by **813** (about
**26.95%**). Recall rises by **8.14** points, while precision falls by **3.69**
points. Full results also report MOTP (82.43 / 81.09) and fragmentation (3635 /
3315) for SORT / occlusion-aware. MOTA, precision, recall, MOTP, and IDF1 are
pooled from event counts; HOTA is an unweighted sequence mean, not TrackEval's
combined HOTA.

Observed results and supported failure analysis:

- **Temporary occlusion:** lost-track recovery can retain an ID, but long gaps
  remain ambiguous.
- **Fast motion:** a constant-velocity filter and gate can fail after abrupt
  camera or pedestrian motion.
- **Similar-looking pedestrians:** generic embeddings are not trained for
  identity discrimination.
- **Detector misses:** low detector recall caps every tracking metric.
- **False detections and stale tracks:** retained tracks can trade recall and
  ID continuity for lower precision.
- **ID switches:** the recorded run measures 3,017 SORT switches and 2,204
  occlusion-aware switches; individual causes require sequence-level review.

The evaluation protocol treats MOT17 classes 1/2/4/7 as targets and suppresses
hypotheses in ignored regions. It therefore is not directly comparable to the
official MOTChallenge protocol, which uses a different class convention.

## Optional Baselines

`scripts/run_baselines.py` can run ByteTrack, OC-SORT, and BoT-SORT on cached
detections, but it is optional and needs an additional package:

```bash
pip install boxmot
PYTHONPATH=src python scripts/run_baselines.py --dataset-root data/MOT17 \
  --dets-cache results/dets_cache --out results/sota
```

Do not call a BoxMOT comparison reproducible until its result JSON and exact
package version have been retained alongside the experiment record.

## Tests

```bash
PYTHONPATH=src python tests/run_all.py
```

The suite is dependency-light and covers tracker lifecycle and recovery,
Kalman stability, Hungarian association, detector batch/OOM handling, metric
regression cases, aggregate semantics, detector-ceiling matching, and
visualisation-path parsing.

## Project Structure

```text
src/mot/       detector, data loader, Kalman filter, trackers, metrics, pipeline
scripts/       benchmarks, evaluation, detector sweep, report, video renderer
tests/         standalone regression suite
WRITEUP.md     detailed technical notes and previously recorded experiment tables
REPORT.md      concise AWS submission report
```

## Limitations

- The appearance descriptor is not a trained person-ReID model.
- Detector recall is a principal bottleneck; association cannot recover boxes
  that were never detected.
- Dataset, detector weights, generated results, logs, and video remain ignored
  and must be retained locally to reproduce the recorded benchmark.
- Full benchmark runtime and VRAM usage depend on the GPU, driver, and model
  build.

## License

MIT. See `LICENSE`; MOT17 has its own terms. The HOTA implementation is a
documented transcription of the MIT-licensed TrackEval formulation.
