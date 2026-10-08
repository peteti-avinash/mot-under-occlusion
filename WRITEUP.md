# Recovering identity through occlusion on MOT17

**Question.** When a pedestrian is occluded for a few seconds and re-emerges, can a
tracker recover the *same identity* — and what does that cost in false positives?

**Design.** One pretrained detector, held fixed; two trackers, swapped underneath. That
way every difference in the results is attributable to association logic rather than to
perception quality. No detector was trained.

- **Detector** — `yolo11x`, `imgsz=1600`, `conf=0.25`, batched with CUDA-OOM retry.
- **SORT** — IoU + Kalman only. The baseline.
- **Occlusion tracker** — SORT plus cascade association, a Lost state that survives
  `max_age=30` coasting frames, and a re-ID step gated on a 1088-d appearance
  descriptor.

Dataset: the seven MOT17 **FRCNN** train sequences (02, 04, 05, 09, 10, 11, 13),
**5,316 frames**. Classes 1/2/4/7 are evaluated; annotated ignore regions are
suppressed. Metrics are 0–100 for MOTA/MOTP/HOTA and 0–1 for IDF1.

**Result.** Holding the detector fixed, adding occlusion-aware association moves
**HOTA 21.51 → 32.01 (+10.51)** and **IDF1 0.267 → 0.379**, with **−27% identity
switches** (−813 over 5,316 frames). It costs **−3.69 precision** (88.48 → 84.79)
and **−1.34 MOTP**, most plausibly because surviving tracks emit boxes for
pedestrians that are not annotated. The trade is "remember more, be wrong more
often" — the right direction for re-identification and the wrong one for
counting. Scope: one detector configuration, one run per tracker, no
hyperparameter sweep, no per-component ablation.

---

## 1. The bugs that actually mattered

The headline finding of this project is not the leaderboard row; it is that **the
plausible-looking numbers were wrong, and in ways that only showed up under mutation
testing.**

### 1.1 HOTA was silently a detection score

The first HOTA implementation returned:

| hypothesis boxes | HOTA | DetA | AssA |
| --- | --- | --- | --- |
| perfect, identity kept | 1.000 | 1.000 | 1.000 |
| perfect boxes, **new track id every frame** | **1.000** | 1.000 | **1.000** |

A tracker that assigns a brand-new identity to every pedestrian in every frame scored a
*perfect* HOTA. The association term was never charged for identity switches, so HOTA
collapsed into DetA. Any tuning done against that number would have been tuning against
pure detection recall.

Fixed by scoring association as a sequence-wide Jaccard over `(gt id, tracker id)`
pairs (`metrics.py:147-291`), which is the TrackEval formulation. Same inputs, now:

| hypothesis boxes | HOTA | DetA | AssA |
| --- | --- | --- | --- |
| perfect, identity kept | 1.000 | 1.000 | 1.000 |
| perfect, 1 switch in 10 frames | 0.707 | 1.000 | 0.500 |
| perfect, **new id every frame** | **0.316** | 1.000 | 0.100 |
| 2 GT ids, only 1 detected | 0.707 | 0.500 | 1.000 |

Note the last two rows: HOTA is `sqrt(DetA * AssA)`, so perfect association with
half the boxes detected still scores 0.707, while re-issuing every identity scores
0.316. The original implementation could not distinguish those cases at all.
`tests/test_metrics.py` pins each of these four rows by hand derivation, including
the per-pair Jaccard arithmetic.

**This bug invalidated every earlier benchmark number in this project.** They are
recomputed in §3 from the corrected implementation.

### 1.2 Track age was incremented twice per frame

`Track.predict()` and `Track.update()` each did `self.age += 1`. A track that was
observed this frame therefore aged twice, while a track that was coasting through an
occlusion aged once per frame. `OcclusionAwareTracker._cascade_match()` uses age to
prefer the older track when two candidates are otherwise comparable, so **matched
tracks looked older than coasting tracks** — the exact opposite of the intended
"prefer the track that has been surviving longest" rule.

Fixed by incrementing only in `predict()`. Two tests now pin the semantics, and both go
red if the `update()` increment is restored:

- a track observed for 4 frames then coasting 2 has age 6, not 8;
- with two tracks, the one that has coasted longer wins the cascade over a freshly
  detected one.

### 1.3 A covariance crash that only appears in the field

Sequences occasionally drive a track's covariance indefinite after many missed frames.
The original float32 Cholesky raised `LinAlgError` inside the Cholesky helper in
`src/mot/kalman.py`.

Isolated by capture: the failing 8x8 covariance already had minimum eigenvalue ≈
**−6.53e4**, and the projected covariance ≈ **−3.26e4** — not a rounding wobble but a
genuinely indefinite matrix. Six candidate rewrites were each run over a 120-frame
sequence; **all six produced identical metrics**, which is what identified the repair as
behaviour-preserving on healthy input.

Then the honest split:

- **Proven fix** — the escalating-jitter fallback in the Cholesky helpers. Restoring the
  old helper turns the captured fixture red.
- **Preventive hardening** — float64 update arithmetic, the Joseph covariance form, and
  float64 symmetrization. These are correct in principle, but **no test demonstrates they
  repair anything** on this dataset. They are kept because they are free at this data
  scale, not because they fixed an observed failure.

A regression test asserts only the *observable* property — the next factorization
survives — and deliberately does not require Joseph form, so it cannot give false credit
for the preventive changes.

### 1.4 Smaller ones

- Detector OOM retry halved the batch until it fit. A test forces OOM and asserts the
  frame order and count are preserved, because a dropped or duplicated frame silently
  corrupts every downstream metric.
- `min_hits` was applied at the wrong point, and an empty detection frame did not age
  existing tracks at all.
- `num_matches` was missing from the aggregate's sum list, so `metrics.json` reported
  `"num_matches": 0.0` for the OVERALL row while every per-sequence value was non-zero.
- Recomputing metrics from the written `.txt` files differs from the in-run numbers by
  0–5 matches per sequence (four of the seven SORT sequences agree exactly on
  `num_matches`; none agree on every scalar to the last digit, because MOTP is itself a
  ratio over those same coordinates). This is
  **not** a bug: `write_results` emits coordinates at
  `%.2f`, which perturbs IoU enough to flip matches sitting within ~1e-4 of the 0.5
  threshold. `metrics.json` from the pipeline run is therefore the authoritative source,
  and `scripts/reevaluate.py` is for regenerating metrics after a *metrics-code* change.

---

## 2. Detection, not association, is the binding constraint

Tracker comparisons are only meaningful once you know how much of the gap is upstream.
`scripts/detector_ceiling.py` measures the ceiling: greedy one-to-one matching at
IoU ≥ 0.5 against ground truth, with the same class filtering as the evaluator and
the same ≥50%-containment ignore-region rule (the evaluator keeps
`inside_fraction <= 0.5` in `metrics.py:138`, the ceiling script suppresses
`>= 0.5` in `detector_ceiling.py:93`; they differ only at exactly 50%, which the
tests pin deliberately), and **no tracker at all**.

| weights | imgsz | conf | recall | precision | TP | det | GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 1280 | 0.25 | 27.65% | 85.28% | 614 | 720 | 2221 |
| yolo11m | 1280 | 0.25 | 32.96% | 83.94% | 732 | 872 | 2221 |
| yolo11l | 1280 | 0.25 | 32.01% | 84.24% | 711 | 844 | 2221 |
| **yolo11x** | **1600** | **0.25** | **40.34%** | 80.72% | 896 | 1110 | 2221 |
| yolo11x | 1600 | 0.20 | 42.64% | 78.20% | 947 | 1211 | 2221 |
| yolo11x | 1600 | 0.30 | 38.27% | 83.01% | 850 | 1024 | 2221 |
| yolo11x | 1600 | 0.35 | 35.88% | 85.24% | 797 | 935 | 2221 |
| yolo11x | 1600 | 0.40 | 33.41% | 88.76% | 742 | 836 | 2221 |

(MOT17-02, first 60 frames, 2,221 ground-truth boxes. These numbers describe
that slice only — compare detector recall on this slice against tracker recall
on the same slice; a full-sequence tracker score is not directly comparable.)

The matching logic behind this table is unit-tested (`tests/test_detector_ceiling.py`,
14 tests, hand-computed IoU boundaries and one-to-one enforcement) because the entire
"detection is the bottleneck" claim rests on it.

Conclusion: recall is roughly **half** of what a perfect tracker would need. Every
tracker here is mostly being graded on how it handles detections that never arrive.
That reframes the occlusion result in §3: identity recovery is a real lever, but it
cannot rescue a detector that misses half the pedestrians.

---

## 3. Results

Detector fixed at `yolo11x` / `imgsz=1600` / `conf=0.25` for both trackers.

### Per sequence (SORT / occlusion)

| sequence | MOTA | MOTP | HOTA | IDF1 | IDS | Frag | MT | ML | Rec | Prec | Matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02 | 25.03 / 27.86 | 80.93 / 79.94 | 16.62 / 22.20 | 0.191 / 0.264 | 522 / 434 | 633 / 593 | 3.0 / 7.0 | 34.0 / 31.0 | 27.76 / 31.94 | 92.52 / 89.79 | 7340 / 8473 |
| MOT17-04 | 42.10 / 43.79 | 84.21 / 83.69 | 34.57 / 39.58 | 0.349 / 0.419 | 844 / 528 | 1075 / 879 | 24.0 / 27.0 | 30.0 / 31.0 | 48.12 / 50.61 | 89.83 / 88.69 | 30343 / 32073 |
| MOT17-05 | 7.50 / 15.20 | 75.41 / 74.59 | 15.93 / 26.97 | 0.159 / 0.308 | 260 / 248 | 318 / 261 | 0.0 / 16.0 | 85.0 / 38.0 | 21.24 / 40.40 | 61.19 / 62.07 | 1548 / 2949 |
| MOT17-09 | 49.97 / 57.13 | 81.79 / 80.61 | 22.85 / 34.23 | 0.222 / 0.433 | 263 / 156 | 269 / 157 | 1.0 / 8.0 | 2.0 / 1.0 | 56.64 / 66.32 | 92.94 / 89.70 | 3158 / 3769 |
| MOT17-10 | 32.66 / 45.54 | 77.28 / 75.32 | 16.26 / 31.57 | 0.154 / 0.397 | 587 / 343 | 692 / 765 | 3.0 / 13.0 | 22.0 / 8.0 | 38.52 / 58.17 | 89.65 / 83.50 | 5249 / 8069 |
| MOT17-11 | 43.02 / 38.77 | 84.29 / 83.77 | 26.51 / 36.30 | 0.217 / 0.369 | 274 / 170 | 302 / 171 | 9.0 / 22.0 | 32.0 / 31.0 | 57.60 / 63.72 | 81.24 / 72.43 | 5325 / 5957 |
| MOT17-13 | 24.22 / 44.97 | 78.21 / 77.61 | 17.79 / 33.26 | 0.214 / 0.398 | 267 / 325 | 346 / 489 | 3.0 / 27.0 | 67.0 / 22.0 | 26.59 / 54.68 | 93.65 / 86.64 | 3052 / 6245 |

Cells are `SORT / occlusion`. Regenerate with `scripts/make_report.py`.

**Overall**

| metric | SORT | occlusion | Δ (occ − SORT) |
| --- | --- | --- | --- |
| mota | 34.87 | 39.66 | +4.80 (better) |
| motp | 82.43 | 81.09 | -1.34 (worse) |
| hota | 21.51 | 32.01 | +10.51 (better) |
| idf1 | 0.267 | 0.379 | +0.11 (better) |
| num_switches | 3017 | 2204 | -813.00 (better) |
| num_fragmentations | 3635 | 3315 | -320.00 (better) |
| mostly_tracked | 6.1 | 17.1 | +11.00 (better) |
| mostly_lost | 38.9 | 23.1 | -15.71 (better) |
| recall | 40.97 | 49.11 | +8.13 (better) |
| precision | 88.48 | 84.79 | -3.69 (worse) |
| num_matches | 56015 | 67535 | +11520.00 (better) |

`OVERALL` pools the MOTChallenge rates (MOTA, precision, recall), weights MOTP by
matches, and recomputes IDF1 from pooled ID counts; only HOTA and its components are
unweighted sequence means, and count metrics are summed — see §5.

**Overall: the occlusion tracker wins on every identity metric and loses on precision.**

| metric | SORT | occlusion | delta | |
| --- | --- | --- | --- | --- |
| MOTA | 34.87 | **39.66** | **+4.80** | better |
| HOTA | 21.51 | **32.01** | **+10.51** | better |
| IDF1 | 0.267 | **0.379** | **+0.112** | better |
| MOTP | **82.43** | 81.09 | −1.34 | worse |
| recall | 40.97 | **49.11** | **+8.13** | better |
| precision | **88.48** | 84.79 | −3.69 | worse |
| ID switches | 3017 | **2204** | **−813 (−27%)** | better |
| fragmentations | 3635 | **3315** | **−320** | better |
| mostly tracked | 6.1 | **17.1** | **+11.0** | better |
| mostly lost | 38.9 | **23.1** | **−15.7** | better |

The identity metrics are the point of the experiment and they all move the right
way: **−27% identity switches**, mostly-tracked nearly triples, and mostly-lost
drops by more than a third. HOTA separates them most sharply (**21.5 → 32.0,
+10.5**), consistent with HOTA scoring association quality rather than detection
quality alone. The recall gain (+8.1) and the switch reduction come together,
but attributing the gain to track retention specifically is an inference, not a
measurement: this is one composite variant against one baseline, at one
detector configuration, with no per-component ablation.

The cost is precision (−3.7). A track that survives an occlusion without being
re-acquired keeps emitting boxes where no pedestrian is annotated, which scores
as false positive and as localization error (MOTP −1.3); that is the most
plausible mechanism, and it is consistent with the per-sequence precision
pattern below, but it was not isolated by experiment. The tracker trades
"confidently wrong less often" for "remember more often". For a surveillance or
re-identification workload that is the correct direction; for a
count-vehicles-on-a-roadline workload it would not be.

Sequence 11 is the honest counter-example: it is the one sequence where the
occlusion tracker **loses** MOTA (43.02 → 38.77) even though it improves identity
(IDF1 0.217 → 0.369, switches 274 → 170). Its precision drop is the largest of any
sequence (81.24 → 72.43). A tracker that holds identity across camera motion and
long empty stretches accumulates stale tracks faster than it recovers them, and
no amount of association cleverness fixes that — only better detection would.
Sequence 13 shows the same effect at the other extreme: MOTA nearly doubles
(24.22 → 44.97) and recall more than doubles (26.59 → 54.68), but precision still
falls.

Taken with §2, the headline is a ceiling: the detector alone misses more than
half the boxes on MOT17-02's first 60 frames (40.34% recall), and the
full-sequence recall numbers in §3 are low for the same reason — though §2
measures a 60-frame slice and §3 measures full sequences, so the two are not
directly comparable. The measured gap between the two trackers is +4.8 MOTA /
+0.11 IDF1; the hand-rolled appearance term accounts for part of it at unknown
share. A trained person-ReID embedding is plausibly worth more, but nothing
here measures that.

---

## 4. Negative and null results

Recording these because they cost time and constrain the design:

- **A generic ImageNet embedding is not person-ReID.** The 1088-d MobileNetV3-small/HSV
  descriptor separates pedestrians from background well enough to gate associations, but
  it is not trained to tell two similar individuals apart. Download attempts for a
  trained person-ReID checkpoint returned `401`/`404`. The appearance cue helps, but the
  headroom a real ReID encoder would unlock is the largest single improvement available.
- **Raising the detector confidence does not fix recall.** The §2 sweep shows
  `conf=0.30` trades recall for precision at the detector level (38.27% vs
  40.34% recall on the same slice). Whether that trade lowers end-to-end MOTA
  was not measured — the pipeline keeps `det_conf_threshold=0.3` for track
  starts; lower-confidence detections still sustain existing ones.
- **Longer `max_age` is not free.** Retaining coasting tracks longer preserves identity
  but converts missed detections into stale false positives; MOTA degrades before HOTA
  does.
- **`motmetrics` cannot be used for HOTA**, and one of its helpers
  (`distances.iou_matrix`, via the removed `np.asfarray`) cannot be used at all on
  NumPy 2 — see README.
- **`scripts/reevaluate.py` reproduces `metrics.json` to rounding, not bit-exactly.**
  Result files are written at `%.2f` precision, so re-scoring from `.txt` can flip a
  match whose IoU sits within ~1e-4 of the 0.5 threshold. Measured over the seven
  SORT sequences, the largest per-sequence disagreement is **0.0017 HOTA**
  (4th decimal place) and **0.014 MOTA** (3rd), consistent with quantisation alone; the
  metric path itself is reproducible. It is a debugging and re-scoring tool rather
  than a substitute for the in-run metrics, since it cannot see detections that
  `min_hits` suppressed before they were ever written.
- **The occlusion tracker is ~6.5x slower end-to-end.** The published per-sequence
  timings are 1.8–2.0 fps against SORT's 13.1–14.8 fps, so a seven-sequence
  benchmark takes 41 min instead of 6.3. The cost scales with sequence length and
  track count, which points at per-track numpy association work rather than
  detection (the appearance encoder is already on the GPU) — but that attribution is
  inference. No CPU/GPU utilisation profile was recorded, so the specific split is
  not measured. MOT17-05 is the outlier at 8.0 fps, and the reason is not
  established here.

---

## 5. Reproducing

```bash
./venv/bin/pip install -r requirements.txt        # CUDA 13.0 wheels; see requirements.txt
./venv/bin/python tests/run_all.py                # 60 tests, 5 suites
./scripts/run_full_benchmark.sh                   # both trackers, all 7 sequences
PYTHONPATH=src ./venv/bin/python scripts/detector_ceiling.py \
    --sequence 02 --configs conf-sweep --out results/detector_ceiling_02.json
```

The benchmark takes roughly 6–7 minutes for SORT and 45–50 minutes for the occlusion
tracker at `imgsz=1600` on one RTX 5060.

**Reproducibility caveats, stated rather than hidden:**

- Absolute metrics depend on the GPU and on `yolo11x` weight drift; the *ranking*
  (SORT vs occlusion) is what this project claims.
- HOTA follows the TrackEval formulation (association via a sequence-wide Jaccard over
  global `(gt id, tracker id)` pairs, per-frame assignment driven by that global
  alignment score), and the implementation is cross-checked against an independent
  transcription and hand-computed cases in `tests/test_metrics.py`. **But the class
  convention differs from the official MOTChallenge protocol**, so the values are
  *not* directly comparable to published HOTA numbers. This project treats classes
  1/2/4/7 as targets; the official protocol scores class 1 only (MOT17-02 has 18,581
  class-1 boxes against this project's 26,960 evaluated targets). Measured on
  *identical* SORT hypothesis files, restricting ground truth to class 1 alone shifts
  per-sequence HOTA by **+0.08 to +2.38 (mean +0.91)** — larger than some of the
  effects discussed above. TrackEval additionally deletes hypotheses matched to
  distractor classes, which pushes the offset slightly the other way. Treat
  cross-paper comparisons as invalid until the class sets are normalised.
- Aggregation is mixed, and conflating the two schemes is itself a silent bug. The
  MOTChallenge rates are **pooled** from summed event counts, MOTP is
  **match-weighted**, and IDF1 is **recomputed** from pooled `idtp`/`idfp`/`idfn` —
  these coincide with TrackEval's `COMBINED_SEQ`. **HOTA and its components are an
  unweighted sequence mean**, which is *not*: TrackEval pools per-alpha TP counts and
  association sums across sequences and recomputes. So `OVERALL` MOTA is a valid
  TrackEval-style combined score while `OVERALL` HOTA is not, and only the latter must
  not be presented as a TrackEval combined value. The gap is material, not cosmetic —
  for SORT, pooled MOTA is 34.87 against a sequence mean of 32.07.
- Ignore regions are applied to hypotheses only (see `README.md`). Ground-truth boxes
  centred inside an ignore region still count as missed targets. Measured over the seven
  SORT sequences, excluding those ground-truth boxes as well moves HOTA by **+0.91 on
  average**, ranging from **−0.14 (MOT17-09)** to **+2.16 (MOT17-04)**, and MOTA by
  **−0.83 on average**, reaching **−6.36 (MOT17-09)**. HOTA is the steadier of the
  two, but not steady enough to ignore: MOT17-04's +2.16 is ~10% of this project's
  21.5 HOTA baseline.
- Metrics recomputed from `.txt` agree with `metrics.json` to within 0.002 HOTA and
  0.015 MOTA per sequence; the residual is `%.2f` quantisation, not a metric
  disagreement.
## SOTA comparison (MOT17-FRCNN, full 7 sequences)

| Tracker | MOTA | MOTP | HOTA | IDF1 | IDS | Frag | MT | ML | Rec | Prec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ByteTrack | 40.89 | 81.13 | 40.39 | 0.490 | 644 | 1628 | 17.7 | 26.7 | 47.73 | 87.76 |
| OC-SORT | 40.60 | 81.74 | 40.89 | 0.517 | 561 | 1972 | 16.4 | 27.0 | 47.30 | 87.86 |
| BoT-SORT | 41.10 | 82.01 | 42.38 | 0.512 | 320 | 1394 | 18.4 | 28.4 | 46.55 | 89.68 |
| Occlusion-Aware (ours) | 39.66 | 81.09 | 32.01 | 0.379 | 2204 | 3315 | 17.1 | 23.1 | 49.11 | 84.79 |
| SORT (ours baseline) | 34.87 | 82.43 | 21.51 | 0.267 | 3017 | 3635 | 6.1 | 38.9 | 40.97 | 88.48 |

## Ablation study (MOT17-FRCNN, full 7 sequences)

| Configuration | MOTA | MOTP | HOTA | IDF1 | IDS | Frag | MT | ML | Rec | Prec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full (ours) | 39.66 | 81.09 | 32.01 | 0.379 | 2204 | 3315 | 17.1 | 23.1 | 49.11 | 84.79 |
| w/o appearance | 39.48 | 81.13 | 31.61 | 0.369 | 2376 | 3324 | 16.9 | 23.7 | 48.69 | 85.09 |
| w/o REID recovery | 39.61 | 81.13 | 31.85 | 0.370 | 2210 | 3323 | 16.9 | 23.4 | 48.79 | 85.09 |
| w/o cascade (Hungarian) | 39.81 | 81.05 | 31.26 | 0.388 | 1730 | 3059 | 21.3 | 23.3 | 49.76 | 84.03 |

