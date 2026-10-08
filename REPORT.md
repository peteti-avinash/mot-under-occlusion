# Multi-Object Tracking Under Occlusion

## 1. Problem

Multi-object tracking fails most visibly when pedestrians cross, leave the
detector briefly, or reappear after an occlusion. The important question is not
only whether a box is present, but whether it keeps the same identity. This
project evaluates whether an appearance-assisted tracker can preserve identities
better than classic SORT while using the same detector for both systems.

## 2. Approach

The baseline is SORT: constant-velocity Kalman prediction with IoU-cost
Hungarian association. The custom occlusion-aware tracker retains that motion
model and adds age-ordered cascade matching, Mahalanobis gating, a lost-track
lifecycle, a gallery of generic MobileNetV3-small plus HSV descriptors, and
appearance-gated re-identification. It is a custom tracker, not DeepSORT; its
appearance encoder is ImageNet-pretrained rather than person-ReID trained.

Both variants use the same fixed COCO-pretrained `yolo11x.pt` detector at
`imgsz=1600` and confidence 0.25. No detector was trained. The benchmark uses
the MOT17 FRCNN training sequences 02, 04, 05, 09, 10, 11, and 13. Evaluation
treats MOT17 classes 1, 2, 4, and 7 as targets and suppresses hypotheses in
annotated ignore regions, so these scores are not directly comparable with the
official MOTChallenge leaderboard.

## 3. Metrics

The evaluator reports MOTA, MOTP, HOTA, IDF1, ID switches, fragmentation,
precision, recall, mostly tracked, and mostly lost. MOTA, precision, recall,
MOTP, and IDF1 are aggregated from pooled event counts; HOTA is an unweighted
mean over sequences rather than TrackEval's combined HOTA. This makes the
aggregation explicit and reproducible from the retained `metrics.json` files.

## 4. Results

| Tracker | MOTA | MOTP | HOTA | IDF1 | ID Switches | Fragmentation | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SORT | 34.87 | 82.43 | 21.51 | 0.267 | 3017 | 3635 | 40.97 | 88.48 |
| Occlusion-aware | 39.66 | 81.09 | 32.01 | 0.379 | 2204 | 3315 | 49.11 | 84.79 |

With detection held fixed, the occlusion-aware tracker improves MOTA by **4.79**
points, HOTA by **10.50** points, and IDF1 by **0.112**. It reduces ID switches
by **813** (**26.95%**) and fragmentation by 320. Recall improves by **8.14**
points. The tradeoff is a **3.69**-point precision decrease and a 1.34-point
MOTP decrease. The custom tracker therefore improves identity continuity and
coverage, while allowing more false or less accurate retained hypotheses.

## 5. Failure Analysis

The observed precision reduction is consistent with retained tracks emitting
hypotheses after a person is no longer annotated or detected. Long occlusions
remain ambiguous, especially for pedestrians with similar clothing because the
descriptor is generic rather than trained for person identity. Abrupt motion
can also violate the constant-velocity prediction and gating assumptions.
Detector recall remains a hard ceiling: association cannot recover a pedestrian
box that was never detected. These mechanisms explain the measured tradeoff but
do not attribute every switch or error to an individual component without a
sequence-level ablation.

## 6. Limitations

The benchmark is one detector configuration and one composite tracker variant,
not a hyperparameter sweep or a trained-ReID comparison. The detector is
computationally demanding at `imgsz=1600` on the RTX 5060 Laptop GPU used for
the run, and the appearance computation makes the occlusion-aware tracker
substantially slower. The local data, weights, result JSON, logs, and 1,050-frame
annotated video are intentionally Git-ignored, so a clone must recreate or
retain those artifacts before reproducing the result.

## 7. Conclusion

On the retained MOT17 FRCNN seven-sequence benchmark, the custom
occlusion-aware tracker improves identity preservation over SORT: HOTA rises
from 21.51 to 32.01, IDF1 from 0.267 to 0.379, and ID switches fall from 3017 to
2204. The recall gain comes with lower precision, which is the central practical
tradeoff. A trained person-ReID encoder and detector improvements are the most
valuable next steps.
