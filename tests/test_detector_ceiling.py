"""Tests for the detector-ceiling measurement script.

The sweep in scripts/detector_ceiling.py is the evidence behind the write-up's
central claim that detection recall, not association, is the binding
constraint. If its matching logic is wrong that claim is worthless, so the
one-to-one matching and the ignore-region suppression are pinned here with
hand-computed values rather than left as ad-hoc script code.

These are pure-NumPy and touch no GPU.
"""
import importlib.util
import os

import numpy as np

from _runner import run_tests

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, "scripts", "detector_ceiling.py")
_spec = importlib.util.spec_from_file_location("detector_ceiling", _PATH)
ceiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ceiling)


def assert_true(condition, message=""):
    if not condition:
        raise AssertionError(message)


def test_iou_identical_boxes_is_one():
    box = [0, 0, 10, 10]
    value = ceiling._iou(ceiling._corners_xyxy(box), ceiling._corners_xyxy(box))
    assert_true(value == 1.0, f"identical boxes should have IoU 1.0, got {value}")


def test_iou_disjoint_boxes_is_zero():
    a = ceiling._corners_xyxy([0, 0, 10, 10])
    b = ceiling._corners_xyxy([20, 20, 10, 10])
    value = ceiling._iou(a, b)
    assert_true(value == 0.0, f"disjoint boxes should have IoU 0.0, got {value}")


def test_iou_half_overlap_is_exactly_threshold():
    """Half-height overlap: inter=50, union=100, so IoU is exactly 0.5."""
    a = ceiling._corners_xyxy([0, 0, 10, 10])
    b = ceiling._corners_xyxy([0, 0, 10, 5])
    value = ceiling._iou(a, b)
    assert_true(abs(value - 0.5) < 1e-9, f"expected IoU 0.5, got {value}")


def test_greedy_match_perfect_pair():
    # _greedy_match takes dets as xyxy and gts as tlwh, both bare 4-vectors.
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    det = np.array([[0, 0, 10, 10]], dtype=np.float32)  # xyxy
    tp, n_det, n_gt = ceiling._greedy_match(det, gt)
    assert_true((tp, n_det, n_gt) == (1, 1, 1),
          f"perfect pair should give (1,1,1), got {(tp, n_det, n_gt)}")


def test_greedy_match_is_one_to_one():
    """Two detections on one ground truth must yield a single true positive.

    Without the one-to-one constraint a duplicated detection would inflate
    recall, which is exactly the failure that would invalidate the sweep.
    """
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    det = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    tp, n_det, n_gt = ceiling._greedy_match(det, gt)
    assert_true((tp, n_det, n_gt) == (1, 2, 1),
          f"duplicate detections should give (1,2,1), got {(tp, n_det, n_gt)}")


def test_greedy_match_rejects_below_threshold():
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    det = np.array([[0, 0, 10, 6]], dtype=np.float32)  # inter=60, union=100
    # 10x6 against 10x10: inter=60, union=100 -> 0.6 >= 0.5, should match.
    tp, _, _ = ceiling._greedy_match(det, gt)
    assert_true(tp == 1, f"IoU 0.6 should match, got {tp}")
    det_low = np.array([[0, 0, 10, 4]], dtype=np.float32)  # 40/100 = 0.4
    tp_low, _, _ = ceiling._greedy_match(det_low, gt)
    assert_true(tp_low == 0, f"IoU 0.4 should not match, got {tp_low}")


def test_greedy_match_picks_best_pairing():
    """The greedy order must take the highest-IoU pair first.

    gt0 sits at x=0..10, gt1 at x=20..30. det0 covers both partially but is a
    better match for gt1; det1 is a perfect match for gt0. Matching det0->gt0
    first would leave det1 with nothing.
    """
    gt = np.array([[0, 0, 10, 10], [20, 0, 10, 10]], dtype=np.float32)
    det = np.array([[15, 0, 30, 10],   # perfect for gt1
                    [0, 0, 10, 10]],   # perfect for gt0
                   dtype=np.float32)
    tp, n_det, n_gt = ceiling._greedy_match(det, gt)
    assert_true((tp, n_det, n_gt) == (2, 2, 2),
          f"both perfect detections should match, got {(tp, n_det, n_gt)}")


def test_greedy_match_with_no_ground_truth():
    det = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    gt = np.zeros((0, 4), dtype=np.float32)
    tp, n_det, n_gt = ceiling._greedy_match(det, gt)
    assert_true((tp, n_det, n_gt) == (0, 1, 0),
          f"detection against empty GT should give (0,1,0), got {(tp, n_det, n_gt)}")


def test_suppressed_when_fully_inside_region():
    dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
    regions = np.array([[7, 0, 0, 10, 10]], dtype=np.float32)  # frame id then tlwh
    out = ceiling._suppressed(dets, regions)
    assert_true(bool(out[0]), "a fully contained detection must be suppressed")


def test_not_suppressed_when_disjoint_from_region():
    dets = np.array([[20, 20, 10, 10]], dtype=np.float32)
    regions = np.array([[7, 0, 0, 10, 10]], dtype=np.float32)
    out = ceiling._suppressed(dets, regions)
    assert_true(not bool(out[0]), "a detection far from the region must survive")


def test_suppressed_uses_max_across_multiple_regions():
    """inside_fraction handles one region, so _suppressed must take the max."""
    dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
    regions = np.array([[7, 50, 50, 10, 10],   # far away, no overlap
                        [7, 0, 0, 10, 10]],   # exact match
                       dtype=np.float32)
    out = ceiling._suppressed(dets, regions)
    assert_true(bool(out[0]),
          "suppression must consider every region, not just the first")


def test_half_overlap_is_suppressed_at_boundary():
    """Overlap fraction exactly 0.5 counts as suppressed (>= 0.5)."""
    dets = np.array([[5, 0, 10, 10]], dtype=np.float32)   # x 5..15
    regions = np.array([[7, 0, 0, 10, 10]], dtype=np.float32)  # x 0..10
    out = ceiling._suppressed(dets, regions)
    assert_true(bool(out[0]), "exactly-half-contained detection must be suppressed")


class _StubDetector:
    """Returns a fixed set of boxes so evaluate_config can be tested offline."""

    def __init__(self, xyxy, classes):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.classes = np.asarray(classes)
        self.calls = 0

    def detect_batch(self, images):
        self.calls += 1
        return [(self.xyxy, np.ones(len(self.xyxy), np.float32), self.classes)]


def _with_stubbed_read_frame(fn):
    """Swap read_frame for a stub for the duration of ``fn``."""
    original = ceiling.read_frame

    def _fake(seq, frame_id, cache):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    ceiling.read_frame = _fake
    try:
        return fn()
    finally:
        ceiling.read_frame = original


def test_evaluate_config_excludes_suppressed_dets_from_the_denominator():
    # Regression: the ignore-region mask was computed and then dropped, so a
    # suppressed detection still counted in n_det and could still be matched.
    # Det 0 matches the GT and sits outside the ignore region; det 1 sits fully
    # inside it. Only det 0 may count.
    xyxy = np.array([[100, 100, 110, 110],      # outside ignore region
                     [110, 110, 120, 120]], np.float32)   # inside
    det = _StubDetector(xyxy, [0, 0])
    gt = {1: np.array([[1, 100, 100, 10, 10]], np.float32)}
    ignore = {1: np.array([[1, 105, 105, 40, 40]], np.float32)}

    def _go():
        return ceiling.evaluate_config(det, "stub", [1], gt, ignore,
                                      batch_size=8)

    recall, precision, tp, n_det, n_gt = _with_stubbed_read_frame(_go)
    assert_true(n_det == 1, f"suppressed detection must not count: n_det={n_det}")
    assert_true(n_gt == 1, f"n_gt={n_gt}")
    assert_true(tp == 1, f"the un-suppressed detection should match: tp={tp}")
    assert_true(abs(precision - 1.0) < 1e-9,
                f"precision should be 1.0, got {precision} (0.5 is the bug)")
    assert_true(abs(recall - 1.0) < 1e-9, f"recall should be 1.0, got {recall}")


def test_evaluate_config_counts_everything_when_there_is_no_ignore_file():
    xyxy = np.array([[100, 100, 110, 110]], np.float32)
    det = _StubDetector(xyxy, [0])
    gt = {1: np.array([[1, 100, 100, 10, 10]], np.float32)}

    def _go():
        return ceiling.evaluate_config(det, "stub", [1], gt, {}, batch_size=8)

    recall, precision, tp, n_det, n_gt = _with_stubbed_read_frame(_go)
    assert_true(n_det == 1 and tp == 1, f"tp={tp} n_det={n_det}")
    assert_true(abs(precision - 1.0) < 1e-9, precision)


if __name__ == "__main__":
    raise SystemExit(run_tests(dict(globals()), "detector_ceiling"))