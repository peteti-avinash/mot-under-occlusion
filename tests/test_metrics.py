"""Analytic checks for the local metric implementations.

These are hand-computed expectations rather than fixtures, so they pin the
definitions themselves. The HOTA cases are the important ones: the previous
implementation dropped the identity-switch term ``H`` and scored 1.0 even when
every frame carried a different track id, which made HOTA a pure detection
score. Any change to ``hota``'s documented definition must update these.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from mot.metrics import hota  # noqa: E402


def _box(track_id, x=0.0):
    return np.array([[track_id, x, 0.0, 10.0, 10.0]], dtype=np.float64)


def test_hota_perfect_identity_is_exactly_one():
    n = 10
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    hyp = {t: _box(1) for t in frames}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(value - 1.0) < 1e-9, value
    assert abs(det_a - 1.0) < 1e-9, det_a
    assert abs(ass_a - 1.0) < 1e-9, ass_a


def test_hota_penalises_identity_switches():
    """Pixel-identical boxes, so only the association term can move HOTA."""
    n = 10
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}

    # One switch at frame 6: 9 frames at AssA 1, 1 frame at AssA 1/2,
    # so A = 9.5 and HOTA = sqrt(9.5)/sqrt(10).
    one_switch = {t: _box(1 if t < 6 else 2) for t in frames}
    value, det_a, _ = hota(one_switch, gt, n)
    assert abs(value - math.sqrt(9.5) / math.sqrt(10)) < 1e-9, value
    assert abs(det_a - 1.0) < 1e-9, "DetA must stay pinned at 1.0"

    # A different id every frame: 1 clean frame plus 9 switched,
    # A = 1 + 9/2 = 5.5, HOTA = sqrt(5.5)/sqrt(10).
    all_switch = {t: _box(t) for t in frames}
    scrambled, det_a, _ = hota(all_switch, gt, n)
    assert abs(scrambled - math.sqrt(5.5) / math.sqrt(10)) < 1e-9, scrambled
    assert abs(det_a - 1.0) < 1e-9, "DetA must stay pinned at 1.0"
    assert scrambled < value, "more switches did not score worse"


def test_hota_half_recall_with_perfect_association():
    """Two gt people, one always detected: DetA 0.5, A 5, denom sqrt10+sqrt10."""
    n = 10
    frames = range(1, n + 1)
    gt = {t: np.concatenate([_box(1, 0.0), _box(2, 500.0)]) for t in frames}
    hyp = {t: _box(1, 0.0) for t in frames}
    value, det_a, ass_a = hota(hyp, gt, n)
    assert abs(value - math.sqrt(5.0) / (math.sqrt(10) + math.sqrt(10))) < 1e-9, value
    assert abs(det_a - 0.5) < 1e-9, det_a
    assert abs(ass_a - 0.5) < 1e-9, ass_a


def test_hota_is_zero_without_hypotheses():
    n = 5
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    assert hota({}, gt, n) == (0.0, 0.0, 0.0)


def test_hota_localisation_error_lowers_score_monotonically():
    """A box that drifts further from the gt must never score higher."""
    n = 6
    frames = range(1, n + 1)
    gt = {t: _box(1) for t in frames}
    previous = None
    for offset in (0.0, 1.0, 2.0, 3.0, 4.0):
        hyp = {t: _box(1, offset) for t in frames}
        value = hota(hyp, gt, n)[0]
        if previous is not None:
            assert value < previous, (
                f"offset {offset} scored {value}, not below {previous}")
        previous = value


if __name__ == "__main__":
    from _runner import run_tests

    raise SystemExit(run_tests(dict(globals()), "metrics"))