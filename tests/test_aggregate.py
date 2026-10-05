"""Aggregate-semantics tests for metrics.aggregate.

The bug these pin: OVERALL averaged *rates* across sequences. That is wrong for
MOTChallenge-style aggregates, where every sequence's objects are pooled into a
single global confusion matrix. It also silently dropped ``num_matches`` from
the summed fields, so ``metrics.json`` reported ``"num_matches": 0.0`` for the
OVERALL row while every per-sequence value was non-zero.

Fixtures satisfy the invariant CLEAR maintains for every real sequence:
``num_objects == num_matches + num_misses`` (each ground-truth box is either
matched or missed). Two deliberately lopsided sequences are used so that
"pooled" and "mean" cannot coincide by accident:

    seq A: 90 matches,  5 misses, 10 FP, 0 switches -> 95 objects
    seq B:  1 match,   9 misses,  0 FP, 0 switches -> 10 objects

    pooled (matches 91, misses 14, FP 10, objects 105)
        MOTA      = 1 - (14 + 10 + 0) / 105 = 77.142857
        recall    = 91 / (91 + 14)         = 86.666667
        precision = 91 / (91 + 10)         = 90.099010
        MOTP      = (100*90 + 0*1) / 91    = 98.901099
        IDF1      = 2*91 / (2*91 + 10 + 14) = 0.883495
                     (idfp = FP = 10, idfn = misses = 5 + 9 = 14)

    mean: MOTA 47.11, recall 52.37, precision 95.00, MOTP 50.00

Those means are recorded next to the pooled values because they are exactly
what the buggy implementation returned, so a failure message can name the
concrete wrong number instead of just "not equal".
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from mot.metrics import TrackMetrics, aggregate  # noqa: E402


def _seq(name, matches, misses, fp, switches, motp, hota):
    """A sequence obeying ``objects == matches + misses``."""
    objects = matches + misses
    return TrackMetrics(
        sequence=name, num_objects=objects, num_matches=matches,
        num_misses=misses, num_switches=switches, num_false_positives=fp,
        motp=motp, hota=hota, num_fragmentations=switches * 2,
        mostly_tracked=objects / 10.0, mostly_lost=objects / 20.0,
        num_frames=100,
        mota=100.0 * (1.0 - (misses + fp + switches) / objects),
        recall=100.0 * matches / (matches + misses),
        precision=100.0 * matches / (matches + fp) if matches + fp else 0.0,
        idtp=matches, idfp=fp, idfn=objects - matches,
    )


A = _seq("A", matches=90, misses=5, fp=10, switches=0, motp=100.0, hota=40.0)
B = _seq("B", matches=1, misses=9, fp=0, switches=0, motp=0.0, hota=20.0)


def test_fixtures_obey_the_clear_object_invariant():
    # If this fails the hand-computed expectations below are meaningless.
    for row in (A, B):
        assert row.num_objects == row.num_matches + row.num_misses, row.sequence


def test_num_matches_is_summed_not_dropped():
    agg = aggregate([A, B])
    assert agg.num_matches == 91, (
        f"num_matches should be the SUM across sequences, got {agg.num_matches}")


def test_counts_are_summed():
    agg = aggregate([A, B])
    assert agg.num_objects == 105
    assert agg.num_misses == 14
    assert agg.num_false_positives == 10
    assert agg.num_switches == 0
    assert agg.num_frames == 200


def test_mota_is_pooled_not_averaged():
    agg = aggregate([A, B])
    assert abs(agg.mota - 77.1428571) < 1e-6, (
        f"MOTA must be pooled from the global confusion matrix: expected "
        f"77.1429, got {agg.mota:.4f} (a mean of 47.11 is the bug)")


def test_recall_is_pooled_not_averaged():
    agg = aggregate([A, B])
    assert abs(agg.recall - 86.6666667) < 1e-6, (
        f"recall must be pooled: expected 86.6667, got {agg.recall:.4f} "
        f"(a mean of 52.37 is the bug)")


def test_precision_is_pooled_not_averaged():
    agg = aggregate([A, B])
    assert abs(agg.precision - 90.0990099) < 1e-6, (
        f"precision must be pooled: expected 90.0990, got {agg.precision:.4f} "
        f"(a mean of 95.00 is the bug)")


def test_motp_is_match_weighted():
    agg = aggregate([A, B])
    assert abs(agg.motp - 98.9010989) < 1e-6, (
        f"MOTP is a mean localization cost over MATCHES, so it must be "
        f"match-weighted: expected 98.9011, got {agg.motp:.4f} "
        f"(a mean of 50.00 is the bug)")


def test_hota_stays_a_sequence_mean():
    # HOTA has no per-object count to pool with; averaging is the documented
    # choice, pinned here so a future "pool everything" refactor cannot
    # silently change a published number.
    agg = aggregate([A, B])
    assert abs(agg.hota - 30.0) < 1e-6, agg.hota


def test_mt_ml_stay_means():
    agg = aggregate([A, B])
    assert abs(agg.mostly_tracked - 5.25) < 1e-6, agg.mostly_tracked
    assert abs(agg.mostly_lost - 2.625) < 1e-6, agg.mostly_lost


def test_idf1_single_sequence_matches_its_own_definition():
    # Guards the identity property for idf1 specifically: the pooled formula must
    # reproduce one sequence's own IDF1, which is what the per-sequence path
    # computes from idtp/idfp/idfn.
    agg = aggregate([A])
    assert abs(agg.idf1 - 2 * 90.0 / (2 * 90.0 + 10 + 5)) < 1e-9, agg.idf1


def test_idf1_is_recomputed_from_pooled_id_counts():
    agg = aggregate([A, B])
    assert abs(agg.idf1 - 182.0 / 206.0) < 1e-9, agg.idf1


def test_single_sequence_aggregate_is_the_identity():
    # Pooling one sequence must reproduce that sequence exactly. Catches any
    # aggregate path that drops or rescales a field.
    agg = aggregate([A])
    for name in ("mota", "motp", "hota", "recall", "precision", "num_matches",
                 "num_objects", "num_misses", "num_false_positives",
                 "num_frames"):
        got = getattr(agg, name)
        want = getattr(A, name)
        assert abs(got - want) < 1e-9, f"{name}: {got} != {want}"


def test_empty_input_is_a_safe_zero():
    agg = aggregate([])
    assert agg.sequence == "OVERALL"
    assert agg.num_matches == 0.0
    assert agg.mota == 0.0


def test_zero_object_sequences_do_not_divide_by_zero():
    # A sequence with no ground truth at all must not make the pooled rates NaN.
    empty = TrackMetrics(sequence="E", num_objects=0, num_matches=0)
    agg = aggregate([A, empty])
    assert agg.mota == abs(agg.mota), "mota must not be NaN"
    assert agg.recall == abs(agg.recall), "recall must not be NaN"
    assert agg.precision == abs(agg.precision), "precision must not be NaN"


if __name__ == "__main__":
    from _runner import run_tests

    raise SystemExit(run_tests(dict(globals()), "metrics aggregate"))