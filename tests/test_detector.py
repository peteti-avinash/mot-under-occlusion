"""Detector tests, focused on the batching path.

The batching loop is the one place where a bug silently corrupts the whole
benchmark: it can drop frames, duplicate them, or reorder them, and every
downstream metric would still be produced without complaint.
"""

from __future__ import annotations

import numpy as np

from _runner import run_tests
from mot.detector import YoloDetector, clip_boxes, xyxy_to_tlwh
from mot import detector as detector_mod


def _fake_detector(predict):
    """Build a YoloDetector without loading real weights.

    ``YoloDetector.__init__`` would instantiate an Ultralytics model, so the
    instance is built empty and given a stub instead.
    """
    det = object.__new__(YoloDetector)
    det.model = type("FakeModel", (), {"predict": staticmethod(predict)})()
    det.device = 0
    det.conf = 0.25
    det.iou = 0.5
    det.imgsz = 64
    det.half = False
    det.max_det = 10
    det.verbose = False
    det.classes = [0]
    return det


def _result(tag, n_boxes=1):
    boxes = np.array([[10.0, 10.0, 30.0, 60.0]] * n_boxes, dtype=np.float32)

    class _Tensor:
        """Stands in for the detached CUDA tensor Ultralytics hands back."""

        def __init__(self, array):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class _Boxes:
        def __init__(self, n):
            self.xyxy = _Tensor(boxes)
            self.conf = _Tensor(np.full(n, 0.9, np.float32))
            self.cls = _Tensor(np.zeros(n, np.int32))

        def __len__(self):
            return n_boxes

    return type("Res", (), {"boxes": _Boxes(n_boxes), "id": tag})()


def test_detect_batch_preserves_frame_order_and_count():
    """Every input frame must come back exactly once, in order."""
    seen = []

    def predict(frames, **kw):
        seen.extend(frames)
        return [_result(f) for f in frames]

    det = _fake_detector(predict)
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(7)]
    out = det.detect_batch(frames, batch_size=3)

    assert len(out) == 7, f"expected 7 results, got {len(out)}"
    assert len(seen) == 7, f"frames passed to the model: {len(seen)}"
    assert [int(f[0, 0, 0]) for f in seen] == list(range(7)), \
        "frames were reordered or duplicated"


def test_detect_batch_halves_batch_on_out_of_memory():
    """A chunk that does not fit must be split, not dropped.

    Regression guard: the retry has to shrink the chunk and retry the *same*
    pending frames, so the output stays complete and ordered even when the
    first attempt raises CUDA OOM.
    """
    attempts = []

    def predict(frames, **kw):
        attempts.append(len(frames))
        if len(frames) > 2:
            raise detector_mod._OUT_OF_MEMORY("CUDA out of memory")
        return [_result(f) for f in frames]

    det = _fake_detector(predict)
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(8)]
    out = det.detect_batch(frames, batch_size=8)

    assert attempts == [8, 4, 2, 2, 2, 2], f"unexpected retry sequence: {attempts}"
    assert len(out) == 8, f"OOM retry lost frames: {len(out)} of 8"


def test_detect_batch_reraises_when_a_single_frame_does_not_fit():
    """Shrinking to one frame cannot help, so the error must surface."""

    def predict(frames, **kw):
        raise detector_mod._OUT_OF_MEMORY("CUDA out of memory")

    det = _fake_detector(predict)
    raised = False
    try:
        det.detect_batch([np.zeros((8, 8, 3), np.uint8)], batch_size=4)
    except detector_mod._OUT_OF_MEMORY:
        raised = True
    assert raised, "a single-frame OOM was swallowed instead of raised"


def test_detect_batch_handles_no_frames():
    det = _fake_detector(lambda frames, **kw: [])
    assert det.detect_batch([], batch_size=4) == []


def test_xyxy_to_tlwh_and_clip():
    tlwh = xyxy_to_tlwh([[10.0, 20.0, 50.0, 100.0]])
    assert np.allclose(tlwh, [[10.0, 20.0, 40.0, 80.0]])

    empty = xyxy_to_tlwh(np.zeros((0, 4)))
    assert empty.shape == (0, 4)

    clipped = clip_boxes([[95.0, 5.0, 40.0, 50.0]], width=100, height=100)
    assert clipped[0, 0] + clipped[0, 2] <= 100 + 1e-6, "box overflows width"
    assert clipped[0, 1] + clipped[0, 3] <= 100 + 1e-6, "box overflows height"
    assert clipped[0, 2] > 0 and clipped[0, 3] > 0, "clip produced a degenerate box"


if __name__ == "__main__":
    raise SystemExit(run_tests(dict(globals()), "detector"))