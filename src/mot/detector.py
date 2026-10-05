"""Per-frame person detection with a pretrained YOLO model.

The assignment explicitly forbids training a detector from scratch, so this
uses Ultralytics' COCO-pretrained YOLO11 and filters to the person class only.

Two practical notes:

* MOT17 footage is large and the person class is small, so the input height
  matters more than model size. Measured by ``scripts/detector_ceiling.py`` on
  MOT17-02 (first 60 frames, 2221 GT boxes, IoU >= 0.5): yolo11x at 1600 reaches
  40.34% recall against 27.65% for yolo11n at 1280. The seven-sequence benchmark
  is 5,316 frames, small enough to afford the large model.
* Detection quality dominates every tracking metric. A better detector makes
  every tracker look better, so the ablation in the write-up holds the
  detector fixed and varies only the tracker.
"""

from __future__ import annotations

import numpy as np

from .appearance import PERSON_CLASS_ID

try:  # torch is only needed for its OOM exception type.
    from torch.cuda import OutOfMemoryError as _OUT_OF_MEMORY
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    _OUT_OF_MEMORY = RuntimeError


class YoloDetector:
    """Thin wrapper around Ultralytics that returns tlwh person boxes."""

    def __init__(self, weights="yolo11x.pt", device=0, conf=0.25, iou=0.5,
                 imgsz=1600, half=True, max_det=100, verbose=False,
                 classes=(PERSON_CLASS_ID,)):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # Ultralytics renamed this flag; `half=True` is deprecated in favour of
        # `quantize`, and still emits a warning on every call, so the new name is
        # preferred when the installed version understands it.
        self.half = half
        self.max_det = max_det
        self.verbose = verbose
        self.classes = list(classes) if classes is not None else None

    def detect(self, image_rgb, conf=None, imgsz=None):
        """Detect in one RGB frame.

        Returns:
            ``(boxes_xyxy, scores, class_ids)`` as numpy arrays. ``boxes_xyxy``
            is ``(N, 4)`` in pixels.
        """
        result = self.model.predict(image_rgb, **self._params(conf, imgsz))[0]
        return _extract(result)

    def detect_batch(self, images, batch_size=8, conf=None, imgsz=None):
        """Detect over a list of frames.

        Ultralytics' ``predict`` has no ``batch`` argument and rejects unknown
        keyword arguments, so batching is expressed by slicing the input and
        issuing one call per slice. This keeps peak GPU memory bounded, which
        matters because MOT17 frames are 1920x1080.
        """
        frames = list(images)
        if not frames:
            return []
        out = []
        pending = frames
        size = max(1, int(batch_size))
        while pending:
            try:
                results = self.model.predict(pending[:size], **self._params(conf, imgsz))
            except _OUT_OF_MEMORY:
                # A chunk can be too large for the GPU at high ``imgsz``; drop
                # back to a single frame rather than losing the detections.
                if size == 1:
                    raise
                size = max(1, size // 2)
                continue
            for result in results:
                out.append(_extract(result))
            pending = pending[size:]
        return out

    def _params(self, conf, imgsz):
        params = dict(
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=self.verbose,
            classes=self.classes,
        )
        # ``quantize`` is a bit-width selector in this Ultralytics release, not
        # a boolean, so fp16 inference is requested explicitly. Inference
        # precision does not affect tracking results, only speed.
        if self.half:
            params["quantize"] = "fp16"
        return params


def _extract(result):
    """Unpack one Ultralytics result into ``(xyxy, scores, class_ids)``."""
    b = getattr(result, "boxes", None)
    if b is None or len(b) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))
    return (b.xyxy.detach().cpu().numpy().astype(np.float32),
            b.conf.detach().cpu().numpy().astype(np.float32),
            b.cls.detach().cpu().numpy().astype(np.int32))


def xyxy_to_tlwh(boxes_xyxy):
    """``[x1, y1, x2, y2]`` -> ``[x, y, w, h]``, clipping to positive size."""
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if b.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.stack([
        b[:, 0],
        b[:, 1],
        np.clip(b[:, 2] - b[:, 0], 1e-3, None),
        np.clip(b[:, 3] - b[:, 1], 1e-3, None),
    ], axis=1)
    return out


def clip_boxes(boxes_tlwh, width, height):
    b = np.asarray(boxes_tlwh, dtype=np.float32).copy()
    if b.shape[0] == 0:
        return b
    b[:, 0] = np.clip(b[:, 0], 0, width - 1)
    b[:, 1] = np.clip(b[:, 1], 0, height - 1)
    b[:, 2] = np.clip(b[:, 2], 1e-3, width - b[:, 0])
    b[:, 3] = np.clip(b[:, 3], 1e-3, height - b[:, 1])
    return b