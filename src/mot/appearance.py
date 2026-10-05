"""Appearance embedding extractor for the re-identification component.

Design note (important, and a deliberate trade-off): the original DeepSORT uses
a ReID CNN (mars-small128) trained on MARS/CUDA-128 on top of the frozen
ImageNet VGG-16. Those weights are not redistributable and the usual mirrors
were not reachable from this environment, so we use an ImageNet-pretrained
backbone from ``torchvision`` as a generic visual descriptor instead.

Consequence: this embedding is *not* trained for person ReID and is
correspondingly weaker on similar-looking pedestrians (which is exactly the
failure mode we are studying). To compensate we concatenate two cheap,
complementary cues:

* a deep feature from a frozen CNN, and
* HSV colour histograms over head/upper-body/torso/lower-body horizontal
  strips, which are strong for "who wears what" and cheap.

The combination is deterministic, offline and dependency-light, and the ablation
in ``results/`` quantifies how much each cue contributes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# COCO person class id in the YOLO family.
PERSON_CLASS_ID = 0


class AppearanceEncoder:
    """Extracts an L2-normalised appearance vector from a person crop."""

    def __init__(self, device="cuda", backbone="mobilenet_v3_small",
                 use_deep=True, use_color=True, color_bins=(8, 4, 4),
                 batch_size=32, crop_ratio=1.0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_deep = use_deep
        self.use_color = use_color
        self.color_bins = color_bins
        self.batch_size = batch_size
        self.crop_ratio = crop_ratio
        self.deep_dim = 0
        self.color_dim = 0

        if use_deep:
            self.model, self.deep_dim = _load_backbone(backbone, self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.transform = _build_transform()

        if use_color:
            h, s, v = color_bins
            # 3 channels x (H*S*V) bins, computed per body strip (4 strips).
            self.color_dim = 4 * h * s * v

    @property
    def dim(self):
        return self.deep_dim + self.color_dim

    @torch.no_grad()
    def encode_batch(self, image, boxes):
        """Encode several crops from one frame.

        Args:
            image: HxWx3 uint8 RGB array (whole frame).
            boxes: ``(N, 4)`` array of ``[x, y, w, h]`` in pixels.

        Returns:
            ``(N, dim)`` float32 array, L2-normalised per row.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        n = boxes.shape[0]
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        parts = []
        if self.use_deep:
            crops = self._make_crops(image, boxes)
            batch = torch.from_numpy(crops).to(self.device, non_blocking=True)
            feats = []
            for start in range(0, n, self.batch_size):
                chunk = batch[start:start + self.batch_size]
                f = self._forward(chunk)
                feats.append(f)
            deep = torch.cat(feats, dim=0).cpu().numpy().astype(np.float32)
            deep = _l2(deep)
            parts.append(deep)

        if self.use_color:
            parts.append(_color_histogram(image, boxes, self.color_bins))

        out = np.concatenate(parts, axis=1).astype(np.float32)
        return _l2(out)

    def _make_crops(self, image, boxes):
        H, W = image.shape[:2]
        out = np.zeros((boxes.shape[0], 3, 224, 128), dtype=np.float32)
        img = torch.from_numpy(image)
        for i, (x, y, w, h) in enumerate(boxes):
            # Slight horizontal context helps distinguish slim/wide silhouettes,
            # but too much background hurts, so the crop is only mildly widened.
            w2 = w * self.crop_ratio
            x1 = int(max(0, x - (w2 - w) / 2))
            y1 = int(max(0, y - (h * 0.1)))
            x2 = int(min(W, x + w + (w2 - w) / 2))
            y2 = int(min(H, y + h * 1.1))
            if x2 <= x1:
                x2 = min(W, x1 + 1)
            if y2 <= y1:
                y2 = min(H, y1 + 1)
            crop = img[y1:y2, x1:x2]
            crop = crop.permute(2, 0, 1).float() / 255.0
            crop = F.interpolate(
                crop.unsqueeze(0), size=(224, 128), mode="bilinear",
                align_corners=False,
            )[0]
            out[i] = crop.numpy()
        return out

    def _forward(self, batch):
        # Normalisation matching the torchvision pretrained weights.
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (batch - mean) / std
        return self.model(x)


def _load_backbone(name, device):
    import torchvision.models as tvm

    weights_enum = {
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet18": "ResNet18_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
    }
    if name not in weights_enum:
        raise ValueError(f"unsupported backbone {name!r}; pick one of {list(weights_enum)}")

    model = getattr(tvm, name)(weights="DEFAULT")
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, torch.nn.Linear):
            dim = cls.in_features
            model.classifier = torch.nn.Identity()
        else:  # Sequential head, first Linear holds the in_features
            for m in cls.modules():
                if isinstance(m, torch.nn.Linear):
                    dim = m.in_features
                    break
            model.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"cannot locate feature dim for backbone {name!r}")
    return model.to(device), int(dim)


def _build_transform():
    """Kept for documentation of the resize geometry actually used in _make_crops."""
    return (224, 128)  # (H, W) — 1:2 aspect matching person ReID conventions


def _color_histogram(image, boxes, bins):
    """HSV histograms over four horizontal body strips."""
    n_strips = 4
    h, s, v = bins
    out = np.zeros((len(boxes), n_strips * h * s * v), dtype=np.float32)
    H, W = image.shape[:2]
    img = image.astype(np.float32) / 255.0

    # HSV conversion via numpy (no extra dependency).
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    idx = mask & (mx == r)
    hue[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    idx = mask & (mx == g)
    hue[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    idx = mask & (mx == b)
    hue[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
    sat = np.where(mx > 1e-6, diff / np.clip(mx, 1e-6, None), 0.0)

    hq = np.clip((hue / 360.0 * h).astype(np.int32), 0, h - 1)
    sq = np.clip((sat * s).astype(np.int32), 0, s - 1)
    vq = np.clip((mx * v).astype(np.int32), 0, v - 1)
    flat = (hq * s * v + sq * v + vq).astype(np.int32)

    for i, (x, y, bw, bh) in enumerate(boxes):
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W, x + bw)), int(min(H, y + bh))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = flat[y1:y2, x1:x2]
        ph = max(1, (y2 - y1) // n_strips)
        for s_i in range(n_strips):
            ys = y1 + s_i * ph
            ye = ys + ph if s_i < n_strips - 1 else y2
            if ye <= ys:
                continue
            strip = patch[ys - y1:ye - y1]
            counts = np.bincount(strip.ravel(), minlength=h * s * v)
            # Per-strip normalisation removes the strong vertical illumination
            # gradient in MOT17's underground-camera footage.
            total = counts.sum()
            if total > 0:
                counts = counts.astype(np.float32) / total
            out[i, s_i * h * s * v:(s_i + 1) * h * s * v] = counts
    return out


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return x / norm