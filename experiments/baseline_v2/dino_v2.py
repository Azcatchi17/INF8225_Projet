"""Label-aware Grounding DINO wrapper + pancreas→tumor cascade.

Mirrors test_msd_no_resnet.py (collaborator's reference) while reusing the
cached model loader and bf16 fallback from agentic/dino_gemini_msd/.

`detect_with_labels` returns the `labels` tensor so the caller can split
pancreas (label 0) from tumor (label 1) when the prompt is
"pancreas . tumor .".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from agentic.dino_gemini_msd import config as _base_config
from agentic.dino_gemini_msd.models import get_dino_model

# tumor_config_v3 metainfo: classes=('pancreas', 'tumor')
PANCREAS_LABEL = 0
TUMOR_LABEL = 1

_DINO_BF16_DISABLED = False


@dataclass
class LabeledBox:
    xyxy: tuple[float, float, float, float]
    score: float
    label: int


def _normalize_prompt(text: str) -> str:
    t = text.strip().lower()
    if not t.endswith("."):
        t += "."
    return t


@torch.no_grad()
def detect_with_labels(
    image_path: str,
    text: str = "pancreas . tumor .",
    score_threshold: float = 0.01,
    top_k: int | None = None,
) -> list[LabeledBox]:
    """Run DINO, keep boxes whose score > threshold, return boxes + labels."""
    from mmdet.apis import inference_detector

    global _DINO_BF16_DISABLED
    model = get_dino_model()
    prompt = _normalize_prompt(text)

    use_bf16 = (
        getattr(_base_config, "USE_BF16", True)
        and torch.cuda.is_available()
        and not _DINO_BF16_DISABLED
    )
    if use_bf16:
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = inference_detector(model, image_path, text_prompt=prompt)
        except RuntimeError as exc:
            if "BFloat16" not in str(exc):
                raise
            _DINO_BF16_DISABLED = True
            warnings.warn(
                "DINO CUDA ops do not support bf16; retrying fp32.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = inference_detector(model, image_path, text_prompt=prompt)
    else:
        result = inference_detector(model, image_path, text_prompt=prompt)

    inst = result.pred_instances
    scores = inst.scores.cpu().numpy()
    bboxes = inst.bboxes.cpu().numpy()
    labels = inst.labels.cpu().numpy()

    keep = scores > score_threshold
    scores, bboxes, labels = scores[keep], bboxes[keep], labels[keep]
    if len(scores) == 0:
        return []

    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]
    return [
        LabeledBox(
            xyxy=(
                float(bboxes[i][0]), float(bboxes[i][1]),
                float(bboxes[i][2]), float(bboxes[i][3]),
            ),
            score=float(scores[i]),
            label=int(labels[i]),
        )
        for i in order
    ]


def _split_by_label(
    boxes: list[LabeledBox],
    pancreas_thr: float,
    tumor_thr: float,
) -> tuple[list[LabeledBox], list[LabeledBox]]:
    pancreas = [b for b in boxes if b.label == PANCREAS_LABEL and b.score > pancreas_thr]
    tumor = [b for b in boxes if b.label == TUMOR_LABEL and b.score > tumor_thr]
    return pancreas, tumor


def _inflate_box(
    box: tuple[float, float, float, float], margin: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (max(0.0, x1 - margin), max(0.0, y1 - margin), x2 + margin, y2 + margin)


def _overlap_ratio(
    tumor: tuple[float, float, float, float],
    pancreas: tuple[float, float, float, float],
) -> float:
    tx1, ty1, tx2, ty2 = tumor
    px1, py1, px2, py2 = pancreas
    ix1, iy1 = max(tx1, px1), max(ty1, py1)
    ix2, iy2 = min(tx2, px2), min(ty2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    tumor_area = max((tx2 - tx1) * (ty2 - ty1), 1e-6)
    return inter / tumor_area


def cascade_select_tumor_box(
    image_path: str,
    text: str = "pancreas . tumor .",
    pancreas_thr: float = 0.3,
    tumor_thr: float = 0.01,
    pancreas_margin_px: float = 20.0,
    min_overlap_ratio: float = 0.1,
) -> LabeledBox | None:
    """Replicate test_msd_no_resnet.py cascade.

    1. Run DINO once on the dual prompt.
    2. Split boxes by label using per-class thresholds.
    3. If pancreas + tumor boxes both exist, inflate top pancreas box by
       `pancreas_margin_px`, keep only tumor boxes with overlap >= min_overlap_ratio,
       return the highest-scoring survivor.
    4. If no pancreas box, fall back to the top tumor box.
    5. Otherwise return None (→ empty mask, classified as no-tumor).
    """
    boxes = detect_with_labels(image_path, text=text, score_threshold=min(pancreas_thr, tumor_thr))
    if not boxes:
        return None
    pancreas_boxes, tumor_boxes = _split_by_label(boxes, pancreas_thr, tumor_thr)

    if pancreas_boxes and tumor_boxes:
        best_p = max(pancreas_boxes, key=lambda b: b.score)
        p_inflated = _inflate_box(best_p.xyxy, pancreas_margin_px)
        survivors = [
            t for t in tumor_boxes
            if _overlap_ratio(t.xyxy, p_inflated) >= min_overlap_ratio
        ]
        if survivors:
            return max(survivors, key=lambda b: b.score)
        return None

    if not pancreas_boxes and tumor_boxes:
        return max(tumor_boxes, key=lambda b: b.score)

    return None
