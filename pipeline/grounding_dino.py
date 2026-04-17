"""Grounding DINO detection wrapper."""
from __future__ import annotations

import numpy as np
import torch

from . import config
from .models import get_dino_model
from .state import Box


def _normalize_prompt(text: str) -> str:
    """DINO expects the prompt to end with a period."""
    t = text.strip().lower()
    if not t.endswith("."):
        t += "."
    return t


@torch.no_grad()
def detect(
    image_path: str,
    text: str,
    score_threshold: float = config.DINO_SCORE_THRESHOLD,
    top_k: int = config.DINO_TOP_K,
) -> list[Box]:
    from mmdet.apis import inference_detector

    model = get_dino_model()
    prompt = _normalize_prompt(text)

    if config.USE_BF16 and torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = inference_detector(model, image_path, text_prompt=prompt)
    else:
        result = inference_detector(model, image_path, text_prompt=prompt)

    instances = result.pred_instances
    scores = instances.scores.cpu().numpy()
    bboxes = instances.bboxes.cpu().numpy()

    keep = scores > score_threshold
    scores, bboxes = scores[keep], bboxes[keep]
    if len(scores) == 0:
        return []

    order = np.argsort(-scores)[:top_k]
    return [
        Box(xyxy=(float(bboxes[i][0]), float(bboxes[i][1]),
                  float(bboxes[i][2]), float(bboxes[i][3])),
            score=float(scores[i]))
        for i in order
    ]
