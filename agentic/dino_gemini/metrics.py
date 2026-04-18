"""Runtime mask metrics + DICE for batch evaluation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import EMPTY_AREA_PCT, OVERSIZED_AREA_PCT
from .state import MaskMetrics


def mask_metrics(
    mask: np.ndarray,
    box_xyxy: list[float] | None,
    gt_mask: np.ndarray | None = None,
) -> MaskMetrics:
    m = np.asarray(mask).astype(bool)
    total = m.size
    area = int(m.sum())
    area_pct = area / total if total else 0.0
    dice_val = dice(m, gt_mask) if gt_mask is not None else None

    if area == 0:
        return MaskMetrics(
            area_pct=0.0,
            n_components=0,
            largest_component_pct=0.0,
            compactness=0.0,
            bbox_agreement_iou=0.0,
            is_empty=True,
            is_oversized=False,
            dice=dice_val,
        )

    labelled, n_comp = ndimage.label(m)
    sizes = ndimage.sum(m, labelled, index=range(1, n_comp + 1))
    largest = float(max(sizes)) if len(sizes) else 0.0
    largest_pct = largest / area if area else 0.0

    # compactness = 4πA / P²  (P via 3x3 erosion)
    eroded = ndimage.binary_erosion(m)
    perimeter = int((m & ~eroded).sum())
    compactness = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    bbox_iou = 0.0
    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        H, W = m.shape
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        box_mask = np.zeros_like(m)
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = True
            inter = int((m & box_mask).sum())
            union = int((m | box_mask).sum())
            bbox_iou = inter / union if union else 0.0

    return MaskMetrics(
        area_pct=float(area_pct),
        n_components=int(n_comp),
        largest_component_pct=float(largest_pct),
        compactness=float(min(compactness, 1.0)),
        bbox_agreement_iou=float(bbox_iou),
        is_empty=area_pct < EMPTY_AREA_PCT,
        is_oversized=area_pct > OVERSIZED_AREA_PCT,
        dice=dice_val,
    )


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / s)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / u)
