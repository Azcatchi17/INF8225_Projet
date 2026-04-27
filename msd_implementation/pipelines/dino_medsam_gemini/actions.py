"""Convert Gemini action dicts into concrete (box, points, labels) updates."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .state import Box, GeminiAction

ActionName = Literal[
    "stop", "add_positive", "add_negative",
    "expand_box", "shrink_box", "replace_box",
]

VALID_ACTIONS = {
    "stop", "add_positive", "add_negative",
    "expand_box", "shrink_box", "replace_box",
}


def _scale_box(xyxy: list[float], scale: float) -> list[float]:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _coerce_xy(p: dict, default_x: float, default_y: float) -> tuple[float, float]:
    """Accept {x:.., y:..}, {x:[a,b]}, {point:[a,b]}, {xy:[a,b]}."""
    for key in ("point", "xy", "coords"):
        v = p.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                pass
    x, y = p.get("x"), p.get("y")
    if isinstance(x, (list, tuple)) and len(x) >= 2 and y is None:
        try:
            return float(x[0]), float(x[1])
        except (TypeError, ValueError):
            pass
    try:
        fx = float(x) if x is not None else default_x
    except (TypeError, ValueError):
        fx = default_x
    try:
        fy = float(y) if y is not None else default_y
    except (TypeError, ValueError):
        fy = default_y
    return fx, fy


def apply_action(
    action: GeminiAction,
    prev_box: list[float],
    prev_points: list[list[float]],
    prev_labels: list[int],
    candidates: list[Box],
) -> tuple[list[float], list[list[float]], list[int]]:
    name = action.action
    p = action.params or {}

    box = list(prev_box)
    points = [list(pt) for pt in prev_points]
    labels = list(prev_labels)

    if name == "add_positive":
        cx, cy = _coerce_xy(p, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        points.append([cx, cy])
        labels.append(1)
    elif name == "add_negative":
        cx, cy = _coerce_xy(p, 0.0, 0.0)
        points.append([cx, cy])
        labels.append(0)
    elif name == "expand_box":
        box = _scale_box(box, float(p.get("scale", 1.15)))
    elif name == "shrink_box":
        box = _scale_box(box, float(p.get("scale", 0.85)))
    elif name == "replace_box":
        idx = int(p.get("idx", 1))
        if 0 <= idx < len(candidates):
            box = list(candidates[idx].xyxy)
        # points reset to empty because the target region changed
        points, labels = [], []
    # "stop" is handled by the caller, never reaches here in the normal flow

    return box, points, labels


def is_action_sane(action: GeminiAction, prev_mask: Optional[np.ndarray]) -> bool:
    """Reject destructive add_positive calls that would move the point OUTSIDE
    the current mask while the mask is healthy (non-empty). Gemini sometimes
    returns a point on the image background, which sends MedSAM chasing noise."""
    if action.action != "add_positive":
        return True
    if prev_mask is None or not prev_mask.any():
        return True
    x, y = _coerce_xy(action.params or {}, -1.0, -1.0)
    H, W = prev_mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < W and 0 <= yi < H):
        return False
    return bool(prev_mask[yi, xi])
