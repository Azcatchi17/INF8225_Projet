"""Convert Gemma action dicts into concrete (box, points, labels) updates."""
from __future__ import annotations

from typing import Literal

from .state import Box, GemmaAction

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


def apply_action(
    action: GemmaAction,
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
        points.append([float(p.get("x", (box[0] + box[2]) / 2)),
                       float(p.get("y", (box[1] + box[3]) / 2))])
        labels.append(1)
    elif name == "add_negative":
        points.append([float(p.get("x", 0.0)), float(p.get("y", 0.0))])
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
