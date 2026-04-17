"""Main agentic loop: DINO → MedSAM → Gemma review → correction → repeat."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import config, grounding_dino as gd, medsam, metrics as M
from .actions import apply_action
from .gemma import MaskAction
from .logging_utils import log_run
from .models import get_gemma_client
from .state import AgentState, GemmaAction, IterationResult


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _as_gemma_action(a: MaskAction) -> GemmaAction:
    return GemmaAction(action=a.action, params=a.params, rationale=a.rationale)


def _run_iteration(
    state: AgentState,
    iteration: int,
    box: list[float],
    points: list[list[float]],
    labels: list[int],
    H: int,
    W: int,
    gemma_action: Optional[GemmaAction],
    gt_mask: np.ndarray | None,
) -> IterationResult:
    t0 = time.perf_counter()
    mask = medsam.segment(
        state.image_embed, H=H, W=W,
        box=box, points=points or None, point_labels=labels or None,
    )
    mm = M.mask_metrics(mask, box, gt_mask=gt_mask)
    elapsed = (time.perf_counter() - t0) * 1000.0

    return IterationResult(
        iteration=iteration,
        box_used_xyxy=list(box),
        points_used=[list(p) for p in points],
        point_labels=list(labels),
        metrics=mm,
        gemma_action=gemma_action,
        dice_vs_gt=mm.dice,
        elapsed_ms=elapsed,
        mask=mask,
    )


def run_agent(
    image_path: str,
    user_text: str,
    max_iter: int = config.MAX_ITER,
    gt_mask: np.ndarray | None = None,
    persist: bool = True,
) -> AgentState:
    state = AgentState.new(image_path, user_text)
    state.image_np = medsam.load_image(image_path)
    H, W = state.image_np.shape[:2]

    gemma = get_gemma_client()

    # 1. Gemma refines the DINO query
    ref = gemma.refine_prompt(user_text)
    state.refined_prompt = ref.search_text
    state.synonyms = list(ref.synonyms)

    # 2. DINO proposes candidate boxes (fall back through synonyms if empty)
    state.candidate_boxes = gd.detect(image_path, state.refined_prompt)
    for syn in state.synonyms:
        if state.candidate_boxes:
            break
        state.candidate_boxes = gd.detect(image_path, syn)

    if not state.candidate_boxes:
        state.fail("no_box")
        if persist:
            log_run(state)
        return state

    # 3. MedSAM: encode image ONCE, cache in state
    state.image_embed = medsam.encode_image(state.image_np)

    # 4. Iteration 0 — box-only
    box0 = list(state.candidate_boxes[0].xyxy)
    it0 = _run_iteration(state, 0, box0, [], [], H, W, None, gt_mask)
    state.push_iter(it0)

    # 5. Refinement loop
    for i in range(1, max_iter):
        prev = state.iterations[-1]

        # Short-circuits save Gemma calls on obvious failures
        if prev.metrics.is_empty:
            cx, cy = _box_center(prev.box_used_xyxy)
            action = MaskAction(
                action="add_positive",
                params={"x": cx, "y": cy},
                rationale="empty short-circuit",
            )
        elif prev.metrics.is_oversized:
            action = MaskAction(
                action="shrink_box", params={"scale": 0.7},
                rationale="oversized short-circuit",
            )
        else:
            action = gemma.analyze_mask(
                state.image_np, prev.mask, prev.metrics.to_gemma_dict(),
                state.refined_prompt, i, max_iter,
            )

        if action.action == "stop":
            state.stop_reason = "gemma_stop"
            break

        ga = _as_gemma_action(action)
        box2, pts2, lbl2 = apply_action(
            ga, prev.box_used_xyxy, prev.points_used, prev.point_labels,
            candidates=state.candidate_boxes,
        )
        it = _run_iteration(state, i, box2, pts2, lbl2, H, W, ga, gt_mask)
        state.push_iter(it)

        if M.mask_iou(it.mask, prev.mask) > config.CONVERGENCE_IOU:
            state.stop_reason = "converged"
            break
    else:
        state.stop_reason = state.stop_reason or "max_iter"

    if persist:
        log_run(state)
    return state
