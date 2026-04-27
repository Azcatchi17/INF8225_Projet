"""Main agentic loop: DINO → MedSAM → Gemini review → correction → repeat."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import config, grounding_dino as gd, medsam, metrics as M
from .actions import apply_action, is_action_sane
from .gemini import MaskAction
from .logging_utils import log_run
from .models import get_gemini_client
from .state import AgentState, Box, GeminiAction, IterationResult, MaskMetrics


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _is_trusted(m: MaskMetrics) -> bool:
    """Mask already looks good enough: skip the Gemini call, early-stop.
    Protects cases where Gemini would 'correct' a correct mask."""
    return (
        not m.is_empty
        and not m.is_oversized
        and m.n_components <= config.TRUST_MAX_COMPONENTS
        and m.bbox_agreement_iou >= config.TRUST_BBOX_IOU
        and m.compactness >= config.TRUST_COMPACTNESS
    )


def _as_gemini_action(a: MaskAction) -> GeminiAction:
    return GeminiAction(action=a.action, params=a.params, rationale=a.rationale)


def _run_iteration(
    state: AgentState,
    iteration: int,
    box: list[float],
    points: list[list[float]],
    labels: list[int],
    H: int,
    W: int,
    gemini_action: Optional[GeminiAction],
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
        gemini_action=gemini_action,
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
    use_gating: bool = False,
    gating_kwargs: dict | None = None,
) -> AgentState:
    state = AgentState.new(image_path, user_text)
    state.image_np = medsam.load_image(image_path)
    H, W = state.image_np.shape[:2]

    gemini = get_gemini_client()

    # 1. Gemini refines the DINO query
    ref = gemini.refine_prompt(user_text)
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

    # 2b. Optional pancreas-ROI / threshold / VLM gate — hard-reject obvious
    # no-tumor slices before any MedSAM work. Produces an empty result state
    # on rejection instead of segmenting a hallucinated box.
    if use_gating:
        from .gating import infer_gated
        gk = dict(gating_kwargs or {})
        gk.setdefault("text", state.refined_prompt)
        gate = infer_gated(image_path, **gk)
        if not gate.is_tumor_detected:
            state.stop_reason = f"gated:{gate.decision}"
            state.candidate_boxes = []  # convention: empty when rejected
            # Return an in-memory empty mask iteration so eval code sees DICE=0
            empty_iter = IterationResult(
                iteration=0,
                box_used_xyxy=[],
                points_used=[],
                point_labels=[],
                metrics=M.mask_metrics(np.zeros((H, W), np.uint8), None, gt_mask),
                gemini_action=None,
                dice_vs_gt=(
                    M.dice(np.zeros((H, W), np.uint8), gt_mask) if gt_mask is not None else None
                ),
                elapsed_ms=0.0,
                mask=np.zeros((H, W), np.uint8),
            )
            state.push_iter(empty_iter)
            if persist:
                log_run(state)
            return state
        # Gate survived — promote its selected box to the head of the list
        if gate.selected_box is not None:
            gate_box = tuple(gate.selected_box)
            gate_score = float(gate.selected_score)
            # Rebuild candidate_boxes with the gated box first
            new_candidates = [Box(xyxy=gate_box, score=gate_score)]
            new_candidates.extend(b for b in state.candidate_boxes if tuple(b.xyxy) != gate_box)
            state.candidate_boxes = new_candidates

    # 3. MedSAM: encode image ONCE, cache in state
    state.image_embed = medsam.encode_image(state.image_np)

    # 4. Iteration 0 — ensemble over top-K detector boxes
    box_list = [list(b.xyxy) for b in state.candidate_boxes]
    _, best_idx = medsam.segment_ensemble(
        state.image_embed, H=H, W=W, boxes=box_list,
        top_k=config.ENSEMBLE_TOP_K,
    )
    box0 = box_list[best_idx]
    it0 = _run_iteration(state, 0, box0, [], [], H, W, None, gt_mask)
    state.push_iter(it0)

    # 5. Refinement loop
    for i in range(1, max_iter):
        prev = state.iterations[-1]

        # Trust region — mask already looks good, skip Gemini and stop
        if _is_trusted(prev.metrics):
            state.stop_reason = "trust_region"
            break

        # Short-circuits save Gemini calls on obvious failures
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
            action = gemini.analyze_mask(
                state.image_np, prev.mask, prev.metrics.to_gemini_dict(),
                state.refined_prompt, i, max_iter,
            )

        if action.action == "stop":
            state.stop_reason = "gemini_stop"
            break

        ga = _as_gemini_action(action)
        if not is_action_sane(ga, prev.mask):
            state.stop_reason = "rejected_add_positive"
            break

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
