"""Agentic loop — Gemini-only: Gemini detects boxes AND reviews masks.

Identical to agentic.dino_gemini.agent except the box-proposal stage uses
`GemmaClient.detect_boxes` instead of Grounding DINO.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from agentic.dino_gemini import medsam, metrics as M
from agentic.dino_gemini.actions import apply_action, is_action_sane
from agentic.dino_gemini.gemma import MaskAction
from agentic.dino_gemini.models import get_gemma_client
from agentic.dino_gemini.state import AgentState, GemmaAction, IterationResult, MaskMetrics

from . import config
from .logging_utils import log_run


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _is_trusted(m: MaskMetrics) -> bool:
    return (
        not m.is_empty
        and not m.is_oversized
        and m.n_components <= config.TRUST_MAX_COMPONENTS
        and m.bbox_agreement_iou >= config.TRUST_BBOX_IOU
        and m.compactness >= config.TRUST_COMPACTNESS
    )


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

    print("[agent] refine_prompt...", flush=True)
    ref = gemma.refine_prompt(user_text)
    state.refined_prompt = ref.search_text
    state.synonyms = list(ref.synonyms)
    print(f"[agent] refined='{ref.search_text}' synonyms={list(ref.synonyms)}", flush=True)

    print("[agent] detect_boxes...", flush=True)
    state.candidate_boxes = gemma.detect_boxes(
        image_path, state.refined_prompt,
        score_threshold=config.VLM_SCORE_THRESHOLD,
        top_k=config.VLM_TOP_K,
    )
    for syn in state.synonyms:
        if state.candidate_boxes:
            break
        print(f"[agent] detect_boxes fallback syn='{syn}'...", flush=True)
        state.candidate_boxes = gemma.detect_boxes(
            image_path, syn,
            score_threshold=config.VLM_SCORE_THRESHOLD,
            top_k=config.VLM_TOP_K,
        )
    print(f"[agent] {len(state.candidate_boxes)} candidate box(es)", flush=True)

    if not state.candidate_boxes:
        state.fail("no_box")
        if persist:
            log_run(state)
        return state

    print("[agent] encode_image...", flush=True)
    state.image_embed = medsam.encode_image(state.image_np)

    box_list = [list(b.xyxy) for b in state.candidate_boxes]
    _, best_idx = medsam.segment_ensemble(
        state.image_embed, H=H, W=W, boxes=box_list,
        top_k=config.ENSEMBLE_TOP_K,
    )
    box0 = box_list[best_idx]
    it0 = _run_iteration(state, 0, box0, [], [], H, W, None, gt_mask)
    state.push_iter(it0)
    print(f"[agent] iter 0 done (box_idx={best_idx} dice={it0.dice_vs_gt})", flush=True)

    # 5. Refinement loop
    for i in range(1, max_iter):
        prev = state.iterations[-1]

        if _is_trusted(prev.metrics):
            state.stop_reason = "trust_region"
            break

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
            print(f"[agent] iter {i}: analyze_mask...", flush=True)
            action = gemma.analyze_mask(
                state.image_np, prev.mask, prev.metrics.to_gemma_dict(),
                state.refined_prompt, i, max_iter,
            )

        if action.action == "stop":
            state.stop_reason = "gemma_stop"
            break

        ga = _as_gemma_action(action)
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
