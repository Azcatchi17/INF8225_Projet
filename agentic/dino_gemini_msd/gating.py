"""Unified gated pipeline: G1 (pancreas ROI) → G2 (filtered DINO) → G3 (decision).

Replaces the agentic loop's first two stages. Built to be stateless and
composable with either the one-shot or the Gemini-refined agent flow."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import config, grounding_dino as gd, medsam
from .state import Box


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    decision: str                       # one of: tumor / no_pancreas / no_box / no_tumor_low_score / no_tumor_vlm_reject / no_tumor_outside_roi
    mask: np.ndarray                    # HxW uint8 final mask (empty when no tumor)
    pancreas_mask: Optional[np.ndarray] = None
    pancreas_bbox: Optional[list[float]] = None
    candidate_boxes: list[dict] = field(default_factory=list)  # [{xyxy, score, in_roi, roi_iou}, ...]
    selected_box: Optional[list[float]] = None
    selected_score: float = 0.0
    vlm_verdict: Optional[dict] = None  # {is_tumor: bool, confidence: float, rationale: str}
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask", None)
        d.pop("pancreas_mask", None)
        return d

    @property
    def is_tumor_detected(self) -> bool:
        return self.decision == "tumor"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _box_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_area(xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = _box_area(a) + _box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _mask_overlap_frac(box: list[float], mask: np.ndarray) -> float:
    """Fraction of box area covered by mask (0..1)."""
    H, W = mask.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    area = crop.size
    if area == 0:
        return 0.0
    return float(crop.sum()) / float(area)


def _clip_box_to_bbox(box: list[float], bbox: list[float], pad: float = 5.0) -> list[float]:
    """Clip box to stay inside pancreas bbox (with a small pad)."""
    bx1, by1, bx2, by2 = bbox
    x1, y1, x2, y2 = box
    return [
        max(x1, bx1 - pad),
        max(y1, by1 - pad),
        min(x2, bx2 + pad),
        min(y2, by2 + pad),
    ]


# ---------------------------------------------------------------------------
# Filtering (G2)
# ---------------------------------------------------------------------------
def filter_boxes_by_pancreas(
    boxes: list[Box],
    pancreas_mask: np.ndarray,
    pancreas_bbox: list[float],
    min_overlap: Optional[float] = None,
    min_center_in_mask: bool = True,
) -> tuple[list[Box], list[dict]]:
    """Keep boxes whose center lies on the pancreas mask AND that overlap
    the pancreas region by at least `min_overlap`.
    Returns (kept_boxes, diag) where diag[i] describes the i-th original box."""
    min_overlap = min_overlap if min_overlap is not None else config.MIN_PANCREAS_OVERLAP

    kept: list[Box] = []
    diag: list[dict] = []
    H, W = pancreas_mask.shape
    for b in boxes:
        cx, cy = _box_center(b.xyxy)
        cxi, cyi = int(round(cx)), int(round(cy))
        center_ok = (0 <= cxi < W and 0 <= cyi < H and bool(pancreas_mask[cyi, cxi]))
        overlap = _mask_overlap_frac(list(b.xyxy), pancreas_mask)
        roi_iou = _bbox_iou(list(b.xyxy), pancreas_bbox)
        keep = overlap >= min_overlap
        if min_center_in_mask:
            keep = keep and center_ok
        diag.append({
            "xyxy": list(b.xyxy),
            "score": float(b.score),
            "center_in_pancreas": center_ok,
            "mask_overlap": overlap,
            "roi_iou": roi_iou,
            "kept": keep,
        })
        if keep:
            kept.append(b)
    return kept, diag


# ---------------------------------------------------------------------------
# Dual-prompt differential scoring (optional G2.5)
# ---------------------------------------------------------------------------
def differential_score(
    boxes_tumor: list[Box],
    boxes_distractor: list[Box],
    alpha: Optional[float] = None,
) -> list[tuple[Box, float]]:
    """Subtract the max score of spatially-overlapping distractor boxes
    (IoU > 0.3) from each tumor box score:

        s_diff = s_tumor - α * max_{iou>0.3}(s_distractor)

    Helps reject boxes that look equally like non-tumor tissue. Returns
    [(box, diff_score), ...] sorted by diff_score desc."""
    alpha = alpha if alpha is not None else config.DUAL_PROMPT_ALPHA
    scored: list[tuple[Box, float]] = []
    for bt in boxes_tumor:
        overlapping = [bd.score for bd in boxes_distractor
                       if _bbox_iou(list(bt.xyxy), list(bd.xyxy)) > 0.3]
        s_neg = max(overlapping) if overlapping else 0.0
        scored.append((bt, float(bt.score - alpha * s_neg)))
    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# VLM verifier (G3)
# ---------------------------------------------------------------------------
def _vlm_verify_crop(
    image_np: np.ndarray,
    box: list[float],
    pad: int = 20,
) -> Optional[dict]:
    """Ask Gemini if a padded crop shows a pancreatic tumor. Returns None
    if Gemini is unavailable so the caller can skip gracefully."""
    try:
        from .models import get_gemma_client
    except Exception:
        return None

    try:
        client = get_gemma_client()
    except Exception:
        return None

    from google.genai import types
    from .gemma import _downscale_png  # private import; stable enough for internal use
    from PIL import Image
    import json
    import re

    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_img = Image.fromarray(image_np[y1:y2, x1:x2])
    crop_png = _downscale_png(crop_img, 384)
    full_png = _downscale_png(Image.fromarray(image_np), 512)

    system = (
        "You are a radiologist assistant reviewing a candidate pancreatic tumor region on CT. "
        "Look at the full image AND the cropped candidate region. "
        "Pancreatic tumors are hypodense (darker gray) masses within or adjacent to the pancreatic parenchyma. "
        "Non-tumor candidates include: normal pancreas, bowel gas (pitch black), blood vessels, adjacent organs. "
        "Return STRICT JSON: {\"is_tumor\": true|false, \"confidence\": 0..1, \"rationale\": <short>}."
    )
    parts = [
        types.Part.from_bytes(data=full_png, mime_type="image/png"),
        types.Part.from_bytes(data=crop_png, mime_type="image/png"),
        types.Part.from_text(text="Is the cropped region a pancreatic tumor?"),
    ]
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.0,
    )
    try:
        raw = client._retry(  # type: ignore[attr-defined]
            lambda: client._client.models.generate_content(  # type: ignore[attr-defined]
                model=client._model_id, contents=parts, config=cfg,  # type: ignore[attr-defined]
            )
        )
    except Exception as exc:
        return {"is_tumor": True, "confidence": 0.5, "rationale": f"vlm_error:{type(exc).__name__}"}

    text = getattr(raw, "text", None)
    if text is None and hasattr(raw, "candidates") and raw.candidates:
        try:
            text = raw.candidates[0].content.parts[0].text  # type: ignore[index]
        except Exception:
            text = None
    if not text:
        return {"is_tumor": True, "confidence": 0.5, "rationale": "vlm_empty"}
    # Best-effort parse
    try:
        t = text.strip().strip("`")
        if t.startswith("json"):
            t = t[4:]
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        data = json.loads(m.group(0) if m else t)
        return {
            "is_tumor": bool(data.get("is_tumor", True)),
            "confidence": float(data.get("confidence", 0.5)),
            "rationale": str(data.get("rationale", "")),
        }
    except Exception:
        return {"is_tumor": True, "confidence": 0.5, "rationale": "vlm_parse_fail"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def infer_gated(
    image_path: str,
    text: str = config.DEFAULT_DINO_TEXT,
    use_pancreas_roi: bool = True,
    use_vlm_verify: bool = True,
    use_dual_prompt: bool = False,
    score_threshold: Optional[float] = None,
    distractor_prompts: Optional[list[str]] = None,
    pancreas_ckpt: Optional[str] = None,
    clip_mask_to_roi: bool = True,
) -> GateResult:
    """Full gated inference for a single image path.

    - G1 (pancreas_roi=True): load image, segment pancreas, build ROI mask+bbox.
    - G2:                     run DINO, filter candidates by ROI overlap.
    - G2.5 (dual_prompt):     optional differential scoring against distractor prompts.
    - G3:                     reject low-score candidates; optional VLM crop verification.
    - Segmentation:           MedSAM with box clipped to ROI, optional mask clipping.
    """
    from . import pancreas_roi  # lazy — avoids torch import until actually needed

    threshold = score_threshold if score_threshold is not None else config.TUMOR_DIFF_THRESHOLD
    image_np = medsam.load_image(image_path)
    H, W = image_np.shape[:2]
    empty_mask = np.zeros((H, W), dtype=np.uint8)

    pancreas_mask: Optional[np.ndarray] = None
    pancreas_bbox: Optional[list[float]] = None
    if use_pancreas_roi:
        pancreas_mask, pancreas_bbox = pancreas_roi.infer_pancreas(
            image_np,
            dilation_px=config.PANCREAS_DILATION_PX,
            ckpt_path=pancreas_ckpt,
        )
        if pancreas_mask is None or pancreas_mask.sum() < config.MIN_PANCREAS_AREA_PX:
            return GateResult(
                decision="no_pancreas",
                mask=empty_mask,
                pancreas_mask=pancreas_mask,
                pancreas_bbox=pancreas_bbox,
                debug={"pancreas_area_px": int(pancreas_mask.sum()) if pancreas_mask is not None else 0},
            )

    # G2 — DINO
    boxes = gd.detect(image_path, text)
    if not boxes:
        return GateResult(
            decision="no_box",
            mask=empty_mask,
            pancreas_mask=pancreas_mask,
            pancreas_bbox=pancreas_bbox,
            debug={"reason": "dino_returned_nothing"},
        )

    # G2.5 — dual-prompt differential
    if use_dual_prompt:
        distractors = distractor_prompts or config.DUAL_PROMPT_DISTRACTORS
        distractor_boxes: list[Box] = []
        for dp in distractors:
            distractor_boxes.extend(gd.detect(image_path, dp, score_threshold=0.1, top_k=5))
        scored = differential_score(boxes, distractor_boxes)
        boxes = [b for b, s in scored if s > 0]  # drop boxes killed by distractors
        box_diff_scores = {id(b): s for b, s in scored}
    else:
        box_diff_scores = {id(b): b.score for b in boxes}

    # G2 — pancreas filter
    diag: list[dict] = []
    if use_pancreas_roi and pancreas_mask is not None and pancreas_bbox is not None:
        kept, diag = filter_boxes_by_pancreas(boxes, pancreas_mask, pancreas_bbox)
        if not kept:
            return GateResult(
                decision="no_tumor_outside_roi",
                mask=empty_mask,
                pancreas_mask=pancreas_mask,
                pancreas_bbox=pancreas_bbox,
                candidate_boxes=diag,
                debug={"reason": "all_boxes_outside_pancreas_roi"},
            )
        boxes = kept
    else:
        diag = [{"xyxy": list(b.xyxy), "score": float(b.score), "kept": True} for b in boxes]

    # Rank by (differential) score, keep top
    boxes.sort(key=lambda b: -box_diff_scores.get(id(b), b.score))
    best_box = boxes[0]
    best_score = box_diff_scores.get(id(best_box), best_box.score)

    # G3 — threshold on calibrated score
    if best_score < threshold:
        return GateResult(
            decision="no_tumor_low_score",
            mask=empty_mask,
            pancreas_mask=pancreas_mask,
            pancreas_bbox=pancreas_bbox,
            candidate_boxes=diag,
            selected_box=list(best_box.xyxy),
            selected_score=float(best_score),
            debug={"threshold": threshold},
        )

    # G3b — VLM verification (only when enabled)
    vlm_verdict = None
    if use_vlm_verify:
        vlm_verdict = _vlm_verify_crop(image_np, list(best_box.xyxy))
        if vlm_verdict is not None and not vlm_verdict.get("is_tumor", True):
            return GateResult(
                decision="no_tumor_vlm_reject",
                mask=empty_mask,
                pancreas_mask=pancreas_mask,
                pancreas_bbox=pancreas_bbox,
                candidate_boxes=diag,
                selected_box=list(best_box.xyxy),
                selected_score=float(best_score),
                vlm_verdict=vlm_verdict,
            )

    # Segment with MedSAM
    box = list(best_box.xyxy)
    if use_pancreas_roi and pancreas_bbox is not None:
        box = _clip_box_to_bbox(box, pancreas_bbox, pad=5.0)
    embed = medsam.encode_image(image_np)
    mask = medsam.segment(embed, H=H, W=W, box=box)
    if clip_mask_to_roi and pancreas_mask is not None:
        mask = (mask.astype(bool) & pancreas_mask.astype(bool)).astype(np.uint8)

    return GateResult(
        decision="tumor",
        mask=mask,
        pancreas_mask=pancreas_mask,
        pancreas_bbox=pancreas_bbox,
        candidate_boxes=diag,
        selected_box=box,
        selected_score=float(best_score),
        vlm_verdict=vlm_verdict,
    )
