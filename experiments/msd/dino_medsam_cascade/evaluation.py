"""Baseline runner using label-aware cascade + MedSAM.

Reads the cached MedSAM wrapper from agentic/ so the image encoder / bf16
path stays identical — only the DINO-side logic changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from agentic.dino_gemini_msd import config as _cfg
from agentic.dino_gemini_msd import medsam
from agentic.dino_gemini_msd import metrics as M

from .cascade_detector import LabeledBox, cascade_select_tumor_box
from .postprocess import pad_box, postprocess_mask as _postprocess_mask


# ---------------------------------------------------------------------------
# Result container (mirrors eval_improved.Summary so plotting code can reuse)
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    n: int
    n_tumor: int
    n_no_tumor: int
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    dice_tumor_present_mean: float
    dice_all_mean: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def _load_tumor_gt(data_root: Path, file_name: str) -> np.ndarray:
    mask_path = data_root / file_name.replace("/images/", "/masks/")
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask == 2).astype(np.uint8)


def _summarise(df: pd.DataFrame) -> Summary:
    has = df["has_tumor"].astype(bool)
    pred = df["predicted_tumor"].astype(bool)
    tp = int((pred & has).sum())
    fn = int((~pred & has).sum())
    fp = int((pred & ~has).sum())
    tn = int((~pred & ~has).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    dice_pos = df[has]["dice"]
    return Summary(
        n=int(len(df)),
        n_tumor=int(has.sum()),
        n_no_tumor=int((~has).sum()),
        sensitivity=sens,
        specificity=spec,
        precision=prec,
        f1=f1,
        dice_tumor_present_mean=float(dice_pos.mean()) if len(dice_pos) else 0.0,
        dice_all_mean=float(df["dice"].mean()) if len(df) else 0.0,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def _segment_with_box(
    image_np: np.ndarray,
    box: LabeledBox,
    pad_frac: float = 0.0,
) -> np.ndarray:
    H, W = image_np.shape[:2]
    embed = medsam.encode_image(image_np)
    box_xyxy = pad_box(box.xyxy, pad_frac, H, W) if pad_frac > 0 else box.xyxy
    # medsam.segment already runs POSTPROCESS_MASK (closing + small-component
    # drop). We additionally apply our P2.3 chain at the caller's option.
    return medsam.segment(embed, H=H, W=W, box=list(box_xyxy))


def run_baseline_v2(
    coco_json: str | Path = None,
    text: str = "pancreas . tumor .",
    pancreas_thr: float = 0.3,
    tumor_thr: float = 0.01,
    pancreas_margin_px: float = 20.0,
    min_overlap_ratio: float = 0.1,
    # P2.2 — box padding before MedSAM
    pad_frac: float = 0.0,
    # P2.3 — extra mask post-processing (on top of medsam.postprocess_mask)
    keep_largest: bool = False,
    min_area_px: int = 0,
    max_area_frac: float = 1.0,
    n: Optional[int] = None,
    data_root: Optional[Path] = None,
    progress_desc: str = "baseline-v2",
) -> tuple[pd.DataFrame, Summary]:
    """DINO cascade (pancreas→tumor) → MedSAM top-1. No gating, no agent.

    P2.2 + P2.3 hooks:
    - pad_frac > 0 grows the tumor box by that fraction before MedSAM
    - keep_largest / min_area_px / max_area_frac run after MedSAM
    """
    coco_json = Path(coco_json) if coco_json else _cfg.MSD_TEST_JSON
    data_root = Path(data_root) if data_root else Path(_cfg.MSD_ROOT)

    with open(coco_json) as f:
        meta = json.load(f)
    images = meta["images"][:n] if n is not None else meta["images"]

    rows = []
    for info in tqdm(images, desc=progress_desc):
        file_name = info["file_name"]
        img_path = data_root / file_name
        if not img_path.exists():
            continue
        gt = _load_tumor_gt(data_root, file_name)
        has = bool(gt.sum() > 0)

        box = cascade_select_tumor_box(
            str(img_path),
            text=text,
            pancreas_thr=pancreas_thr,
            tumor_thr=tumor_thr,
            pancreas_margin_px=pancreas_margin_px,
            min_overlap_ratio=min_overlap_ratio,
        )

        if box is None:
            H, W = gt.shape[:2]
            mask = np.zeros((H, W), np.uint8)
            score = 0.0
            decision = "empty"
        else:
            image_np = medsam.load_image(str(img_path))
            mask = _segment_with_box(image_np, box, pad_frac=pad_frac)
            if keep_largest or min_area_px > 0 or max_area_frac < 1.0:
                mask = _postprocess_mask(
                    mask,
                    keep_largest=keep_largest,
                    min_area_px=min_area_px,
                    max_area_frac=max_area_frac,
                )
            score = box.score
            decision = "tumor"

        pred = bool(mask.sum() > 0)
        rows.append({
            "file_name": file_name,
            "image_id": info.get("id"),
            "has_tumor": has,
            "predicted_tumor": pred,
            "score": score,
            "decision": decision,
            "dice": M.dice(mask, gt),
        })

    df = pd.DataFrame(rows)
    return df, _summarise(df)
