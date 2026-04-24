"""Test-set evaluation with sensitivity / specificity / DICE metrics.

Runs the gated pipeline (and optionally the baseline) across a whole split,
so the notebook can draw ablation plots side-by-side."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from . import config, grounding_dino as gd, medsam, metrics as M
from .gating import GateResult, infer_gated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_tumor_gt(data_root: Path, file_name: str) -> np.ndarray:
    mask_path = data_root / file_name.replace("/images/", "/masks/")
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask == 2).astype(np.uint8)


def _baseline_oneshot(img_path: str, text: str) -> tuple[np.ndarray, float]:
    """DINO → MedSAM top-1 box, no gate, no agent. Mirrors eval.py::_one_shot."""
    image_np = medsam.load_image(img_path)
    H, W = image_np.shape[:2]
    boxes = gd.detect(img_path, text)
    if not boxes:
        return np.zeros((H, W), np.uint8), 0.0
    embed = medsam.encode_image(image_np)
    mask = medsam.segment(embed, H=H, W=W, box=list(boxes[0].xyxy))
    return mask, float(boxes[0].score)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    n: int
    n_tumor: int
    n_no_tumor: int
    # Classification (does the pipeline decide tumor/no-tumor correctly?)
    sensitivity: float  # among tumor-present images, fraction correctly flagged
    specificity: float  # among tumor-absent images, fraction correctly empty
    precision: float
    f1: float
    # Segmentation quality (DICE reported separately per subset)
    dice_tumor_present_mean: float
    dice_all_mean: float  # treating empty predictions on empty GT as DICE=1
    # Confusion counts
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


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


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------
def run_baseline(
    coco_json: str | Path = None,
    text: str = config.DEFAULT_DINO_TEXT,
    n: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, Summary]:
    coco_json = Path(coco_json) if coco_json else config.MSD_TEST_JSON
    data_root = Path(data_root) if data_root else Path(config.MSD_ROOT)
    with open(coco_json) as f:
        meta = json.load(f)
    images = meta["images"][:n] if n is not None else meta["images"]

    rows = []
    for info in tqdm(images, desc="baseline"):
        file_name = info["file_name"]
        img_path = data_root / file_name
        if not img_path.exists():
            continue
        gt = _load_tumor_gt(data_root, file_name)
        has = bool(gt.sum() > 0)
        mask, score = _baseline_oneshot(str(img_path), text)
        pred = bool(mask.sum() > 0)
        rows.append({
            "file_name": file_name,
            "image_id": info.get("id"),
            "has_tumor": has,
            "predicted_tumor": pred,
            "score": score,
            "decision": "tumor" if pred else "empty",
            "dice": M.dice(mask, gt),
        })
    df = pd.DataFrame(rows)
    return df, _summarise(df)


def run_gated(
    coco_json: str | Path = None,
    text: str = config.DEFAULT_DINO_TEXT,
    use_pancreas_roi: bool = True,
    use_vlm_verify: bool = False,
    use_dual_prompt: bool = False,
    score_threshold: Optional[float] = None,
    pancreas_ckpt: Optional[str] = None,
    n: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, Summary]:
    coco_json = Path(coco_json) if coco_json else config.MSD_TEST_JSON
    data_root = Path(data_root) if data_root else Path(config.MSD_ROOT)
    with open(coco_json) as f:
        meta = json.load(f)
    images = meta["images"][:n] if n is not None else meta["images"]

    rows = []
    for info in tqdm(images, desc="gated"):
        file_name = info["file_name"]
        img_path = data_root / file_name
        if not img_path.exists():
            continue
        gt = _load_tumor_gt(data_root, file_name)
        has = bool(gt.sum() > 0)
        res: GateResult = infer_gated(
            str(img_path), text=text,
            use_pancreas_roi=use_pancreas_roi,
            use_vlm_verify=use_vlm_verify,
            use_dual_prompt=use_dual_prompt,
            score_threshold=score_threshold,
            pancreas_ckpt=pancreas_ckpt,
        )
        pred = res.is_tumor_detected
        rows.append({
            "file_name": file_name,
            "image_id": info.get("id"),
            "has_tumor": has,
            "predicted_tumor": pred,
            "score": float(res.selected_score),
            "decision": res.decision,
            "dice": M.dice(res.mask, gt),
        })
    df = pd.DataFrame(rows)
    return df, _summarise(df)


# ---------------------------------------------------------------------------
# Convenience: ablation over multiple configurations
# ---------------------------------------------------------------------------
def run_ablation(
    coco_json: str | Path = None,
    score_threshold: Optional[float] = None,
    pancreas_ckpt: Optional[str] = None,
    n: Optional[int] = None,
    include_vlm: bool = False,
    save_dir: Optional[Path] = None,
) -> dict[str, tuple[pd.DataFrame, Summary]]:
    """Run {baseline, +G1, +G1+G3, [+G1+G3+VLM]} on the same images.

    Returns {name: (per-image df, summary)}."""
    results: dict[str, tuple[pd.DataFrame, Summary]] = {}

    results["baseline"] = run_baseline(coco_json=coco_json, n=n)
    results["+G1 pancreas ROI"] = run_gated(
        coco_json=coco_json, n=n,
        use_pancreas_roi=True, use_vlm_verify=False, use_dual_prompt=False,
        score_threshold=0.0, pancreas_ckpt=pancreas_ckpt,  # threshold=0 isolates ROI effect
    )
    results["+G1+G3 threshold"] = run_gated(
        coco_json=coco_json, n=n,
        use_pancreas_roi=True, use_vlm_verify=False, use_dual_prompt=False,
        score_threshold=score_threshold, pancreas_ckpt=pancreas_ckpt,
    )
    if include_vlm:
        results["+G1+G3+VLM"] = run_gated(
            coco_json=coco_json, n=n,
            use_pancreas_roi=True, use_vlm_verify=True, use_dual_prompt=False,
            score_threshold=score_threshold, pancreas_ckpt=pancreas_ckpt,
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        for name, (df, summ) in results.items():
            safe = name.replace(" ", "_").replace("+", "p")
            df.to_csv(save_dir / f"ablation_{ts}_{safe}.csv", index=False)
        summary_table = {
            name: summ.to_dict() for name, (_, summ) in results.items()
        }
        with open(save_dir / f"ablation_{ts}_summary.json", "w") as f:
            json.dump(summary_table, f, indent=2)

    return results
