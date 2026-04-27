"""Calibrate the tumor-presence threshold on the val split.

Uses the pancreas-gated DINO score as the decision variable. We need a
per-image binary label 'has tumor' (area(GT mask==2) > 0) and the gated
best box score. A simple threshold sweep picks τ that maximises either
F1 or Youden's J."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from . import config, grounding_dino as gd, medsam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_mask(data_root: Path, file_name: str) -> np.ndarray:
    mask_path = data_root / file_name.replace("/images/", "/masks/")
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _tumor_present(mask: np.ndarray, min_px: int = 25) -> bool:
    return int((mask == 2).sum()) >= min_px


def _gated_best_score(
    image_path: str,
    text: str,
    use_pancreas_roi: bool,
    pancreas_ckpt: Optional[str],
) -> tuple[float, Optional[list[float]]]:
    """Returns (best_score, box) among DINO candidates that pass the pancreas
    ROI filter. Score is 0.0 if no surviving candidate."""
    from . import pancreas_roi
    from .gating import filter_boxes_by_pancreas

    image_np = medsam.load_image(image_path)
    H, W = image_np.shape[:2]

    pancreas_mask = None
    pancreas_bbox = None
    if use_pancreas_roi:
        pancreas_mask, pancreas_bbox = pancreas_roi.infer_pancreas(
            image_np,
            dilation_px=config.PANCREAS_DILATION_PX,
            ckpt_path=pancreas_ckpt,
        )
        if pancreas_mask is None or pancreas_mask.sum() < config.MIN_PANCREAS_AREA_PX:
            return 0.0, None

    boxes = gd.detect(image_path, text, score_threshold=0.01, top_k=10)
    if not boxes:
        return 0.0, None

    if use_pancreas_roi and pancreas_mask is not None and pancreas_bbox is not None:
        boxes, _ = filter_boxes_by_pancreas(boxes, pancreas_mask, pancreas_bbox)
    if not boxes:
        return 0.0, None
    boxes.sort(key=lambda b: -b.score)
    return float(boxes[0].score), list(boxes[0].xyxy)


# ---------------------------------------------------------------------------
# Scoring pass over val
# ---------------------------------------------------------------------------
@dataclass
class CalibResult:
    threshold: float
    f1: float
    sensitivity: float  # TPR, recall
    specificity: float  # TNR
    youden_j: float
    tp: int
    fp: int
    tn: int
    fn: int
    thresholds: list[float] = field(default_factory=list)
    f1_curve: list[float] = field(default_factory=list)
    sens_curve: list[float] = field(default_factory=list)
    spec_curve: list[float] = field(default_factory=list)
    scores_pos: list[float] = field(default_factory=list)
    scores_neg: list[float] = field(default_factory=list)
    per_image: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def collect_scores(
    val_json: str | Path = None,
    text: str = config.DEFAULT_DINO_TEXT,
    use_pancreas_roi: bool = True,
    pancreas_ckpt: Optional[str] = None,
    n: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> list[dict]:
    """Run pancreas-gated DINO on val set, record score + tumor-presence label."""
    val_json = Path(val_json) if val_json else config.MSD_ROOT / "val.json"
    data_root = Path(data_root) if data_root else Path(config.MSD_ROOT)

    with open(val_json) as f:
        meta = json.load(f)
    images = meta["images"]
    if n is not None:
        images = images[:n]

    rows = []
    for info in tqdm(images, desc="calibrate"):
        file_name = info["file_name"]
        img_path = data_root / file_name
        if not img_path.exists():
            continue
        gt = _load_mask(data_root, file_name)
        has_tumor = _tumor_present(gt)
        score, box = _gated_best_score(
            str(img_path), text, use_pancreas_roi, pancreas_ckpt,
        )
        rows.append({
            "file_name": file_name,
            "image_id": info.get("id"),
            "has_tumor": has_tumor,
            "score": score,
            "box": box,
        })
    return rows


def sweep_thresholds(rows: list[dict], n_thresholds: int = 50) -> CalibResult:
    """Grid-search τ ∈ [0, max_score], pick argmax of F1 (fallback Youden's J)."""
    scores_pos = np.array([r["score"] for r in rows if r["has_tumor"]], dtype=np.float32)
    scores_neg = np.array([r["score"] for r in rows if not r["has_tumor"]], dtype=np.float32)

    # Cover the full range of observed scores.
    all_scores = np.concatenate([scores_pos, scores_neg]) if (len(scores_pos) + len(scores_neg)) else np.array([0.0])
    smax = float(all_scores.max()) if all_scores.size else 1.0
    thresholds = np.linspace(0.0, max(smax, 1e-3) * 1.001, n_thresholds).tolist()

    f1_curve, sens_curve, spec_curve = [], [], []
    best = CalibResult(threshold=0.0, f1=0.0, sensitivity=0.0, specificity=0.0,
                       youden_j=0.0, tp=0, fp=0, tn=0, fn=0)
    for tau in thresholds:
        tp = int((scores_pos >= tau).sum())
        fn = int((scores_pos < tau).sum())
        fp = int((scores_neg >= tau).sum())
        tn = int((scores_neg < tau).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        j = sens + spec - 1.0
        f1_curve.append(f1)
        sens_curve.append(sens)
        spec_curve.append(spec)
        if f1 > best.f1 or (f1 == best.f1 and j > best.youden_j):
            best = CalibResult(
                threshold=float(tau), f1=f1, sensitivity=sens,
                specificity=spec, youden_j=j,
                tp=tp, fp=fp, tn=tn, fn=fn,
            )

    best.thresholds = thresholds
    best.f1_curve = f1_curve
    best.sens_curve = sens_curve
    best.spec_curve = spec_curve
    best.scores_pos = scores_pos.tolist()
    best.scores_neg = scores_neg.tolist()
    best.per_image = rows
    return best


def calibrate_threshold(
    val_json: str | Path = None,
    text: str = config.DEFAULT_DINO_TEXT,
    use_pancreas_roi: bool = True,
    pancreas_ckpt: Optional[str] = None,
    n: Optional[int] = None,
    n_thresholds: int = 50,
) -> CalibResult:
    rows = collect_scores(val_json, text, use_pancreas_roi, pancreas_ckpt, n=n)
    return sweep_thresholds(rows, n_thresholds=n_thresholds)


def save_calibration(result: CalibResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
