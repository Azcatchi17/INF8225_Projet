"""Shared candidate logic for the MSD pancreas recall-oriented pipeline.

The goal is to keep Grounding DINO as the proposal generator, ResNet as the
post-DINO verifier, and MedSAM as the segmenter, while avoiding the brittle
top-1 decision that turns many recoverable cases into false negatives.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class ProposalConfig:
    pancreas_score_thresh: float = 0.30
    tumor_score_thresh: float = 0.05
    pancreas_margin: int = 35
    min_pancreas_overlap: float = 0.05
    min_box_area: int = 75
    max_box_area: int = 18000
    top_k_candidates: int = 5
    nms_iou: float = 0.50
    crop_margin: int = 8
    max_masks: int = 2
    resnet_std_penalty: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)


def get_resnet_checkpoint_dir() -> Path:
    """Return the persistent directory used for ResNet ensemble artifacts."""
    env_path = os.environ.get("RESNET_CHECKPOINT_DIR") or os.environ.get("INF8225_DRIVE_ROOT")
    if env_path:
        return Path(env_path)

    data_path = Path("data")
    if data_path.is_symlink():
        resolved = data_path.resolve()
        if resolved.name == "data":
            return resolved.parent

    for candidate in (
        Path("/content/drive/MyDrive/Projet_Medsam"),
        Path("/content/drive/MyDrive/INF8225_Projet"),
        Path("/content/drive/MyDrive/INF8225"),
    ):
        if candidate.exists():
            return candidate

    return Path(".")


def ensure_3c(img_np: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 image without changing already multi-channel PNGs."""
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    if img_np.shape[-1] > 3:
        img_np = img_np[..., :3]
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def clip_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def expand_box(box: Iterable[float], margin: float, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return clip_box([x1 - margin, y1 - margin, x2 + margin, y2 + margin], width, height)


def intersection_area(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_tumor_overlap(box: Iterable[float], true_tumor_mask: np.ndarray) -> tuple[int, float]:
    height, width = true_tumor_mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clip_box(box, width, height)]
    if x2 <= x1 or y2 <= y1:
        return 0, 0.0
    crop = true_tumor_mask[y1:y2, x1:x2]
    pixels = int(crop.sum())
    area = max(1, (x2 - x1) * (y2 - y1))
    return pixels, pixels / area


def _nms(candidates: list[dict], max_iou: float) -> list[dict]:
    kept: list[dict] = []
    for cand in sorted(candidates, key=lambda c: -float(c["dino_score"])):
        if all(iou(cand["box"], prev["box"]) < max_iou for prev in kept):
            kept.append(cand)
    return kept


def extract_dino_candidates(
    scores: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple[int, int],
    config: ProposalConfig | None = None,
) -> list[dict]:
    """Extract top-k tumor candidates after a relaxed pancreas spatial gate.

    `labels` is expected to follow the prompt "pancreas . tumor .":
    label 0 = pancreas, label 1 = tumor.
    """
    cfg = config or ProposalConfig()
    height, width = image_shape

    pancreas_mask = (labels == 0) & (scores >= cfg.pancreas_score_thresh)
    tumor_mask = (labels == 1) & (scores >= cfg.tumor_score_thresh)

    pancreas_boxes = bboxes[pancreas_mask]
    pancreas_scores = scores[pancreas_mask]
    tumor_boxes = bboxes[tumor_mask]
    tumor_scores = scores[tumor_mask]

    pancreas_roi = None
    if len(pancreas_boxes) > 0:
        best_idx = int(np.argmax(pancreas_scores))
        pancreas_roi = expand_box(pancreas_boxes[best_idx], cfg.pancreas_margin, width, height)

    candidates: list[dict] = []
    for box_raw, score in zip(tumor_boxes, tumor_scores):
        box = clip_box(box_raw, width, height)
        area = box_area(box)
        if not (cfg.min_box_area <= area <= cfg.max_box_area):
            continue

        overlap = 1.0
        source = "no_pancreas_fallback"
        if pancreas_roi is not None:
            overlap = intersection_area(box, pancreas_roi) / max(area, 1.0)
            source = "pancreas_roi"
            if overlap < cfg.min_pancreas_overlap:
                continue

        candidates.append(
            {
                "box": box,
                "dino_score": float(score),
                "pancreas_overlap": float(overlap),
                "box_area": float(area),
                "source": source,
            }
        )

    candidates = _nms(candidates, cfg.nms_iou)
    return candidates[: cfg.top_k_candidates]


def crop_from_box(img_3c: np.ndarray, box: Iterable[float], margin: int) -> np.ndarray | None:
    height, width = img_3c.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in expand_box(box, margin, width, height)]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_3c[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def score_candidates_with_resnet(
    img_3c: np.ndarray,
    candidates: list[dict],
    ensemble_models: list,
    crop_transform,
    device: str,
    config: ProposalConfig | None = None,
) -> list[dict]:
    """Attach mean ensemble tumor probability to each candidate."""
    import torch

    cfg = config or ProposalConfig()
    if not candidates:
        return []

    tensors = []
    valid_indices = []
    for idx, cand in enumerate(candidates):
        crop = crop_from_box(img_3c, cand["box"], cfg.crop_margin)
        if crop is None:
            continue
        tensors.append(crop_transform(Image.fromarray(crop)).unsqueeze(0))
        valid_indices.append(idx)

    if not tensors:
        return []

    with torch.no_grad():
        batch = torch.cat(tensors, dim=0).to(device)
        model_probs = []
        for model in ensemble_models:
            probs = torch.softmax(model(batch), dim=1)[:, 1]
            model_probs.append(probs.detach().cpu().numpy())

    probs_by_model = np.stack(model_probs, axis=0)
    mean_probs = probs_by_model.mean(axis=0)
    std_probs = probs_by_model.std(axis=0)

    scored = [dict(c) for c in candidates]
    for local_idx, cand_idx in enumerate(valid_indices):
        mean_prob = float(mean_probs[local_idx])
        std_prob = float(std_probs[local_idx])
        scored[cand_idx]["resnet_prob"] = mean_prob
        scored[cand_idx]["resnet_std"] = std_prob
        scored[cand_idx]["resnet_votes_050"] = int((probs_by_model[:, local_idx] >= 0.50).sum())
        scored[cand_idx]["resnet_score"] = float(
            np.clip(mean_prob - cfg.resnet_std_penalty * std_prob, 0.0, 1.0)
        )

    scored = [c for c in scored if "resnet_prob" in c]
    scored.sort(key=lambda c: (-candidate_score(c), -float(c["dino_score"])))
    return scored


def candidate_score(candidate: dict) -> float:
    return float(candidate.get("resnet_score", candidate.get("resnet_prob", 0.0)))


def image_score(candidates: list[dict]) -> float:
    if not candidates:
        return 0.0
    return float(max(candidate_score(c) for c in candidates))


def select_positive_candidates(
    candidates: list[dict],
    threshold: float,
    config: ProposalConfig | None = None,
) -> list[dict]:
    cfg = config or ProposalConfig()
    selected = [c for c in candidates if candidate_score(c) >= threshold]
    selected.sort(key=lambda c: (-candidate_score(c), -float(c["dino_score"])))
    return selected[: cfg.max_masks]


def threshold_metrics(rows: list[dict], threshold: float, beta: float = 2.0) -> dict:
    tp = fn = fp = tn = 0
    for row in rows:
        pred = float(row["image_score"]) >= threshold
        truth = bool(row["has_tumor"])
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    beta2 = beta * beta
    f_beta = (
        (1 + beta2) * precision * recall / ((beta2 * precision) + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f_beta": f_beta,
    }


def find_best_threshold(
    rows: list[dict],
    beta: float = 2.0,
    max_fp_rate: float = 0.25,
    n_thresholds: int = 99,
    min_recall: float = 0.90,
) -> tuple[float, dict, list[dict]]:
    n_neg = sum(1 for row in rows if not row["has_tumor"])
    max_fp = int(np.floor(n_neg * max_fp_rate))
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    sweep = [threshold_metrics(rows, float(t), beta=beta) for t in thresholds]
    feasible = [m for m in sweep if m["fp"] <= max_fp]
    recall_pool = [m for m in feasible if m["recall"] >= min_recall]
    pool = recall_pool or feasible or sweep
    best = max(pool, key=lambda m: (m["recall"], m["f_beta"], -m["fp"], m["precision"]))
    return float(best["threshold"]), best, sweep
