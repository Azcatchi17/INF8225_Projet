"""3-slice override of `score_candidates_with_resnet`.

The original function in ``msd_recall_strategy`` crops the candidate box
out of the single-slice 3-channel-replicated image. Here we crop the same
box out of the {prev, curr, next} 3-slice stack so the ResNet sees real
spatial context.

The image PATH (not just the array) is required because we need to know
which slice we are scoring, so we can find the neighbours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.msd._shared.proposal_strategy import ProposalConfig, candidate_score
from experiments.msd.three_slice_context.slice_stack import stack_3slice_image, stack_3slice_crop


def score_candidates_3slice(
    image_path: str,
    candidates: list[dict],
    ensemble_models: list,
    crop_transform,
    device: str,
    config: ProposalConfig | None = None,
    pre_stacked: np.ndarray | None = None,
) -> list[dict]:
    """Same contract as `score_candidates_with_resnet` but uses 3-slice context.

    Parameters
    ----------
    image_path : str
        Path to the current slice PNG. Used to locate prev/next slices.
    pre_stacked : optional pre-built (H,W,3) stack
        Pass this if you already built the stack (saves IO when you call
        `score_candidates_3slice` and another function that needs it).
    """
    cfg = config or ProposalConfig()
    if not candidates:
        return []

    stacked = pre_stacked if pre_stacked is not None else stack_3slice_image(image_path)

    tensors = []
    valid_indices = []
    for idx, cand in enumerate(candidates):
        crop = stack_3slice_crop(image_path, cand["box"], cfg.crop_margin) \
            if pre_stacked is None else _crop_from_pre_stacked(pre_stacked, cand["box"], cfg.crop_margin)
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
        scored[cand_idx]["resnet_votes_050"] = int(
            (probs_by_model[:, local_idx] >= 0.50).sum()
        )
        scored[cand_idx]["resnet_score"] = float(
            np.clip(mean_prob - cfg.resnet_std_penalty * std_prob, 0.0, 1.0)
        )

    scored = [c for c in scored if "resnet_prob" in c]
    scored.sort(key=lambda c: (-candidate_score(c), -float(c["dino_score"])))
    return scored


def _crop_from_pre_stacked(stacked: np.ndarray, box, margin: int) -> np.ndarray | None:
    """Crop helper when the 3-slice stack is already built."""
    from experiments.msd._shared.proposal_strategy import clip_box, expand_box

    h, w = stacked.shape[:2]
    box_clipped = clip_box(box, w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in expand_box(box_clipped, margin, w, h)]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = stacked[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
