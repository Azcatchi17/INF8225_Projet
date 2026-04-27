"""3-slice v2 final test: ResNet-50 ensemble + crop_margin=30 + MedSAM.

Reads ``optimal_threshold_resnet50_wide_crop.{txt,json}``, loads the
``resnet50_wide_crop_fold_*.pth`` ensemble, scores test candidates with the
wider 3-slice context, runs MedSAM on the survivors, and writes
``data/results/dice_final_report_resnet50_wide_crop.csv``.

Usage (from repo root):
    python -m experiments.msd.resnet50_wide_crop.evaluate
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io, transform
from torchvision import transforms
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEDSAM_DIR = ROOT / "MedSAM"
if MEDSAM_DIR.is_dir() and str(MEDSAM_DIR) not in sys.path:
    sys.path.insert(0, str(MEDSAM_DIR))

from MedSAM.MedSAM_Inference import medsam_inference
from MedSAM.segment_anything import sam_model_registry

from experiments.msd._shared.proposal_strategy import (
    ProposalConfig,
    candidate_score,
    ensure_3c,
    extract_dino_candidates,
    get_resnet_checkpoint_dir,
    image_score,
    select_positive_candidates,
)
from experiments.msd.three_slice_context.score import score_candidates_3slice
from experiments.msd.three_slice_context.slice_stack import stack_3slice_image


CHECKPOINT_PREFIX = "resnet50_wide_crop_fold"
THRESH_TXT = "optimal_threshold_resnet50_wide_crop.txt"
THRESH_JSON = "optimal_threshold_resnet50_wide_crop.json"
OUT_CSV = "data/results/dice_final_report_resnet50_wide_crop.csv"


def calculate_dice(mask_true, mask_pred):
    m_true = np.asarray(mask_true).astype(bool)
    m_pred = np.asarray(mask_pred).astype(bool)
    if m_true.sum() + m_pred.sum() == 0:
        return 1.0
    return 2 * np.logical_and(m_true, m_pred).sum() / (m_true.sum() + m_pred.sum())


def load_threshold_and_config():
    threshold_override = os.environ.get("RESNET_THRESHOLD_OVERRIDE")

    if os.path.exists(THRESH_JSON):
        with open(THRESH_JSON, "r") as f:
            data = json.load(f)
        threshold = float(data["threshold"])
        cfg = ProposalConfig(**data.get("proposal_config", {}))
        print(f"Seuil 3-slice v2 charge depuis {THRESH_JSON} : {threshold:.2f}")
        if threshold_override is not None:
            threshold = float(threshold_override)
            print(f"Override RESNET_THRESHOLD_OVERRIDE : {threshold:.2f}")
        return threshold, cfg

    if os.path.exists(THRESH_TXT):
        with open(THRESH_TXT, "r") as f:
            threshold = float(f.read().strip())
        print(f"Seuil 3-slice v2 charge depuis {THRESH_TXT} : {threshold:.2f}")
        # Fallback config also bumps crop_margin to match training
        cfg = ProposalConfig(crop_margin=30)
        if threshold_override is not None:
            threshold = float(threshold_override)
            print(f"Override RESNET_THRESHOLD_OVERRIDE : {threshold:.2f}")
        return threshold, cfg

    threshold = float(threshold_override) if threshold_override is not None else 0.25
    print(
        f"Aucun seuil 3-slice v2 calibre trouve. Seuil force a {threshold:.2f}."
    )
    return threshold, ProposalConfig(crop_margin=30)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()
OPTIMAL_THRESH, PROPOSAL_CFG = load_threshold_and_config()

# Defensive: if a stale JSON omitted crop_margin, force it back to 30
if PROPOSAL_CFG.crop_margin != 30:
    print(
        f"Note: crop_margin lu = {PROPOSAL_CFG.crop_margin}, force a 30 pour matcher l'entrainement."
    )
    PROPOSAL_CFG = ProposalConfig(
        **{**PROPOSAL_CFG.to_dict(), "crop_margin": 30}
    )

# --- Modeles ---
dino_model = init_detector(
    "work_dirs/tumor_config_v3/tumor_config_v3.py",
    "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth",
    device=device,
)

ensemble_models = []
checkpoint_dir = get_resnet_checkpoint_dir()
print(f"Chargement des checkpoints ResNet-50 3-slice depuis : {checkpoint_dir.resolve()}")
for i in range(1, 6):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    ckpt_path = checkpoint_dir / f"{CHECKPOINT_PREFIX}_{i}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint manquant : {ckpt_path}. Lance d'abord le notebook 02 v2."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()
    ensemble_models.append(model)

crop_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

medsam_model = (
    sam_model_registry["vit_b"](checkpoint="MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
    .to(device)
    .eval()
)

base_dir = "data/MSD_pancreas"
with open(os.path.join(base_dir, "test.json"), "r") as f:
    test_images_list = json.load(f)["images"]

results_list = []
os.makedirs("data/results", exist_ok=True)
print(
    f"\nDebut de l'evaluation finale 3-slice v2 sur {len(test_images_list)} images..."
)

for img_info in tqdm(test_images_list, desc="Inference Test 3-slice v2"):
    file_name = img_info["file_name"]
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_3c = ensure_3c(io.imread(img_path))
    stacked = stack_3slice_image(img_path)
    H, W = img_3c.shape[:2]

    true_seg_raw = io.imread(mask_path)
    true_seg = (true_seg_raw == 2).astype(np.uint8)
    has_tumor = true_seg.sum() > 0

    full_medsam_seg = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        pred = result.pred_instances
        candidates = extract_dino_candidates(
            pred.scores.cpu().numpy(),
            pred.bboxes.cpu().numpy(),
            pred.labels.cpu().numpy(),
            image_shape=(H, W),
            config=PROPOSAL_CFG,
        )
        candidates = score_candidates_3slice(
            img_path,
            candidates,
            ensemble_models,
            crop_transform,
            device,
            config=PROPOSAL_CFG,
            pre_stacked=stacked,
        )
        selected = select_positive_candidates(
            candidates, OPTIMAL_THRESH, config=PROPOSAL_CFG
        )

        if selected:
            img_1024 = transform.resize(
                img_3c,
                (1024, 1024),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )
            image_embedding = medsam_model.image_encoder(
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )

            for cand in selected:
                box_1024 = (
                    np.array([cand["box"]]) / np.array([W, H, W, H]) * 1024
                )
                full_medsam_seg[
                    medsam_inference(medsam_model, image_embedding, box_1024, H, W) > 0
                ] = 1

    final_dice = calculate_dice(true_seg, full_medsam_seg)
    best_score = image_score(candidates)
    best_prob = max([float(c.get("resnet_prob", 0.0)) for c in candidates], default=0.0)
    best_dino = max([c["dino_score"] for c in candidates], default=0.0)

    fn_cause = ""
    if has_tumor and final_dice == 0.0:
        if not candidates:
            fn_cause = "dino_or_spatial_filters"
        elif best_score < OPTIMAL_THRESH:
            fn_cause = "resnet_threshold"
        else:
            fn_cause = "medsam_or_wrong_box"

    results_list.append(
        {
            "file_name": file_name,
            "has_tumor": bool(has_tumor),
            "final_dice": final_dice,
            "n_candidates": len(candidates),
            "n_selected": len(selected),
            "best_candidate_score": best_score,
            "best_resnet_prob": best_prob,
            "best_dino_score": best_dino,
            "selected_scores": [candidate_score(c) for c in selected],
            "fn_cause": fn_cause,
        }
    )


df = pd.DataFrame(results_list)
df.to_csv(OUT_CSV, index=False)

df_tumor = df[df["has_tumor"] == True]
df_no_tumor = df[df["has_tumor"] == False]

print("\n" + "=" * 50)
print(f"RESULTATS FINAUX 3-SLICE v2 (Seuil = {OPTIMAL_THRESH:.2f})")
print("=" * 50)
print(f"DICE MOYEN GLOBAL : {df['final_dice'].mean():.4f}")
print(f"DICE MEDIAN GLOBAL : {df['final_dice'].median():.4f}")

if not df_tumor.empty:
    print("\n--- SCANS AVEC TUMEUR ---")
    print(f"DICE MOYEN : {df_tumor['final_dice'].mean():.4f}")
    print(f"DICE MEDIAN : {df_tumor['final_dice'].median():.4f}")
    df_fn = df_tumor[df_tumor["final_dice"] == 0.0]
    fn_count = len(df_fn)
    print(
        f"FAUX NEGATIFS TOTAUX : {fn_count} / {len(df_tumor)} "
        f"({(fn_count / len(df_tumor)) * 100:.1f}%)"
    )
    if fn_count > 0:
        print("\nCauses FN :")
        print(df_fn["fn_cause"].value_counts().to_string())

if not df_no_tumor.empty:
    print("\n--- SCANS SANS TUMEUR ---")
    print(f"DICE MOYEN : {df_no_tumor['final_dice'].mean():.4f}")
    fp_count = len(df_no_tumor[df_no_tumor["final_dice"] == 0.0])
    print(f"Faux Positifs : {fp_count} / {len(df_no_tumor)}")

print(f"\nCSV : {OUT_CSV}")
print("=" * 50)
