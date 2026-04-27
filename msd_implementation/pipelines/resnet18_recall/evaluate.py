# script_4_test_final_recall.py
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

from colab.drive_paths import output_dir
from MedSAM.MedSAM_Inference import medsam_inference
from MedSAM.segment_anything import sam_model_registry

from msd_implementation.pipelines.common.proposal_strategy import (
    ProposalConfig,
    candidate_score,
    ensure_3c,
    extract_dino_candidates,
    get_resnet_checkpoint_dir,
    image_score,
    score_candidates_with_resnet,
    select_positive_candidates,
)


METRICS_DIR = output_dir("msd_implementation", "resnet18_recall", "metrics")
OUT_CSV = METRICS_DIR / "dice_final_report_resnet18_recall.csv"
THRESH_JSON = METRICS_DIR / "optimal_threshold_resnet18.json"
THRESH_TXT = METRICS_DIR / "optimal_threshold_resnet18.txt"


def calculate_dice(mask_true, mask_pred):
    m_true = np.asarray(mask_true).astype(bool)
    m_pred = np.asarray(mask_pred).astype(bool)
    if m_true.sum() + m_pred.sum() == 0:
        return 1.0
    return 2 * np.logical_and(m_true, m_pred).sum() / (m_true.sum() + m_pred.sum())


def load_threshold_and_config():
    threshold_override = os.environ.get("RESNET_THRESHOLD_OVERRIDE")

    if THRESH_JSON.exists():
        with open(THRESH_JSON, "r") as f:
            data = json.load(f)
        threshold = float(data["threshold"])
        cfg = ProposalConfig(**data.get("proposal_config", {}))
        print(f"Seuil charge depuis {THRESH_JSON} : {threshold:.2f}")
        if threshold_override is not None:
            threshold = float(threshold_override)
            print(f"Override RESNET_THRESHOLD_OVERRIDE : {threshold:.2f}")
        return threshold, cfg

    if THRESH_TXT.exists():
        with open(THRESH_TXT, "r") as f:
            threshold = float(f.read().strip())
        print(f"Seuil charge depuis {THRESH_TXT} : {threshold:.2f}")
        if threshold_override is not None:
            threshold = float(threshold_override)
            print(f"Override RESNET_THRESHOLD_OVERRIDE : {threshold:.2f}")
        return threshold, ProposalConfig()

    threshold = float(threshold_override) if threshold_override is not None else 0.25
    print(f"Aucun seuil calibre trouve. Seuil force a {threshold:.2f}.")
    return threshold, ProposalConfig()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()
OPTIMAL_THRESH, PROPOSAL_CFG = load_threshold_and_config()

# --- Modeles ---
dino_config = "msd_implementation/configs/grounding_dino/pancreas_tumor.py"
dino_checkpoint = "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth"
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

ensemble_models = []
checkpoint_dir = get_resnet_checkpoint_dir("resnet18_recall")
print(f"Chargement des checkpoints ResNet depuis : {checkpoint_dir.resolve()}")
for i in range(1, 6):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(checkpoint_dir / f"resnet18_recall_fold_{i}.pth", map_location=device))
    model = model.to(device).eval()
    ensemble_models.append(model)

crop_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

medsam_model = sam_model_registry["vit_b"](
    checkpoint="MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
).to(device).eval()

# --- Donnees ---
base_dir = "data/MSD_pancreas"
with open(os.path.join(base_dir, "test.json"), "r") as f:
    test_images_list = json.load(f)["images"]

results_list = []
print(f"\nDebut de l'evaluation finale recall sur {len(test_images_list)} images...")

for img_info in tqdm(test_images_list, desc="Inference Test"):
    file_name = img_info["file_name"]
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_3c = ensure_3c(io.imread(img_path))
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
        candidates = score_candidates_with_resnet(
            img_3c,
            candidates,
            ensemble_models,
            crop_transform,
            device,
            config=PROPOSAL_CFG,
        )
        selected = select_positive_candidates(candidates, OPTIMAL_THRESH, config=PROPOSAL_CFG)

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
                box_1024 = np.array([cand["box"]]) / np.array([W, H, W, H]) * 1024
                full_medsam_seg[medsam_inference(medsam_model, image_embedding, box_1024, H, W) > 0] = 1

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


# --- Bilan ---
df = pd.DataFrame(results_list)
df.to_csv(OUT_CSV, index=False)

df_tumor = df[df["has_tumor"] == True]
df_no_tumor = df[df["has_tumor"] == False]

print("\n" + "=" * 50)
print(f"RESULTATS FINAUX RECALL (Seuil = {OPTIMAL_THRESH:.2f})")
print("=" * 50)
print(f"DICE MOYEN GLOBAL : {df['final_dice'].mean():.4f}")
print(f"DICE MEDIAN GLOBAL : {df['final_dice'].median():.4f}")

if not df_tumor.empty:
    print("\n--- SCANS AVEC TUMEUR ---")
    print(f"DICE MOYEN : {df_tumor['final_dice'].mean():.4f}")
    print(f"DICE MEDIAN : {df_tumor['final_dice'].median():.4f}")
    df_fn = df_tumor[df_tumor["final_dice"] == 0.0]
    fn_count = len(df_fn)
    print(f"FAUX NEGATIFS TOTAUX : {fn_count} / {len(df_tumor)} ({(fn_count / len(df_tumor)) * 100:.1f}%)")
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
