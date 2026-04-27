"""3-slice variant of calibrate_threshold.py.

Loads the ``three_slice_fold_*.pth`` ensemble, scores validation
candidates with 3-slice context, and writes calibrated thresholds to
``optimal_threshold_three_slice_context.{txt,json}``.

Usage (from repo root):
    python -m msd_implementation.pipelines.three_slice_context.calibrate_threshold
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io
from torchvision import transforms
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.drive_paths import output_dir
from msd_implementation.pipelines.common.proposal_strategy import (
    ProposalConfig,
    ensure_3c,
    extract_dino_candidates,
    find_best_threshold,
    get_resnet_checkpoint_dir,
    image_score,
)
from msd_implementation.pipelines.three_slice_context.score import score_candidates_3slice
from msd_implementation.pipelines.three_slice_context.slice_stack import stack_3slice_image


CHECKPOINT_PREFIX = "three_slice_fold"
METRICS_DIR = output_dir("msd_implementation", "three_slice_context", "metrics")
OUT_TXT = METRICS_DIR / "optimal_threshold_three_slice_context.txt"
OUT_JSON = METRICS_DIR / "optimal_threshold_three_slice_context.json"
OUT_CSV = METRICS_DIR / "calibration_threshold_three_slice.csv"
OUT_SWEEP = METRICS_DIR / "threshold_sweep_3slice.csv"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()

PROPOSAL_CFG = ProposalConfig(
    tumor_score_thresh=0.05,
    pancreas_margin=35,
    min_pancreas_overlap=0.05,
    min_box_area=75,
    max_box_area=18000,
    top_k_candidates=5,
    max_masks=2,
)

F_BETA = 2.0
MAX_FP_RATE = 0.25
MIN_RECALL = 0.90

# --- Modeles ---
dino_model = init_detector(
    "work_dirs/tumor_config_v3/tumor_config_v3.py",
    "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth",
    device=device,
)

ensemble_models = []
checkpoint_dir = get_resnet_checkpoint_dir("three_slice_context")
print(f"Chargement des checkpoints ResNet 3-slice depuis : {checkpoint_dir.resolve()}")
for i in range(1, 6):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    ckpt_path = checkpoint_dir / f"{CHECKPOINT_PREFIX}_{i}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint manquant : {ckpt_path}. Lance d'abord le notebook 02 (train 3-slice)."
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

base_dir = "data/MSD_pancreas"
val_json_path = os.path.join(base_dir, "val.json")

with open(val_json_path, "r") as f:
    val_data = json.load(f)

print(
    f"\nDebut de la calibration 3-slice sur {len(val_data['images'])} images de validation..."
)
rows = []

for img_info in tqdm(val_data["images"], desc="Calibration Val Set 3-slice"):
    file_name = img_info["file_name"]
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_3c = ensure_3c(io.imread(img_path))
    stacked = stack_3slice_image(img_path)
    true_seg_raw = io.imread(mask_path)
    has_tumor = (true_seg_raw == 2).sum() > 0

    with torch.no_grad():
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        pred = result.pred_instances
        candidates = extract_dino_candidates(
            pred.scores.cpu().numpy(),
            pred.bboxes.cpu().numpy(),
            pred.labels.cpu().numpy(),
            image_shape=img_3c.shape[:2],
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

    rows.append(
        {
            "file_name": file_name,
            "has_tumor": bool(has_tumor),
            "image_score": image_score(candidates),
            "n_candidates": len(candidates),
            "best_dino_score": float(max([c["dino_score"] for c in candidates], default=0.0)),
            "best_resnet_prob": float(max([c.get("resnet_prob", 0.0) for c in candidates], default=0.0)),
            "best_candidate_score": image_score(candidates),
            "candidates": candidates,
        }
    )

df = pd.DataFrame(
    [{k: v for k, v in row.items() if k != "candidates"} for row in rows]
)
df.to_csv(OUT_CSV, index=False)

optimal_thresh, best_metrics, sweep = find_best_threshold(
    rows,
    beta=F_BETA,
    max_fp_rate=MAX_FP_RATE,
    min_recall=MIN_RECALL,
    n_thresholds=99,
)
pd.DataFrame(sweep).to_csv(OUT_SWEEP, index=False)

print("\n" + "=" * 58)
print("RECHERCHE DU SEUIL OPTIMAL 3-SLICE")
print("=" * 58)
print(f"Budget FP validation : <= {MAX_FP_RATE:.0%} des scans sans tumeur")
print(f"Objectif recall validation : >= {MIN_RECALL:.0%} si possible")
print(f"-> Seuil optimal : {optimal_thresh:.2f}")
print(
    "-> F2: {f_beta:.4f} | Recall: {recall:.2f} | Precision: {precision:.2f} | "
    "Specificity: {specificity:.2f}".format(**best_metrics)
)
print(
    "-> Impact validation : {fp} Faux Positifs | {fn} Faux Negatifs".format(
        **best_metrics
    )
)

print("\nTable courte autour des seuils utiles:")
for metrics in sweep:
    t = metrics["threshold"]
    if abs((t * 100) % 5) < 1e-6:
        print(
            "tau={threshold:.2f} | TP={tp:02d} FP={fp:02d} TN={tn:02d} FN={fn:02d} | "
            "R={recall:.2f} P={precision:.2f} F2={f_beta:.3f}".format(**metrics)
        )

with open(OUT_TXT, "w") as f:
    f.write(f"{optimal_thresh:.2f}")

with open(OUT_JSON, "w") as f:
    json.dump(
        {
            "threshold": optimal_thresh,
            "metrics": best_metrics,
            "proposal_config": PROPOSAL_CFG.to_dict(),
            "f_beta": F_BETA,
            "max_fp_rate": MAX_FP_RATE,
            "min_recall": MIN_RECALL,
            "sweep": sweep,
            "checkpoint_prefix": CHECKPOINT_PREFIX,
        },
        f,
        indent=2,
    )

print(f"\nSauvegarde : {OUT_TXT}, {OUT_JSON}")
print(f"Details candidats : {OUT_CSV}")
print(f"Sweep seuils : {OUT_SWEEP}")
