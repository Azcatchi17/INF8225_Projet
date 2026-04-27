# script_3_calibrate_threshold.py
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io
from torchvision import transforms
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from colab.drive_paths import output_dir
from msd_implementation.pipelines.common.proposal_strategy import (
    ProposalConfig,
    ensure_3c,
    extract_dino_candidates,
    find_best_threshold,
    get_resnet_checkpoint_dir,
    image_score,
    score_candidates_with_resnet,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()

# Strategie recall: DINO propose plus large, ResNet decide ensuite.
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
METRICS_DIR = output_dir("msd_implementation", "resnet18_recall", "metrics")
OUT_CSV = METRICS_DIR / "calibration_threshold_multi_candidate.csv"
OUT_SWEEP = METRICS_DIR / "threshold_sweep_multi_candidate.csv"
OUT_TXT = METRICS_DIR / "optimal_threshold_resnet18.txt"
OUT_JSON = METRICS_DIR / "optimal_threshold_resnet18.json"

# --- Modeles ---
dino_config = "msd_implementation/configs/grounding_dino/pancreas_tumor.py"
dino_checkpoint = "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth"
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

ensemble_models = []
checkpoint_dir = get_resnet_checkpoint_dir("resnet18_recall")
print(f"Chargement des checkpoints ResNet depuis : {checkpoint_dir.resolve()}")
print("Chargement de l'ensemble ResNet-18 (5 modeles)...")
for i in range(1, 6):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model.load_state_dict(torch.load(checkpoint_dir / f"resnet18_recall_fold_{i}.pth", map_location=device))
    model = model.to(device)
    model.eval()
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

print(f"\nDebut de la calibration multi-candidats sur {len(val_data['images'])} images de validation...")
rows = []

for img_info in tqdm(val_data["images"], desc="Calibration Val Set"):
    file_name = img_info["file_name"]
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_3c = ensure_3c(io.imread(img_path))
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
        candidates = score_candidates_with_resnet(
            img_3c,
            candidates,
            ensemble_models,
            crop_transform,
            device,
            config=PROPOSAL_CFG,
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
    [
        {k: v for k, v in row.items() if k != "candidates"}
        for row in rows
    ]
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
print("RECHERCHE DU SEUIL OPTIMAL MULTI-CANDIDATS")
print("=" * 58)
print(f"Budget FP validation : <= {MAX_FP_RATE:.0%} des scans sans tumeur")
print(f"Objectif recall validation : >= {MIN_RECALL:.0%} si possible")
print(f"-> Seuil optimal : {optimal_thresh:.2f}")
print(
    "-> F2: {f_beta:.4f} | Recall: {recall:.2f} | Precision: {precision:.2f} | Specificity: {specificity:.2f}".format(
        **best_metrics
    )
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
        },
        f,
        indent=2,
    )

print(f"\nSauvegarde : {OUT_TXT}, {OUT_JSON}")
print(f"Details candidats : {OUT_CSV}")
print(f"Sweep seuils : {OUT_SWEEP}")
