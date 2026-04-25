# script_0_calibrate_dino.py
import json
import os

import numpy as np
import torch
from skimage import io
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector

from msd_recall_strategy import ProposalConfig, box_tumor_overlap, ensure_3c, extract_dino_candidates


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

# --- Chargement DINO ---
dino_model = init_detector(
    "work_dirs/tumor_config_v3/tumor_config_v3.py",
    "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth",
    device=device,
)

# --- Donnees ---
base_dir = "data/MSD_pancreas"
val_json_path = os.path.join(base_dir, "val.json")

with open(val_json_path, "r") as f:
    val_data = json.load(f)

print(f"\nDebut de la calibration DINO sur {len(val_data['images'])} images de validation...")

thresholds_to_test = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
results = {t: {"tp": 0, "fn": 0, "total_fp_boxes": 0, "candidate_boxes": 0} for t in thresholds_to_test}

for img_info in tqdm(val_data["images"], desc="Inference brute"):
    file_name = img_info["file_name"]
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    img_3c = ensure_3c(io.imread(img_path))
    true_seg_raw = io.imread(mask_path)
    has_tumor = (true_seg_raw == 2).sum() > 0
    true_tumor_mask = (true_seg_raw == 2).astype(np.uint8)

    with torch.no_grad():
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        pred = result.pred_instances
        scores = pred.scores.cpu().numpy()
        bboxes = pred.bboxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()

    for thresh in thresholds_to_test:
        cfg = ProposalConfig(tumor_score_thresh=thresh)
        candidates = extract_dino_candidates(
            scores,
            bboxes,
            labels,
            image_shape=img_3c.shape[:2],
            config=cfg,
        )

        tumor_found = False
        fp_boxes_in_image = 0
        for cand in candidates:
            overlap_pixels, _ = box_tumor_overlap(cand["box"], true_tumor_mask)
            if overlap_pixels > 0:
                tumor_found = True
            else:
                fp_boxes_in_image += 1

        results[thresh]["candidate_boxes"] += len(candidates)
        results[thresh]["total_fp_boxes"] += fp_boxes_in_image
        if has_tumor:
            if tumor_found:
                results[thresh]["tp"] += 1
            else:
                results[thresh]["fn"] += 1

print("\n" + "=" * 88)
print(
    f"{'Seuil DINO':<12} | {'Tumeurs trouvees (TP)':<22} | {'Ratees (FN)':<12} | "
    f"{'Candidats':<10} | {'FP boxes':<10}"
)
print("-" * 88)

for thresh in thresholds_to_test:
    tp = results[thresh]["tp"]
    fn = results[thresh]["fn"]
    fp = results[thresh]["total_fp_boxes"]
    cand = results[thresh]["candidate_boxes"]
    print(f"{thresh:<12.3f} | {tp:<22} | {fn:<12} | {cand:<10} | {fp:<10}")
print("=" * 88)

print("\nAnalyse :")
print("1. Choisis le plus haut seuil DINO qui garde les FN DINO au minimum.")
print("2. Si les FP boxes explosent a bas seuil, ce n'est pas bloquant tant que ResNet est re-entraine avec extract_hard_negatives.py.")
print("3. Le seuil runtime recommande pour la strategie recall reste 0.01 avant calibration ResNet.")
