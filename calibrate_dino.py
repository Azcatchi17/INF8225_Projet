# script_0_calibrate_dino.py
import os
import json
import numpy as np
import pandas as pd
import torch
from skimage import io
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

from colab.drive_paths import output_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

# --- Chargement DINO ---
dino_config = "msd_implementation/configs/grounding_dino/pancreas_tumor.py"
dino_checkpoint = "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth"
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

# --- Données ---
base_dir = "data/MSD_pancreas"
val_json_path = os.path.join(base_dir, "val.json")

with open(val_json_path, 'r') as f:
    val_data = json.load(f)

print(f"\nDebut de la calibration DINO sur {len(val_data['images'])} images de validation...")

thresholds_to_test = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
results = {t: {'tp': 0, 'fn': 0, 'total_fp_boxes': 0} for t in thresholds_to_test}

for img_info in tqdm(val_data['images'], desc="Inference brute"):
    file_name = img_info['file_name']
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue
        
    true_seg_raw = io.imread(mask_path)
    has_tumor = (true_seg_raw == 2).sum() > 0
    
    # Masque binaire de la tumeur pour vérifier l'overlap
    true_tumor_mask = np.zeros_like(true_seg_raw, dtype=np.uint8)
    true_tumor_mask[true_seg_raw == 2] = 1

    with torch.no_grad():
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        
    pancreas_mask = (labels == 0) & (scores > 0.3)
    pancreas_boxes = bboxes[pancreas_mask]
    pancreas_scores = scores[pancreas_mask]
    
    # Toutes les prédictions "tumeur" brutes
    all_tumor_mask = (labels == 1)
    all_tumor_boxes = bboxes[all_tumor_mask]
    all_tumor_scores = scores[all_tumor_mask]

    # Simulation pour chaque seuil
    for t in thresholds_to_test:
        valid_tumor_boxes = []
        
        # 1. Filtre de confiance
        t_mask = all_tumor_scores >= t
        t_boxes = all_tumor_boxes[t_mask]
        
        # 2. Cascade du pancréas
        if len(pancreas_boxes) > 0 and len(t_boxes) > 0:
            p_x1, p_y1, p_x2, p_y2 = pancreas_boxes[np.argmax(pancreas_scores)]
            p_x1, p_y1 = max(0, p_x1 - 20), max(0, p_y1 - 20)
            p_x2, p_y2 = p_x2 + 20, p_y2 + 20

            for box in t_boxes:
                inter_x1, inter_y1 = max(box[0], p_x1), max(box[1], p_y1)
                inter_x2, inter_y2 = min(box[2], p_x2), min(box[3], p_y2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    overlap_ratio = ((inter_x2 - inter_x1) * (inter_y2 - inter_y1)) / ((box[2] - box[0]) * (box[3] - box[1]))
                    if overlap_ratio >= 0.1:
                        valid_tumor_boxes.append(box)
                        
        elif len(pancreas_boxes) == 0 and len(t_boxes) > 0:
            valid_tumor_boxes = t_boxes
            
        # 3. Évaluation par rapport à la vérité terrain
        tumor_found = False
        fp_boxes_in_image = 0
        
        for box in valid_tumor_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Filtre géométrique
            if not (100 <= (x2 - x1) * (y2 - y1) <= 15000):
                continue
                
            box_gt = true_tumor_mask[max(0, y1):min(true_tumor_mask.shape[0], y2), max(0, x1):min(true_tumor_mask.shape[1], x2)]
            if box_gt.sum() > 0:
                tumor_found = True
            else:
                fp_boxes_in_image += 1
                
        results[t]['total_fp_boxes'] += fp_boxes_in_image
        if has_tumor:
            if tumor_found:
                results[t]['tp'] += 1
            else:
                results[t]['fn'] += 1

# --- Affichage des résultats ---
print("\n" + "="*65)
print(f"{'Seuil DINO':<12} | {'Tumeurs trouvées (TP)':<22} | {'Ratées (FN)':<12} | {'Nb total Faux Positifs (Boîtes)':<30}")
print("-" * 65)

for t in thresholds_to_test:
    tp = results[t]['tp']
    fn = results[t]['fn']
    fp = results[t]['total_fp_boxes']
    print(f"{t:<12.2f} | {tp:<22} | {fn:<12} | {fp:<30}")
print("="*65)

metrics_dir = output_dir("msd_implementation", "dino_calibration", "metrics")
sweep_path = metrics_dir / "dino_threshold_sweep_legacy.csv"
summary_path = metrics_dir / "dino_calibration_summary_legacy.json"

pd.DataFrame(
    [{"threshold": threshold, **metrics} for threshold, metrics in results.items()]
).to_csv(sweep_path, index=False)

min_fn = min(metrics["fn"] for metrics in results.values())
recommended_threshold = max(
    threshold for threshold, metrics in results.items() if metrics["fn"] == min_fn
)
with open(summary_path, "w") as f:
    json.dump(
        {
            "recommended_threshold": recommended_threshold,
            "selection_rule": "highest_threshold_with_minimum_fn",
            "minimum_fn": min_fn,
            "results": {str(threshold): metrics for threshold, metrics in results.items()},
        },
        f,
        indent=2,
    )

print("\nAnalyse :")
print("1. Regarde la colonne 'Ratées (FN)'. Cherche le seuil le plus HAUT qui garde ce nombre à son strict minimum (idéalement 0 ou proche des ratés inévitables de DINO).")
print("2. Vérifie la colonne 'Faux Positifs'. Ce chiffre correspond aux futures images 'Classe 0' de ton dataset ResNet. Plus il est haut, plus l'entraînement ResNet sera long.")
print(f"\nCSV : {sweep_path}")
print(f"Resume JSON : {summary_path}")
