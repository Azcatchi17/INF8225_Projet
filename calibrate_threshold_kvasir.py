import torch
import numpy as np
from tqdm import tqdm

# 1. PARAMÈTRES
BASE_THRESHOLD = 0.05  # Le seuil le plus bas pour tout capturer
thresholds_to_test = np.arange(0.1, 1.0, 0.1)
calibration_results = {}

print(f"Phase 1 : Extraction des prédictions (Seuil de base = {BASE_THRESHOLD})")
cached_predictions = []

# --- PHASE 1 : INFERENCE DINO UNIQUE ---
for image, gt_mask, gt_boxes in tqdm(val_dataloader, desc="Inférence DINO"):
    # Appel à Grounding DINO une seule fois par image
    boxes, scores = run_dino_inference(image, box_threshold=BASE_THRESHOLD, text_prompt="polyp .")
    
    # Stockage en mémoire RAM
    cached_predictions.append({
        'image': image,
        'gt_mask': gt_mask,
        'gt_boxes': gt_boxes, # Pour la méthode IoU si besoin
        'boxes': boxes,       # Tenseur de toutes les boîtes
        'scores': scores      # Tenseur de tous les scores
    })

print("\nPhase 2 : Calibration des seuils hors-ligne")

# --- PHASE 2 : FILTRAGE ET EVALUATION RAPIDE ---
for thresh in thresholds_to_test:
    dice_scores = []
    
    for item in cached_predictions:
        boxes = item['boxes']
        scores = item['scores']
        
        # Filtrage instantané via masque booléen (Pas d'appel au modèle !)
        valid_mask = scores >= thresh
        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        
        if len(valid_boxes) == 0:
            dice_scores.append(0.0)
            continue
            
        # Sélection Top-1 (Kvasir)
        best_box = valid_boxes[torch.argmax(valid_scores)]
        
        # Inférence MedSAM (obligatoire si on optimise le DICE)
        pred_mask = run_medsam_inference(item['image'], best_box)
        
        # Calcul DICE
        intersection = np.logical_and(pred_mask, item['gt_mask']).sum()
        somme = pred_mask.sum() + item['gt_mask'].sum()
        dice = (2. * intersection) / somme if somme > 0 else 1.0
        
        dice_scores.append(dice)
        
    mean_dice = np.mean(dice_scores)
    calibration_results[thresh] = mean_dice
    print(f"-> Seuil {thresh:.2f} : DICE = {mean_dice:.4f}")

# Résultat final
best_thresh = max(calibration_results, key=calibration_results.get)
print(f"\nMeilleur seuil : {best_thresh:.2f}")