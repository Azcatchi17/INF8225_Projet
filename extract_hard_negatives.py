# script_1_extract_hard_negatives.py
import os
import glob
import json
import random
import numpy as np
import torch
from skimage import io
from PIL import Image
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

# ==========================================
# CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extraction sur : {device}")

dino_config = 'tumor_config_v3.py'
dino_checkpoint = 'work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth'
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

base_dir = "data/MSD_pancreas"
# NOUVEAU DOSSIER pour ne pas écraser l'ancien
output_base_dir = "data/classifier_dataset_hard" 

TARGET_NEGATIVE_RATIO = 3
margin_crop = 5

# ==========================================
# FONCTION D'EXTRACTION
# ==========================================
def extract_hard_patches(split_name):
    print(f"\n--- Traitement du set : {split_name} ---")
    
    # Lecture depuis le JSON pour être sûr d'avoir les bonnes images du split
    with open(os.path.join(base_dir, f"{split_name}.json"), 'r') as f:
        split_data = json.load(f)
    
    out_dir_0 = os.path.join(output_base_dir, split_name, "0")
    out_dir_1 = os.path.join(output_base_dir, split_name, "1")
    os.makedirs(out_dir_0, exist_ok=True)
    os.makedirs(out_dir_1, exist_ok=True)

    patch_count = 0

    # 1. PHASE D'EXTRACTION (Avec Cascade)
    for img_info in tqdm(split_data['images'], desc=f"Extraction {split_name}"):
        img_rel_path = img_info['file_name']
        img_path = os.path.join(base_dir, img_rel_path)
        
        mask_rel_path = img_rel_path.replace("/images/", "/masks/")
        mask_path = os.path.join(base_dir, mask_rel_path)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path): 
            continue
            
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        # Vérité terrain
        true_seg_raw = io.imread(mask_path)
        true_tumor = np.zeros_like(true_seg_raw, dtype=np.uint8)
        true_tumor[true_seg_raw == 2] = 1

        with torch.no_grad():
            # PROMPT COMPLET pour avoir les mêmes propositions qu'en inférence
            result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
            pred = result.pred_instances
            
            scores = pred.scores.cpu().numpy()
            bboxes = pred.bboxes.cpu().numpy()
            labels = pred.labels.cpu().numpy()
            
            pancreas_mask = (labels == 0) & (scores > 0.3) 
            tumor_mask = (labels == 1) & (scores > 0.05)    
            
            pancreas_boxes = bboxes[pancreas_mask]
            pancreas_scores = scores[pancreas_mask]
            tumor_boxes = bboxes[tumor_mask]
            tumor_scores = scores[tumor_mask]

            valid_tumor_boxes = []
            
            # FILTRE SPATIAL (La Cascade)
            if len(pancreas_boxes) > 0 and len(tumor_boxes) > 0:
                best_panc_idx = np.argmax(pancreas_scores)
                p_x1, p_y1, p_x2, p_y2 = pancreas_boxes[best_panc_idx]
                
                margin_panc = 20
                p_x1, p_y1 = max(0, p_x1 - margin_panc), max(0, p_y1 - margin_panc)
                p_x2, p_y2 = p_x2 + margin_panc, p_y2 + margin_panc

                for t_box in tumor_boxes:
                    t_x1, t_y1, t_x2, t_y2 = t_box
                    inter_x1, inter_y1 = max(t_x1, p_x1), max(t_y1, p_y1)
                    inter_x2, inter_y2 = min(t_x2, p_x2), min(t_y2, p_y2)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        tumor_area = (t_x2 - t_x1) * (t_y2 - t_y1)
                        overlap_ratio = intersection_area / tumor_area
                        if overlap_ratio >= 0.1:
                            valid_tumor_boxes.append(t_box)
                            
            elif len(pancreas_boxes) == 0 and len(tumor_boxes) > 0:
                valid_tumor_boxes = tumor_boxes # Fallback

            # SAUVEGARDE DES PATCHS
            for box in valid_tumor_boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Check overlap avec la vérité terrain de la tumeur
                box_gt = true_tumor[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                box_area = (x2 - x1) * (y2 - y1)
                overlap_pixels = box_gt.sum()
                overlap_ratio = overlap_pixels / box_area if box_area > 0 else 0
                
                if overlap_ratio >= 0.40 and overlap_pixels >= 50:
                    out_folder = out_dir_1 # Vraie tumeur
                elif overlap_ratio <= 0.05 and overlap_pixels == 0:
                    out_folder = out_dir_0 # HARD NEGATIVE ! (DINO a cru que c'était une tumeur, mais c'est faux)
                else:
                    continue # Cas ambigu, on ignore pour ne pas perturber le modèle
                
                c_x1, c_y1 = max(0, x1 - margin_crop), max(0, y1 - margin_crop)
                c_x2, c_y2 = min(W, x2 + margin_crop), min(H, y2 + margin_crop)
                
                crop = img_3c[c_y1:c_y2, c_x1:c_x2]
                
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    crop_img = Image.fromarray(crop)
                    save_path = os.path.join(out_folder, f"patch_{os.path.basename(img_path)}_{patch_count}.png")
                    crop_img.save(save_path)
                    patch_count += 1

    # 2. PHASE D'EQUILIBRAGE
    print(f"\nEquilibrage des classes pour {split_name}...")
    files_0 = glob.glob(os.path.join(out_dir_0, "*.png"))
    files_1 = glob.glob(os.path.join(out_dir_1, "*.png"))
    
    count_1 = len(files_1)
    count_0 = len(files_0)
    
    print(f"Avant equilibrage : {count_0} Hard Negatives (0), {count_1} Tumeurs (1)")
    
    target_count_0 = count_1 * TARGET_NEGATIVE_RATIO
    
    if count_0 > target_count_0:
        random.shuffle(files_0)
        files_to_delete = files_0[target_count_0:]
        for f_path in files_to_delete:
            os.remove(f_path)
        print(f"-> Supprime {len(files_to_delete)} fichiers de la classe 0.")
    else:
        print("-> Pas de suppression necessaire.")
        
    final_0 = len(glob.glob(os.path.join(out_dir_0, "*.png")))
    print(f"Apres equilibrage : {final_0} Hard Negatives (0), {count_1} Tumeurs (1)\n")

# Lancement sur les deux sets vitaux
extract_hard_patches("train")
extract_hard_patches("val")