import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from skimage import io, transform
from tqdm import tqdm

# Imports Grounding DINO
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# Imports MedSAM
from MedSAM.segment_anything import sam_model_registry
from MedSAM.MedSAM_Inference import medsam_inference

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================
def calculate_dice(mask_true, mask_pred):
    m_true = np.asarray(mask_true).astype(bool)
    m_pred = np.asarray(mask_pred).astype(bool)
    if m_true.sum() + m_pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(m_true, m_pred).sum()
    return 2 * intersection / (m_true.sum() + m_pred.sum())

# ==========================================
# 1. INITIALISATION DES MODELES
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()

# --- Modèle 1: Grounding DINO (Config V3) ---
dino_config = 'tumor_config_v3.py'
dino_checkpoint = 'work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth'
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

# --- Modèle 2: MedSAM ---
medsam_checkpoint = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# ==========================================
# 2. CONFIGURATION DES CHEMINS (MSD Pancreas)
# ==========================================
base_dir = "data/MSD_pancreas"
test_json_path = os.path.join(base_dir, "test.json")
output_folder = "data/outputs_medsam_dino_msd"

os.makedirs(output_folder, exist_ok=True)
os.makedirs("data/results", exist_ok=True)

with open(test_json_path, 'r') as f:
    test_data = json.load(f)

test_images_list = test_data['images'] 

dice_list = []

print(f"\nDebut de l'evaluation sur les {len(test_images_list)} images du fichier test.json...")

# ==========================================
# 3. BOUCLE D'INFERENCE (Cascade Pancréas -> Tumeur par Overlap)
# ==========================================
for img_info in tqdm(test_images_list, desc="Inference Test Set"):
    file_name = img_info['file_name']
    img_id = img_info['id']
    
    img_path = os.path.join(base_dir, file_name)
    mask_rel_path = file_name.replace("/images/", "/masks/")
    mask_path = os.path.join(base_dir, mask_rel_path)

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue
        
    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    true_seg_raw = io.imread(mask_path)
    true_seg = np.zeros_like(true_seg_raw, dtype=np.uint8)
    true_seg[true_seg_raw == 2] = 1 # On évalue uniquement la tumeur

    has_tumor = true_seg.sum() > 0
    full_medsam_seg = np.zeros((H, W), dtype=np.uint8)
    best_score = 0.0  

    with torch.no_grad():
        # 1. On interroge les DEUX classes en même temps
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        pred_instances = result.pred_instances
        
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy() # 0 = pancréas, 1 = tumeur
        
        # 2. Séparation des prédictions
        pancreas_mask = (labels == 0) & (scores > 0.3) 
        tumor_mask = (labels == 1) & (scores > 0.01)    
        
        pancreas_boxes = bboxes[pancreas_mask]
        pancreas_scores = scores[pancreas_mask]
        tumor_boxes = bboxes[tumor_mask]
        tumor_scores = scores[tumor_mask]

        best_tumor_box = None

        # 3. LOGIQUE DE CASCADE (Filtre spatial)
        if len(pancreas_boxes) > 0 and len(tumor_boxes) > 0:
            best_panc_idx = np.argmax(pancreas_scores)
            p_x1, p_y1, p_x2, p_y2 = pancreas_boxes[best_panc_idx]
            
            margin = 20
            p_x1, p_y1 = max(0, p_x1 - margin), max(0, p_y1 - margin)
            p_x2, p_y2 = p_x2 + margin, p_y2 + margin

            valid_tumor_boxes = []
            valid_tumor_scores = []
            
            for t_box, t_score in zip(tumor_boxes, tumor_scores):
                t_x1, t_y1, t_x2, t_y2 = t_box
                
                inter_x1, inter_y1 = max(t_x1, p_x1), max(t_y1, p_y1)
                inter_x2, inter_y2 = min(t_x2, p_x2), min(t_y2, p_y2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    tumor_area = (t_x2 - t_x1) * (t_y2 - t_y1)
                    overlap_ratio = intersection_area / tumor_area
                    
                    if overlap_ratio >= 0.1:
                        valid_tumor_boxes.append(t_box)
                        valid_tumor_scores.append(t_score)
            
            if valid_tumor_boxes:
                best_idx = np.argmax(valid_tumor_scores)
                best_tumor_box = valid_tumor_boxes[best_idx]
                best_score = float(valid_tumor_scores[best_idx])
                
        elif len(pancreas_boxes) == 0 and len(tumor_boxes) > 0:
            best_idx = np.argmax(tumor_scores)
            best_tumor_box = tumor_boxes[best_idx]
            best_score = float(tumor_scores[best_idx])

        # 4. Envoi à MedSAM si une tumeur a survécu au filtre
        if best_tumor_box is not None:
            img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
            
            box_np = np.array([best_tumor_box]) 
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            
            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
            full_medsam_seg[medsam_seg > 0] = 1

    dice_score = calculate_dice(true_seg, full_medsam_seg)
    dice_list.append({
        "image_id": img_id,
        "file_name": os.path.basename(file_name),
        "dice": dice_score,
        "has_tumor": has_tumor,
        "best_dino_score": best_score 
    })

# ==========================================
# 4. ANALYSE DES RESULTATS ET DES SEUILS
# ==========================================
df = pd.DataFrame(dice_list)
df.to_csv("data/results/dice_final_report_msd_oneshot.csv", index=False)

df_tumor = df[df['has_tumor'] == True]
df_no_tumor = df[df['has_tumor'] == False]

print("\n" + "="*40)
print("RESULTATS GLOBAUX (Toutes les images)")
print("-" * 40)
print(f"DICE MOYEN : {df['dice'].mean():.4f}")
print(f"DICE MEDIAN : {df['dice'].median():.4f}")

print("\n" + "="*40)
print(f"SCANS AVEC TUMEUR ({len(df_tumor)} images)")
print("-" * 40)
if not df_tumor.empty:
    print(f"DICE MOYEN : {df_tumor['dice'].mean():.4f}")
    print(f"DICE MEDIAN : {df_tumor['dice'].median():.4f}")

print("\n" + "="*40)
print(f"SCANS SANS TUMEUR ({len(df_no_tumor)} images)")
print("-" * 40)
if not df_no_tumor.empty:
    print(f"DICE MOYEN : {df_no_tumor['dice'].mean():.4f}")
    print(f"FAUX POSITIFS (DICE = 0.0) : {len(df_no_tumor[df_no_tumor['dice'] == 0.0])} images")

print("\n" + "="*40)
print("ETUDE DU SEUIL DE DETECTION (DINO SCORE)")
print("-" * 40)

stats = df.groupby('has_tumor')['best_dino_score'].describe()
print("Statistiques des scores de confiance :")
print(stats[['mean', 'min', '50%', 'max']].to_string())

if not df_no_tumor.empty and not df_tumor.empty:
    max_fp_score = df_no_tumor['best_dino_score'].max()
    print(f"\n-> Le score maximal sur un scan SAIN est de : {max_fp_score:.4f}")
    
    sim_fp = len(df_no_tumor[df_no_tumor['best_dino_score'] > max_fp_score])
    sim_fn = len(df_tumor[df_tumor['best_dino_score'] <= max_fp_score])
    
    print(f"Impact d'un threshold a {max_fp_score:.4f} :")
    print(f"- Faux positifs restants : {sim_fp} / {len(df_no_tumor)}")
    print(f"- Vraies tumeurs ignorees (Faux negatifs) : {sim_fn} / {len(df_tumor)}")
print("="*40)

print("\n" + "="*40)
print("ANALYSE DES FAUX NÉGATIFS (Tumeurs ratées)")
print("-" * 40)
if not df_tumor.empty:
    df_fn = df_tumor[df_tumor['dice'] == 0.0]
    print(f"Tumeurs totalement ratées (Dice = 0.0) : {len(df_fn)} / {len(df_tumor)} images")
    if not df_fn.empty:
        print(f"Soit un taux de Faux Négatifs de : {(len(df_fn)/len(df_tumor))*100:.1f}%")
        print("\nNote : Sur ces images, DINO n'a fait aucune proposition (ou elles ont été filtrées).")