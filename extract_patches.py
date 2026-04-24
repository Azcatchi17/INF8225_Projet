# script_1_extract_patches.py
import os
import glob
import random
import numpy as np
import torch
from skimage import io
from PIL import Image
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extraction sur : {device}")

dino_model = init_detector('tumor_config_v3.py', 'work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth', device=device)

base_dir = "data/MSD_pancreas"
output_base_dir = "data/classifier_dataset"

# Ratio de déséquilibre ciblé (ex: 3 signifie 3 Faux Positifs pour 1 Vraie Tumeur)
TARGET_NEGATIVE_RATIO = 3

def extract_patches_for_split(split_name):
    print(f"\n--- Traitement du set : {split_name} ---")
    images_dir = os.path.join(base_dir, split_name, "images")
    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    
    out_dir_0 = os.path.join(output_base_dir, split_name, "0")
    out_dir_1 = os.path.join(output_base_dir, split_name, "1")
    os.makedirs(out_dir_0, exist_ok=True)
    os.makedirs(out_dir_1, exist_ok=True)

    # NOUVELLE MARGE : Réduite pour éviter la dilution du signal tumoral
    margin = 5
    patch_count = 0

    # 1. PHASE D'EXTRACTION GLOBALE
    for img_path in tqdm(image_paths, desc=f"Extraction {split_name}"):
        mask_path = img_path.replace("/images/", "/masks/")
        if not os.path.exists(mask_path): 
            continue
            
        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        true_seg_raw = io.imread(mask_path)
        true_tumor = np.zeros_like(true_seg_raw, dtype=np.uint8)
        true_tumor[true_seg_raw == 2] = 1

        with torch.no_grad():
            result = inference_detector(dino_model, img_path, text_prompt="tumor .")
            pred = result.pred_instances
            scores = pred.scores.cpu().numpy()
            bboxes = pred.bboxes.cpu().numpy()
            
            mask = scores > 0.01
            tumor_boxes = bboxes[mask]

            for box in tumor_boxes:
                x1, y1, x2, y2 = map(int, box)
                
                box_gt = true_tumor[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                
                box_area = (x2 - x1) * (y2 - y1)
                overlap_pixels = box_gt.sum()
                overlap_ratio = overlap_pixels / box_area if box_area > 0 else 0
                
                if overlap_ratio >= 0.40 and overlap_pixels >= 50:
                    out_folder = out_dir_1
                elif overlap_ratio <= 0.05 and overlap_pixels == 0:
                    out_folder = out_dir_0
                else:
                    continue
                
                c_x1, c_y1 = max(0, x1 - margin), max(0, y1 - margin)
                c_x2, c_y2 = min(W, x2 + margin), min(H, y2 + margin)
                
                crop = img_3c[c_y1:c_y2, c_x1:c_x2]
                
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    crop_img = Image.fromarray(crop)
                    save_path = os.path.join(out_folder, f"patch_{os.path.basename(img_path)}_{patch_count}.png")
                    crop_img.save(save_path)
                    patch_count += 1

    # 2. PHASE D'EQUILIBRAGE (Sous-échantillonnage de la classe 0)
    print(f"Equilibrage des classes pour {split_name}...")
    files_0 = glob.glob(os.path.join(out_dir_0, "*.png"))
    files_1 = glob.glob(os.path.join(out_dir_1, "*.png"))
    
    count_1 = len(files_1)
    count_0 = len(files_0)
    
    print(f"Avant equilibrage : {count_0} Faux Positifs (0), {count_1} Tumeurs (1)")
    
    target_count_0 = count_1 * TARGET_NEGATIVE_RATIO
    
    if count_0 > target_count_0:
        # On mélange aléatoirement et on supprime l'excédent
        random.shuffle(files_0)
        files_to_delete = files_0[target_count_0:]
        
        for f_path in files_to_delete:
            os.remove(f_path)
            
        print(f"-> Supprime {len(files_to_delete)} fichiers de la classe 0.")
    else:
        print("-> Pas de suppression necessaire, le ratio est deja atteint ou inferieur.")
        
    final_0 = len(glob.glob(os.path.join(out_dir_0, "*.png")))
    print(f"Apres equilibrage : {final_0} Faux Positifs (0), {count_1} Tumeurs (1)\n")

# Initialisation
extract_patches_for_split("train")
extract_patches_for_split("val")