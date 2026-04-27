import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision
from skimage import io, transform
from tqdm import tqdm

# Fonction utilitaire
def calculate_dice(mask_true, mask_pred):
    m_true = np.asarray(mask_true).astype(bool)
    m_pred = np.asarray(mask_pred).astype(bool)
    if m_true.sum() + m_pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(m_true, m_pred).sum()
    return 2 * intersection / (m_true.sum() + m_pred.sum())

# ==========================================
# FONCTION PRINCIPALE
# ==========================================
def run_calibration(dino_config, dino_checkpoint, medsam_checkpoint, 
                    base_img_folder, base_mask_folder, val_json_path):
    
    # Imports tardifs pour s'assurer que le path est bien configuré
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    from MedSAM.segment_anything import sam_model_registry
    from MedSAM.MedSAM_Inference import medsam_inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil détecté : {device}")
    
    register_all_modules()

    # Initialisation des modèles
    dino_model = init_detector(dino_config, dino_checkpoint, device=device)
    medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint).to(device)
    medsam_model.eval()

    with open(val_json_path, 'r') as f:
        val_images_list = json.load(f)['images']

    # --- PHASE 1 : MISE EN CACHE ---
    print("\n[Phase 1/2] Extraction des descripteurs et boîtes de base (Seuil 0.05)...")
    cache = []
    BASE_THRESH = 0.05

    for img_info in tqdm(val_images_list, desc="Mise en cache"):
        file_name = img_info['file_name']
        img_path = os.path.join(base_img_folder, file_name)
        mask_path = os.path.join(base_mask_folder, file_name)

        if not os.path.exists(img_path):
            continue

        img_np = io.imread(img_path)
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
        H, W, _ = img_3c.shape

        true_seg = io.imread(mask_path)
        true_seg = true_seg[:,:,0] if len(true_seg.shape) == 3 else true_seg
        true_seg = (true_seg > 0).astype(np.uint8)

        with torch.no_grad():
            result = inference_detector(dino_model, img_path, text_prompt="polyp.")
            scores = result.pred_instances.scores
            bboxes = result.pred_instances.bboxes
            
            mask_conf = scores >= BASE_THRESH
            valid_boxes = bboxes[mask_conf]
            valid_scores = scores[mask_conf]

            img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        cache.append({
            'H': H, 'W': W,
            'true_seg': true_seg,
            'boxes': valid_boxes, 
            'scores': valid_scores,
            'embedding': image_embedding.cpu() 
        })

    # --- PHASE 2 : CALIBRATION ---
    print("\n[Phase 2/2] Test des seuils (Optimisation du DICE)...")
    thresholds_to_test = np.arange(0.05, 1.0, 0.05)

    best_thresh = 0.05
    best_dice = 0.0

    for thresh in thresholds_to_test:
        dice_scores = []
        
        for item in cache:
            valid_mask = item['scores'] >= thresh
            current_boxes = item['boxes'][valid_mask]
            current_scores = item['scores'][valid_mask]
            
            full_medsam_seg = np.zeros((item['H'], item['W']), dtype=np.uint8)
            
            if len(current_boxes) > 0:
                keep_indices = torchvision.ops.nms(current_boxes, current_scores, iou_threshold=0.5)
                final_boxes = current_boxes[keep_indices].cpu().numpy()
                
                embed_gpu = item['embedding'].to(device)
                
                with torch.no_grad():
                    for box in final_boxes:
                        box_np = np.array([box])
                        box_1024 = box_np / np.array([item['W'], item['H'], item['W'], item['H']]) * 1024
                        medsam_seg = medsam_inference(medsam_model, embed_gpu, box_1024, item['H'], item['W'])
                        full_medsam_seg[medsam_seg > 0] = 1
            
            dice = calculate_dice(item['true_seg'], full_medsam_seg)
            dice_scores.append(dice)
            
        mean_dice = np.mean(dice_scores)
        print(f"-> Seuil {thresh:.2f} : DICE Moyen = {mean_dice:.4f}")
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_thresh = thresh

    print("\n" + "="*40)
    print("CALIBRATION TERMINÉE")
    print(f"Meilleur seuil (box_threshold) : {best_thresh:.2f}")
    print(f"DICE correspondant sur validation : {best_dice:.4f}")
    print("="*40)

# ==========================================
# POINT D'ENTRÉE DU SCRIPT
# ==========================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calibration des seuils")
    parser.add_argument('--mode', type=str, choices=['zeroshot', 'finetuned'], 
                        default='finetuned', help="Choix du modèle à évaluer")
    args = parser.parse_args()

    # 1. Résolution de la racine du projet
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # 2. Positionnement dans la racine pour les imports (MedSAM, mmdet)
    os.chdir(PROJECT_ROOT)
    sys.path.append(PROJECT_ROOT)

    # 3. Définition des chemins absolus
    if args.mode == 'zeroshot':
        print("\n=== MODE ZERO-SHOT ACTIVÉ ===")
        cfg_path = os.path.join(PROJECT_ROOT, 'models_weights/grounding_dino_swin-t_pretrain_obj365_goldg.py')
        ckpt_path = os.path.join(PROJECT_ROOT, 'models_weights/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth')
    else:
        print("\n=== MODE FINE-TUNED ACTIVÉ ===")
        cfg_path = os.path.join(PROJECT_ROOT, 'work_dirs/polyp_config_v2/polyp_config_v2.py')
        ckpt_path = os.path.join(PROJECT_ROOT, 'work_dirs/polyp_config_v2/best_coco_bbox_mAP_epoch_5.pth')

    medsam_ckpt = os.path.join(PROJECT_ROOT, "MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
    
    img_dir = os.path.join(PROJECT_ROOT, "data/Kvasir-SEG/images")
    mask_dir = os.path.join(PROJECT_ROOT, "data/Kvasir-SEG/masks")
    val_json = os.path.join(PROJECT_ROOT, "data/Kvasir-SEG/val.json")

    # 4. Lancement
    run_calibration(cfg_path, ckpt_path, medsam_ckpt, img_dir, mask_dir, val_json)