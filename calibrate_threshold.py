# script_3_calibrate_threshold.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from skimage import io
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Appareil detecte : {device}")

register_all_modules()

# --- Modèles ---
dino_model = init_detector('work_dirs/tumor_config_v3/tumor_config_v3.py', 
                           'work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth', device=device)

ensemble_models = []
print("Chargement de l'ensemble ResNet-18 (5 modeles)...")
for i in range(1, 6):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model.load_state_dict(torch.load(f"resnet_fold_{i}.pth", map_location=device))
    model = model.to(device)
    model.eval()
    ensemble_models.append(model)

crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

base_dir = "data/MSD_pancreas"
val_json_path = os.path.join(base_dir, "val.json") # CIBLE : VALIDATION

with open(val_json_path, 'r') as f:
    val_data = json.load(f)

print(f"\nDebut de la calibration sur les {len(val_data['images'])} images de validation...")
results_list = []

for img_info in tqdm(val_data['images'], desc="Calibration Val Set"):
    file_name = img_info['file_name']
    img_path = os.path.join(base_dir, file_name)
    mask_path = os.path.join(base_dir, file_name.replace("/images/", "/masks/"))

    if not os.path.exists(img_path) or not os.path.exists(mask_path): continue
        
    img_np = io.imread(img_path)
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
    H, W, _ = img_3c.shape

    true_seg_raw = io.imread(mask_path)
    has_tumor = (true_seg_raw == 2).sum() > 0
    prob_tumor = 0.0 

    with torch.no_grad():
        result = inference_detector(dino_model, img_path, text_prompt="pancreas . tumor .")
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        
        pancreas_mask = (labels == 0) & (scores > 0.3)
        tumor_mask = (labels == 1) & (scores > 0.05)    
        
        pancreas_boxes, pancreas_scores = bboxes[pancreas_mask], scores[pancreas_mask]
        tumor_boxes, tumor_scores = bboxes[tumor_mask], scores[tumor_mask]

        best_tumor_box = None

        if len(pancreas_boxes) > 0 and len(tumor_boxes) > 0:
            p_x1, p_y1, p_x2, p_y2 = pancreas_boxes[np.argmax(pancreas_scores)]
            p_x1, p_y1 = max(0, p_x1 - 20), max(0, p_y1 - 20)
            p_x2, p_y2 = p_x2 + 20, p_y2 + 20

            valid_tumor_boxes, valid_tumor_scores = [], []
            for t_box, t_score in zip(tumor_boxes, tumor_scores):
                inter_x1, inter_y1 = max(t_box[0], p_x1), max(t_box[1], p_y1)
                inter_x2, inter_y2 = min(t_box[2], p_x2), min(t_box[3], p_y2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    if ((inter_x2 - inter_x1) * (inter_y2 - inter_y1)) / ((t_box[2] - t_box[0]) * (t_box[3] - t_box[1])) >= 0.1:
                        valid_tumor_boxes.append(t_box)
                        valid_tumor_scores.append(t_score)
            
            if valid_tumor_boxes:
                best_tumor_box = valid_tumor_boxes[np.argmax(valid_tumor_scores)]
                
        elif len(pancreas_boxes) == 0 and len(tumor_boxes) > 0:
            best_tumor_box = tumor_boxes[np.argmax(tumor_scores)]

        if best_tumor_box is not None:
            x1, y1, x2, y2 = map(int, best_tumor_box)
            if 100 <= (x2 - x1) * (y2 - y1) <= 15000:
                c_x1, c_y1 = max(0, x1 - 5), max(0, y1 - 5)
                c_x2, c_y2 = min(W, x2 + 5), min(H, y2 + 5)
                crop_np = img_3c[c_y1:c_y2, c_x1:c_x2]
                
                if crop_np.shape[0] > 0 and crop_np.shape[1] > 0:
                    crop_tensor = crop_transform(Image.fromarray(crop_np)).unsqueeze(0).to(device)
                    prob_tumor = sum(torch.nn.functional.softmax(model(crop_tensor), dim=1)[0][1].item() for model in ensemble_models) / 5.0

    results_list.append({'has_tumor': has_tumor, 'resnet_prob': prob_tumor})

df = pd.DataFrame(results_list)

# Ajuster ce quota en fonction de la taille du set de validation
# Si Val contient 50 sains, 15 FP est correct. Si Val contient 100 sains, mettre 30 FP.
MAX_FAUX_POSITIFS = int(len(df[df['has_tumor'] == False]) * 0.30) 

best_f2 = -1.0
optimal_thresh = 0.5

for t in np.linspace(0.01, 0.99, 99):
    tp = len(df[(df['has_tumor'] == True) & (df['resnet_prob'] >= t)])
    fn = len(df[(df['has_tumor'] == True) & (df['resnet_prob'] < t)])
    fp = len(df[(df['has_tumor'] == False) & (df['resnet_prob'] >= t)])
    
    if fp > MAX_FAUX_POSITIFS or (tp + fp) == 0 or (tp + fn) == 0: continue
        
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f_beta = (5 * precision * recall) / ((4 * precision) + recall)
    
    if f_beta > best_f2:
        best_f2, optimal_thresh = f_beta, t

print(f"\n-> Seuil Optimal trouve sur Validation : {optimal_thresh:.2f} (Sauvegarde dans optimal_threshold.txt)")
with open("optimal_threshold.txt", "w") as f:
    f.write(str(optimal_thresh))