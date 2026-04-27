"""3-slice v2 extraction: same as v1 but with ``crop_margin=30``.

The wider crop exposes the surrounding pancreatic parenchyma to the
classifier so the "tumor vs healthy pancreas" decision is no longer made
on the lesion alone. Output dataset: ``data/classifier_dataset_resnet50_wide_crop/``.

Usage (from repo root):
    python -m experiments.msd.resnet50_wide_crop.extract_hard_negatives
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import io
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.msd._shared.proposal_strategy import (
    ProposalConfig,
    box_tumor_overlap,
    clip_box,
    expand_box,
    extract_dino_candidates,
)
from experiments.msd.three_slice_context.slice_stack import stack_3slice_image


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extraction 3-slice v2 (crop_margin=30) sur : {device}")

random.seed(42)
np.random.seed(42)

dino_config = "work_dirs/tumor_config_v3/tumor_config_v3.py"
dino_checkpoint = "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth"
dino_model = init_detector(dino_config, dino_checkpoint, device=device)

base_dir = "data/MSD_pancreas"
output_base_dir = "data/classifier_dataset_resnet50_wide_crop"

# Key change vs v1: crop_margin=30 (was 10). The classifier gets enough
# pancreatic context to see the differential between tumor and surrounding
# parenchyma. The wider crop is automatically resized to 224x224 by the
# train transforms, so no architectural change is required.
MINING_CFG = ProposalConfig(
    tumor_score_thresh=0.01,
    pancreas_margin=45,
    min_pancreas_overlap=0.00,
    min_box_area=50,
    max_box_area=26000,
    top_k_candidates=12,
    nms_iou=0.65,
    crop_margin=30,
)

TARGET_NEGATIVE_RATIO = 4
MIN_NEGATIVE_RATIO = 2
GT_POSITIVE_JITTERS = 2
POSITIVE_OVERLAP_RATIO = 0.25
MIN_POSITIVE_PIXELS = 25
LOW_OVERLAP_NEGATIVE_RATIO = 0.02
LOW_OVERLAP_NEGATIVE_PIXELS = 10


def reset_output_dirs(*dirs):
    for folder in dirs:
        os.makedirs(folder, exist_ok=True)
        for path in glob.glob(os.path.join(folder, "*.png")):
            os.remove(path)


def tumor_bbox(true_tumor_mask):
    ys, xs = np.where(true_tumor_mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def jitter_box(box, width, height, max_shift=0.12, max_scale=0.18):
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    cx += random.uniform(-max_shift, max_shift) * bw
    cy += random.uniform(-max_shift, max_shift) * bh
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    bw *= scale
    bh *= scale

    return clip_box(
        [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], width, height
    )


def crop_3slice(stacked_img: np.ndarray, box, margin: int) -> np.ndarray | None:
    h, w = stacked_img.shape[:2]
    box_clipped = clip_box(box, w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in expand_box(box_clipped, margin, w, h)]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = stacked_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def save_patch(
    stacked_img,
    box,
    out_folder,
    split_name,
    img_path,
    patch_count,
    margin=None,
    tag="patch",
):
    margin = MINING_CFG.crop_margin if margin is None else margin
    crop = crop_3slice(stacked_img, box, margin=margin)
    if crop is None:
        return patch_count

    stem = Path(img_path).stem
    save_path = os.path.join(
        out_folder, f"{split_name}_{tag}_{stem}_{patch_count:06d}.png"
    )
    Image.fromarray(crop).save(save_path)
    return patch_count + 1


def extract_hard_patches(split_name):
    print(f"\n--- Traitement du set : {split_name} (3-slice v2, margin=30) ---")

    with open(os.path.join(base_dir, f"{split_name}.json"), "r") as f:
        split_data = json.load(f)

    out_dir_0 = os.path.join(output_base_dir, split_name, "0")
    out_dir_1 = os.path.join(output_base_dir, split_name, "1")
    reset_output_dirs(out_dir_0, out_dir_1)

    patch_count = 0
    dino_positive_count = 0
    gt_positive_count = 0
    hard_negative_count = 0

    for img_info in tqdm(split_data["images"], desc=f"Extraction {split_name}"):
        img_rel_path = img_info["file_name"]
        img_path = os.path.join(base_dir, img_rel_path)
        mask_path = os.path.join(base_dir, img_rel_path.replace("/images/", "/masks/"))

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        stacked_img = stack_3slice_image(img_path)
        H, W = stacked_img.shape[:2]

        true_seg_raw = io.imread(mask_path)
        true_tumor = (true_seg_raw == 2).astype(np.uint8)

        gt_box = tumor_bbox(true_tumor)
        if gt_box is not None:
            boxes = [expand_box(gt_box, MINING_CFG.crop_margin, W, H)]
            boxes.extend(jitter_box(gt_box, W, H) for _ in range(GT_POSITIVE_JITTERS))
            for box in boxes:
                patch_count = save_patch(
                    stacked_img,
                    box,
                    out_dir_1,
                    split_name,
                    img_path,
                    patch_count,
                    margin=MINING_CFG.crop_margin,
                    tag="gtpos",
                )
                gt_positive_count += 1

        with torch.no_grad():
            result = inference_detector(
                dino_model, img_path, text_prompt="pancreas . tumor ."
            )
            pred = result.pred_instances
            candidates = extract_dino_candidates(
                pred.scores.cpu().numpy(),
                pred.bboxes.cpu().numpy(),
                pred.labels.cpu().numpy(),
                image_shape=(H, W),
                config=MINING_CFG,
            )

        for cand in candidates:
            overlap_pixels, overlap_ratio = box_tumor_overlap(cand["box"], true_tumor)

            is_positive = (
                overlap_ratio >= POSITIVE_OVERLAP_RATIO
                and overlap_pixels >= MIN_POSITIVE_PIXELS
            )
            is_safe_negative = overlap_pixels == 0 or (
                overlap_ratio <= LOW_OVERLAP_NEGATIVE_RATIO
                and overlap_pixels <= LOW_OVERLAP_NEGATIVE_PIXELS
            )

            if is_positive:
                out_folder = out_dir_1
                dino_positive_count += 1
                tag = "dinopos"
            elif is_safe_negative:
                out_folder = out_dir_0
                hard_negative_count += 1
                tag = "hardneg"
            else:
                continue

            patch_count = save_patch(
                stacked_img,
                cand["box"],
                out_folder,
                split_name,
                img_path,
                patch_count,
                margin=MINING_CFG.crop_margin,
                tag=tag,
            )

    print(f"\nEquilibrage des classes pour {split_name}...")
    files_0 = glob.glob(os.path.join(out_dir_0, "*.png"))
    files_1 = glob.glob(os.path.join(out_dir_1, "*.png"))

    count_1 = len(files_1)
    count_0 = len(files_0)

    print(f"Avant equilibrage : {count_0} Hard Negatives (0), {count_1} Tumeurs (1)")
    print(f"  -> Positifs GT jitteres : {gt_positive_count}")
    print(f"  -> Positifs DINO : {dino_positive_count}")
    print(f"  -> Negatifs DINO : {hard_negative_count}")

    target_count_0 = count_1 * TARGET_NEGATIVE_RATIO
    if count_0 > target_count_0:
        random.shuffle(files_0)
        files_to_delete = files_0[target_count_0:]
        for f_path in files_to_delete:
            os.remove(f_path)
        print(f"-> Supprime {len(files_to_delete)} fichiers de la classe 0.")
    else:
        print("-> Pas de suppression necessaire.")

    files_0 = glob.glob(os.path.join(out_dir_0, "*.png"))
    files_1 = glob.glob(os.path.join(out_dir_1, "*.png"))
    min_count_0 = len(files_1) * MIN_NEGATIVE_RATIO
    if len(files_0) < min_count_0 and len(files_0) > 0:
        max_count_1 = max(1, len(files_0) // MIN_NEGATIVE_RATIO)
        if len(files_1) > max_count_1:
            dino_pos = [p for p in files_1 if f"{split_name}_dinopos_" in Path(p).name]
            gt_pos = [p for p in files_1 if f"{split_name}_gtpos_" in Path(p).name]
            random.shuffle(dino_pos)
            random.shuffle(gt_pos)
            keep = set((gt_pos + dino_pos)[:max_count_1])
            files_to_delete = [p for p in files_1 if p not in keep]
            for f_path in files_to_delete:
                os.remove(f_path)
            print(
                "-> Supprime "
                f"{len(files_to_delete)} positifs pour garder au moins "
                f"{MIN_NEGATIVE_RATIO}:1 negatifs/positif."
            )

    final_0 = len(glob.glob(os.path.join(out_dir_0, "*.png")))
    final_1 = len(glob.glob(os.path.join(out_dir_1, "*.png")))
    print(f"Apres equilibrage : {final_0} Hard Negatives (0), {final_1} Tumeurs (1)\n")


if __name__ == "__main__":
    extract_hard_patches("train")
    extract_hard_patches("val")
