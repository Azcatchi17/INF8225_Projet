import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from MedSAM.MedSAM_Inference import show_mask, show_box

def calculate_dice(mask_true, mask_pred):
    m_true = np.asarray(mask_true).astype(bool)
    m_pred = np.asarray(mask_pred).astype(bool)
    if m_true.sum() + m_pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(m_true, m_pred).sum()
    return 2 * intersection / (m_true.sum() + m_pred.sum())

def load_rgb_image(img_path):
    """Charge une image et assure le format RGB 3 canaux."""
    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    return img_3c

def show_segmentation(img_name, img_folder, masks_folder, output_masks_folder, bboxes):
    """
    Visualisation complète Image/Pred/GT.
    Note : On passe les fonctions de dessin sam (show_box/mask) en argument pour éviter les imports sam ici.
    """
    img_path = os.path.join(img_folder, img_name)
    img_id = os.path.splitext(img_name)[0]
    img_3c = load_rgb_image(img_path)

    medsam_seg_path = os.path.join(output_masks_folder, "seg_" + os.path.basename(img_path))
    medsam_seg = io.imread(medsam_seg_path)
    medsam_seg[medsam_seg > 0] = 1

    true_seg_path = os.path.join(masks_folder, img_name)
    true_seg = io.imread(true_seg_path)
    if len(true_seg.shape) == 3: true_seg = true_seg[:, :, 0]
    true_seg[true_seg > 0] = 1

    _, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img_3c)
    ax[1].imshow(img_3c)
    ax[2].imshow(img_3c)

    if img_id in bboxes:
        for bbox in bboxes[img_id]["bbox"]:
            box = np.array([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])
            show_box(box, ax[0])
            show_box(box, ax[1])

    show_mask(medsam_seg, ax[1])
    show_mask(true_seg, ax[2])

    ax[0].set_title("Input & BBox")
    ax[1].set_title("MedSAM Pred")
    ax[2].set_title("Ground Truth")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()