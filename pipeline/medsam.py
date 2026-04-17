"""MedSAM wrapper with image-embedding cache + point-prompt support.

The stock `MedSAM/MedSAM_Inference.py::medsam_inference` re-encodes the
image every call and only accepts boxes. We wrap it so that:
- `encode_image` runs the heavy image encoder ONCE per image,
- `segment` reuses the cached embedding and accepts points + box.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, transform

from . import config
from .models import get_medsam_model


def _device(embed_or_model) -> str:
    if hasattr(embed_or_model, "device"):
        return str(embed_or_model.device)
    return str(next(embed_or_model.parameters()).device)


def load_image(image_path: str) -> np.ndarray:
    """Read → 3-channel uint8. Matches the pattern in test_gd.ipynb."""
    img = io.imread(image_path)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _to_1024_tensor(image_np: np.ndarray, device: str) -> torch.Tensor:
    side = config.MEDSAM_INPUT_SIDE
    img_1024 = transform.resize(
        image_np, (side, side), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    return (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )


@torch.no_grad()
def encode_image(image_np: np.ndarray) -> torch.Tensor:
    """Run the heavy image encoder once. Result is meant to be cached."""
    model = get_medsam_model()
    device = _device(model)
    img_1024 = _to_1024_tensor(image_np, device)
    if config.USE_BF16 and device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.image_encoder(img_1024)
    return model.image_encoder(img_1024)


def _box_to_1024(box_xyxy: list[float], H: int, W: int) -> np.ndarray:
    side = config.MEDSAM_INPUT_SIDE
    box_np = np.array([box_xyxy], dtype=np.float32)
    return box_np / np.array([W, H, W, H], dtype=np.float32) * side


def _points_to_1024(
    points: list[list[float]], H: int, W: int
) -> np.ndarray:
    side = config.MEDSAM_INPUT_SIDE
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = pts[:, 0] / W * side
    pts[:, 1] = pts[:, 1] / H * side
    return pts


@torch.no_grad()
def segment(
    image_embed: torch.Tensor,
    H: int,
    W: int,
    box: list[float] | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
) -> np.ndarray:
    """Decode a mask from the cached embedding and prompt(s).

    Returns a (H, W) uint8 mask at the original image resolution.
    """
    model = get_medsam_model()
    device = _device(image_embed)

    box_torch = None
    if box is not None:
        b1024 = _box_to_1024(box, H, W)
        box_torch = torch.as_tensor(b1024, dtype=torch.float, device=device)
        if box_torch.ndim == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

    pts_arg = None
    if points and len(points) > 0:
        pts_1024 = _points_to_1024(points, H, W)
        pts_tensor = torch.as_tensor(pts_1024, dtype=torch.float, device=device)
        pts_tensor = pts_tensor.unsqueeze(0)  # (1, N, 2)
        labels = torch.as_tensor(
            point_labels or [1] * len(points), dtype=torch.int, device=device
        ).unsqueeze(0)  # (1, N)
        pts_arg = (pts_tensor, labels)

    sparse_emb, dense_emb = model.prompt_encoder(
        points=pts_arg, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
    )
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
