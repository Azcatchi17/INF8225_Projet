"""2D pancreas segmenter (tiny UNet) trained on MSD label==1.

Provides the anatomical gate used to filter DINO tumor candidates.
Small-footprint PyTorch UNet — trains in ~5 min on Colab T4."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader, Dataset

from . import config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PancreasMaskDataset(Dataset):
    """Returns (1, img_size, img_size) float + (img_size, img_size) bin mask.

    Reads the COCO-style test/train.json and flips the path to masks/ so we
    can load the multi-class MSD mask and isolate label==1 (pancreas)."""

    def __init__(
        self,
        coco_json: str | Path,
        data_root: str | Path,
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        with open(coco_json) as f:
            meta = json.load(f)
        self.images = meta["images"]
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        info = self.images[idx]
        img_path = self.data_root / info["file_name"]
        mask_path = self.data_root / info["file_name"].replace("/images/", "/masks/")

        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        pancreas = (mask == 1).astype(np.float32)

        img = sk_resize(img, (self.img_size, self.img_size), order=1,
                        preserve_range=True, anti_aliasing=True).astype(np.float32)
        pancreas = sk_resize(pancreas, (self.img_size, self.img_size), order=0,
                             preserve_range=True, anti_aliasing=False).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                pancreas = np.ascontiguousarray(pancreas[:, ::-1])
            if np.random.rand() < 0.3:
                img = img + np.random.randn(*img.shape).astype(np.float32) * 0.02
                img = np.clip(img, 0, 1)

        return (
            torch.from_numpy(img[None]).float(),
            torch.from_numpy(pancreas).float(),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """4-level UNet, ~1.9M params at base=32. Plenty for a single organ."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        self.e1 = _double_conv(in_ch, base)
        self.e2 = _double_conv(base, base * 2)
        self.e3 = _double_conv(base * 2, base * 4)
        self.e4 = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _double_conv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------
def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (prob * t).sum(1)
    return 1 - (2 * inter + eps) / (prob.sum(1) + t.sum(1) + eps)


def _bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = _soft_dice_loss(logits, target).mean()
    return bce + dice


def _batch_dice(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr)
    tgt = target > 0.5
    inter = (pred & tgt).sum((-1, -2)).float()
    union = pred.sum((-1, -2)).float() + tgt.sum((-1, -2)).float()
    dice = (2 * inter / (union + 1e-6))
    return float(dice.mean().item())


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    best_val_dice: float
    history: list[dict]
    save_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def train_pancreas_unet(
    train_json: str | Path,
    val_json: str | Path,
    data_root: str | Path,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 256,
    save_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    num_workers: int = 2,
) -> TrainResult:
    """Train the UNet on MSD pancreas labels (label==1). Saves best val-DICE ckpt."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(save_path) if save_path else config.PANCREAS_CKPT

    train_ds = PancreasMaskDataset(train_json, data_root, img_size=img_size, augment=True)
    val_ds = PancreasMaskDataset(val_json, data_root, img_size=img_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))

    model = TinyUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_dice = 0.0
    history: list[dict] = []
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        running = 0.0
        for img, msk in train_dl:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(img).squeeze(1)
            loss = _bce_dice_loss(logits, msk)
            loss.backward()
            opt.step()
            running += float(loss.item())
        sched.step()
        train_loss = running / max(len(train_dl), 1)

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for img, msk in val_dl:
                img = img.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)
                val_dice += _batch_dice(model(img).squeeze(1), msk)
        val_dice /= max(len(val_dl), 1)

        history.append({"epoch": ep, "train_loss": train_loss, "val_dice": val_dice})
        print(f"ep {ep:02d}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "state_dict": model.state_dict(),
                "val_dice": best_dice,
                "img_size": img_size,
                "epoch": ep,
            }, save_path)

    return TrainResult(best_val_dice=best_dice, history=history, save_path=str(save_path))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
_CACHE: dict[str, object] = {}


def get_pancreas_unet(ckpt_path: Optional[str | Path] = None, device: Optional[str] = None) -> dict:
    """Cached loader. Returns a dict with the model + img_size + device."""
    path = Path(ckpt_path) if ckpt_path else Path(config.PANCREAS_CKPT)
    key = str(path.resolve())
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    if not path.exists():
        raise FileNotFoundError(
            f"Pancreas UNet checkpoint not found at {path}. "
            f"Run train_pancreas_unet() first."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyUNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _CACHE[key] = {
        "model": model,
        "img_size": int(ckpt.get("img_size", 256)),
        "device": device,
        "val_dice": float(ckpt.get("val_dice", 0.0)),
    }
    return _CACHE[key]  # type: ignore[return-value]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    keep_idx = int(np.argmax(sizes)) + 1
    return (lbl == keep_idx).astype(np.uint8)


@torch.no_grad()
def infer_pancreas(
    image_np: np.ndarray,
    threshold: float = 0.5,
    dilation_px: int = 15,
    keep_largest: bool = True,
    ckpt_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> tuple[np.ndarray, Optional[list[float]]]:
    """Return (pancreas_mask HxW uint8, bbox [x1,y1,x2,y2] or None).

    `image_np` can be HxW grayscale or HxWx3 uint8 (as returned by medsam.load_image)."""
    unet = get_pancreas_unet(ckpt_path, device)
    model: TinyUNet = unet["model"]  # type: ignore[assignment]
    img_size: int = unet["img_size"]  # type: ignore[assignment]
    dev: str = unet["device"]  # type: ignore[assignment]

    if image_np.ndim == 3:
        img_gray = image_np[..., 0].astype(np.float32) / 255.0
    else:
        img_gray = image_np.astype(np.float32) / 255.0
    H, W = img_gray.shape

    img_rs = sk_resize(img_gray, (img_size, img_size), order=1,
                       preserve_range=True, anti_aliasing=True).astype(np.float32)
    x = torch.from_numpy(img_rs[None, None]).float().to(dev)
    logits = model(x).squeeze().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))

    mask_small = (prob > threshold).astype(np.uint8)
    mask = sk_resize(mask_small, (H, W), order=0, preserve_range=True,
                     anti_aliasing=False).astype(np.uint8)

    if keep_largest and mask.sum() > 0:
        mask = _largest_component(mask)

    if mask.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8), None

    if dilation_px > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_px).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask, None
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return mask, bbox


def reset_cache() -> None:
    _CACHE.clear()
