"""P2.2 + P2.3 helpers: box padding and mask post-processing.

Kept in the experiments folder so we can sweep parameters without touching
`experiments/msd/dino_gemini_msd/medsam.py` (which is already branched on
`config.POSTPROCESS_MASK`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# P2.2 — box padding
# ---------------------------------------------------------------------------
def pad_box(
    box: tuple[float, float, float, float],
    pad_frac: float,
    H: int,
    W: int,
) -> tuple[float, float, float, float]:
    """Grow a box by pad_frac of its own width / height, clipped to image.

    pad_frac=0.08 adds 8% of the box dimension on each side → 16% total.
    MedSAM tends to produce slightly tighter masks when the prompt box is a
    bit larger than the object (the model interprets the box as "region of
    interest", not "tight contour").
    """
    if pad_frac <= 0:
        return box
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    dx, dy = w * pad_frac, h * pad_frac
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(W - 1), x2 + dx),
        min(float(H - 1), y2 + dy),
    )


# ---------------------------------------------------------------------------
# P2.3 — mask post-processing
# ---------------------------------------------------------------------------
def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (8-connectivity)."""
    if mask is None or not mask.any():
        return mask
    lbl, n = ndimage.label(mask.astype(bool))
    if n <= 1:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask.astype(bool), lbl, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1  # labels are 1..n
    return (lbl == largest).astype(np.uint8)


def filter_by_size(
    mask: np.ndarray,
    min_area_px: int = 50,
    max_area_frac: float = 0.2,
) -> np.ndarray:
    """Zero out mask if total area is outside [min_area_px, max_area_frac * H*W].

    Returns the same mask when within bounds, else an all-zero mask. Used to
    kill both micro-spurs (MedSAM speckles on empty prompts) and the rare
    case where MedSAM segments the whole pancreas because the box was off.
    """
    if mask is None or not mask.any():
        return mask
    total = mask.size
    area = int(mask.sum())
    if area < min_area_px:
        return np.zeros_like(mask, dtype=np.uint8)
    if area > max_area_frac * total:
        return np.zeros_like(mask, dtype=np.uint8)
    return mask.astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    keep_largest: bool = True,
    min_area_px: int = 50,
    max_area_frac: float = 0.2,
) -> np.ndarray:
    """Full P2.3 chain: keep-largest → size filter. No-op branches when flags off."""
    out = mask
    if keep_largest:
        out = keep_largest_cc(out)
    if min_area_px > 0 or max_area_frac < 1.0:
        out = filter_by_size(out, min_area_px=min_area_px, max_area_frac=max_area_frac)
    return out
