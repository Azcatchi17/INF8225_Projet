"""3-slice channel stacking utilities.

For a given image path of the form
``.../<split>/images/pancreas_<id>_s<slice>.png``, we build a 3-channel
crop where each channel comes from a different slice of the same scan:

    channel 0 = slice - 1
    channel 1 = slice + 0  (current)
    channel 2 = slice + 1

If a strict neighbour (delta=+/-1) is not present on disk, we look at
+/-2, then +/-3 (configurable via ``MAX_NEIGHBOR_GAP``). If even that
fails, we fall back to the current slice itself, so the model receives a
duplicated channel rather than zero padding. This means the worst case
behaves exactly like the previous 2D pipeline.

Available slices are not contiguous in the MSD-pancreas 2D dataset
(annotated slices are sub-sampled), so the search must be permissive.
The current slice is always located on disk; neighbours are looked up
across all three split folders (``train``, ``val``, ``test``) because the
splits are by image, not by patient, and a neighbouring slice may live
in a different split.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from skimage import io

from experiments.msd._shared.proposal_strategy import clip_box, expand_box

_FILENAME_RE = re.compile(r"pancreas_(\d+)_s(\d+)\.png$")
_SPLIT_DIRS = ("train", "val", "test")
MAX_NEIGHBOR_GAP = 3  # search up to +/-3 slices before falling back


def parse_slice(image_path: str | Path) -> Optional[tuple[int, int]]:
    """Return (patient_id, slice_index) parsed from the file name, or None."""
    name = Path(image_path).name
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _msd_root(image_path: Path) -> Optional[Path]:
    """Walk up to find the ``MSD_pancreas`` root containing train/val/test/images."""
    p = Path(image_path).resolve()
    while p.parent != p:
        if p.name == "MSD_pancreas":
            return p
        p = p.parent
    return None


def find_neighbor_path(image_path: str | Path, delta: int) -> Optional[Path]:
    """Return the on-disk path for slice + delta of the same patient.

    Searches across train/val/test image folders. Returns None if the
    neighbour does not exist anywhere.
    """
    parsed = parse_slice(image_path)
    if parsed is None:
        return None
    patient, slice_idx = parsed
    target = slice_idx + delta
    if target < 0:
        return None

    msd = _msd_root(Path(image_path))
    if msd is None:
        # fallback: same parent directory only
        candidate = Path(image_path).parent / f"pancreas_{patient:03d}_s{target}.png"
        return candidate if candidate.exists() else None

    for split in _SPLIT_DIRS:
        candidate = msd / split / "images" / f"pancreas_{patient:03d}_s{target}.png"
        if candidate.exists():
            return candidate
    return None


def find_best_neighbor(
    image_path: str | Path, direction: int, max_gap: int = MAX_NEIGHBOR_GAP
) -> Optional[Path]:
    """Find the closest existing neighbour in the requested direction.

    direction = -1 (previous) or +1 (next). Tries delta = direction*1, *2, ..., *max_gap.
    """
    assert direction in (-1, 1)
    for k in range(1, max_gap + 1):
        candidate = find_neighbor_path(image_path, direction * k)
        if candidate is not None:
            return candidate
    return None


def _load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as a uint8 (H, W) array. MSD PNGs are 3-channel grayscale."""
    img = io.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def stack_3slice_image(image_path: str | Path) -> np.ndarray:
    """Return an (H, W, 3) uint8 stack of [prev, curr, next] for the whole image.

    Falls back to the current slice for any missing neighbour.
    """
    image_path = Path(image_path)
    curr = _load_grayscale(image_path)

    prev_path = find_best_neighbor(image_path, direction=-1)
    next_path = find_best_neighbor(image_path, direction=+1)

    prev = _load_grayscale(prev_path) if prev_path is not None else curr
    nxt = _load_grayscale(next_path) if next_path is not None else curr

    if prev.shape != curr.shape:
        prev = curr
    if nxt.shape != curr.shape:
        nxt = curr

    return np.stack([prev, curr, nxt], axis=-1).astype(np.uint8)


def stack_3slice_crop(
    image_path: str | Path,
    box: Iterable[float],
    margin: int,
) -> Optional[np.ndarray]:
    """Crop the same box from {prev, curr, next} slices and stack as channels.

    Returns an (H, W, 3) uint8 array, or None if the box is empty.
    """
    img_3slice = stack_3slice_image(image_path)
    h, w = img_3slice.shape[:2]
    box_clipped = clip_box(box, w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in expand_box(box_clipped, margin, w, h)]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_3slice[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
