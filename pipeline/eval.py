"""Batch evaluation + one-shot vs agentic comparison."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from . import config, grounding_dino as gd, medsam, metrics as M
from .agent import run_agent
from .state import AgentState


def _load_gt(file_name: str) -> np.ndarray:
    mask = io.imread(config.KVASIR_MASKS / file_name)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _one_shot(img_path: Path, gt: np.ndarray) -> tuple[float, int]:
    """Baseline: single DINO → MedSAM pass, no Gemma, no refinement."""
    image_np = medsam.load_image(str(img_path))
    H, W = image_np.shape[:2]
    boxes = gd.detect(str(img_path), config.DEFAULT_DINO_TEXT)
    if not boxes:
        return 0.0, 0
    embed = medsam.encode_image(image_np)
    mask = medsam.segment(embed, H=H, W=W, box=list(boxes[0].xyxy))
    return M.dice(mask, gt), 1


def run_batch(
    test_json: str | Path = config.KVASIR_TEST_JSON,
    user_text: str = "find the polyp",
    n: Optional[int] = None,
    max_iter: int = config.MAX_ITER,
) -> pd.DataFrame:
    with open(test_json) as f:
        test = json.load(f)
    images = test["images"]
    if n is not None:
        images = images[:n]

    rows = []
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    for img_info in tqdm(images, desc="agentic"):
        file_name = img_info["file_name"]
        img_path = config.KVASIR_IMAGES / file_name
        if not img_path.exists():
            continue
        gt = _load_gt(file_name)
        state = run_agent(str(img_path), user_text, max_iter=max_iter, gt_mask=gt)
        best = state.best_iter()
        last = state.iterations[-1] if state.iterations else None
        rows.append({
            "image_id": img_info["id"],
            "file_name": file_name,
            "n_iterations": len(state.iterations),
            "stop_reason": state.stop_reason,
            "first_iter_dice": state.iterations[0].dice_vs_gt if state.iterations else 0.0,
            "last_iter_dice": last.dice_vs_gt if last else 0.0,
            "best_iter": best.iteration if best else -1,
            "final_dice": best.dice_vs_gt if best else 0.0,
        })

    df = pd.DataFrame(rows)
    config.AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.AGENT_RESULTS_DIR / f"summary_{run_id}.csv", index=False)
    return df


def compare_oneshot_vs_agentic(
    test_json: str | Path = config.KVASIR_TEST_JSON,
    user_text: str = "find the polyp",
    n: int = 50,
    max_iter: int = config.MAX_ITER,
) -> pd.DataFrame:
    """Run both pipelines on the same images, save a CSV + a bar-chart PNG."""
    import matplotlib.pyplot as plt

    with open(test_json) as f:
        test = json.load(f)
    images = test["images"][:n]

    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    rows = []
    for img_info in tqdm(images, desc="compare"):
        file_name = img_info["file_name"]
        img_path = config.KVASIR_IMAGES / file_name
        if not img_path.exists():
            continue
        gt = _load_gt(file_name)

        one_shot_dice, _ = _one_shot(img_path, gt)
        state: AgentState = run_agent(
            str(img_path), user_text, max_iter=max_iter, gt_mask=gt
        )
        best = state.best_iter()
        last = state.iterations[-1] if state.iterations else None
        final_dice = best.dice_vs_gt if best else 0.0

        rows.append({
            "image_id": img_info["id"],
            "file_name": file_name,
            "one_shot_dice": one_shot_dice,
            "agentic_dice": final_dice,
            "last_iter_dice": last.dice_vs_gt if last else 0.0,
            "dice_delta": final_dice - one_shot_dice,
            "n_iterations": len(state.iterations),
            "best_iter": best.iteration if best else -1,
            "stop_reason": state.stop_reason,
        })

    df = pd.DataFrame(rows)
    config.AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.AGENT_RESULTS_DIR / f"comparison_{run_id}.csv"
    df.to_csv(csv_path, index=False)

    # Per-image DICE bar chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(len(df))
    axes[0].bar(x - 0.2, df["one_shot_dice"], width=0.4, label="one-shot")
    axes[0].bar(x + 0.2, df["agentic_dice"], width=0.4, label="agentic")
    axes[0].set_ylabel("DICE")
    axes[0].set_title(
        f"One-shot vs agentic (n={len(df)})   "
        f"mean: {df['one_shot_dice'].mean():.3f} → {df['agentic_dice'].mean():.3f}"
    )
    axes[0].legend()
    axes[1].bar(x, df["n_iterations"], color="tab:gray")
    axes[1].set_ylabel("# iterations")
    axes[1].set_xlabel("image index")
    plt.tight_layout()
    png_path = config.AGENT_RESULTS_DIR / f"comparison_{run_id}.png"
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"saved {csv_path}")
    print(f"saved {png_path}")
    return df
