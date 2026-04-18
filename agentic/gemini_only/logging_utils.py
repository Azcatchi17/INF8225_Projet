"""Same as dino_gemini.logging_utils but writes under agent_{runs,results,cache}_vlm."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from agentic.dino_gemini.state import AgentState


def _run_dir(run_id: str) -> Path:
    d = config.AGENT_RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_iteration(state: AgentState, iteration_idx: int) -> Path:
    d = _run_dir(state.run_id)
    it = state.iterations[iteration_idx]
    path = d / f"iter_{it.iteration:02d}.json"
    payload = {"run_id": state.run_id,
               "image_path": state.image_path,
               **it.to_dict()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_run_summary(state: AgentState) -> Path:
    d = _run_dir(state.run_id)
    path = d / "run.json"
    state.to_json(path)
    return path


def save_masks(state: AgentState) -> list[Path]:
    config.AGENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    img_stem = Path(state.image_path).stem
    paths = []
    for it in state.iterations:
        if it.mask is None:
            continue
        out = config.AGENT_RESULTS_DIR / f"mask_{img_stem}_iter_{it.iteration:02d}.png"
        Image.fromarray((it.mask * 255).astype(np.uint8)).save(out)
        paths.append(out)
    return paths


def pickle_state(state: AgentState) -> Path:
    config.AGENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.AGENT_CACHE_DIR / f"{state.run_id}.pkl"
    snapshot = {
        "state": state.to_dict(),
        "iteration_masks": [
            (it.iteration, it.mask) for it in state.iterations if it.mask is not None
        ],
    }
    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    return path


def log_run(state: AgentState) -> dict:
    run_json = write_run_summary(state)
    for i in range(len(state.iterations)):
        write_iteration(state, i)
    mask_paths = save_masks(state)
    pkl = pickle_state(state)
    return {
        "run_json": str(run_json),
        "masks": [str(p) for p in mask_paths],
        "pickle": str(pkl),
    }
