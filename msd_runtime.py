"""Runtime paths for local, Colab and TamIA/Slurm executions."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


MSD_DATA_DIR = _path_from_env("MSD_DATA_DIR", PROJECT_ROOT / "data" / "MSD_pancreas")
CLASSIFIER_DATASET_DIR = _path_from_env(
    "MSD_CLASSIFIER_DATASET_DIR",
    PROJECT_ROOT / "data" / "classifier_dataset_hard",
)
RESULTS_DIR = _path_from_env("MSD_RESULTS_DIR", PROJECT_ROOT / "data" / "results")
RESNET_DIR = _path_from_env(
    "MSD_RESNET_DIR",
    os.environ.get("RESNET_CHECKPOINT_DIR", os.environ.get("INF8225_DRIVE_ROOT", PROJECT_ROOT)),
)

DINO_CONFIG_PATH = _path_from_env(
    "DINO_CONFIG_PATH",
    PROJECT_ROOT / "work_dirs" / "tumor_config_v3" / "tumor_config_v3.py",
)
DINO_CHECKPOINT_PATH = _path_from_env(
    "DINO_CHECKPOINT_PATH",
    PROJECT_ROOT / "work_dirs" / "tumor_config_v3" / "best_coco_bbox_mAP_epoch_25.pth",
)
MEDSAM_CHECKPOINT_PATH = _path_from_env(
    "MEDSAM_CHECKPOINT_PATH",
    PROJECT_ROOT / "MedSAM" / "work_dir" / "MedSAM" / "medsam_vit_b.pth",
)

OPTIMAL_THRESHOLD_TXT = _path_from_env("OPTIMAL_THRESHOLD_TXT", PROJECT_ROOT / "optimal_threshold.txt")
OPTIMAL_THRESHOLD_JSON = _path_from_env("OPTIMAL_THRESHOLD_JSON", PROJECT_ROOT / "optimal_threshold.json")


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESNET_DIR.mkdir(parents=True, exist_ok=True)


def resnet_path(run: int) -> Path:
    return RESNET_DIR / f"resnet_fold_{run}.pth"


def resnet_threshold_path(run: int) -> Path:
    return RESNET_DIR / f"threshold_run_{run}.txt"
