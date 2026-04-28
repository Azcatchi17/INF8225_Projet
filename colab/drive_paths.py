"""Shared Drive/output layout helpers for Colab notebooks.

The notebooks should write every generated artifact under:

    Projet_Medsam/outputs/<implementation>/<pipeline>/

In Colab, ``colab.setup.setup()`` symlinks the local ``outputs/`` folder to
that Drive location. Locally, these helpers fall back to ``<repo>/outputs``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVE_CANDIDATES = [
    Path("/content/drive/Shareddrives/Projet_Medsam"),
    Path("/content/drive/Shareddrives/INF8225_Projet"),
    Path("/content/drive/Shareddrives/INF8225"),
    Path("/content/drive/MyDrive/Projet_Medsam"),
]

DATA_DIRS = [
    "data/Kvasir-SEG",
    "data/MSD_pancreas",
]

WORK_DIRS = [
    "work_dirs/polyp_config_v2",
    "work_dirs/tumor_config_v3",
    "work_dirs/pancreas_unet",
    "work_dir/MedSAM",
    "models_weights",
]

REQUIRED_FILES = [
    "work_dir/MedSAM/medsam_vit_b.pth",
    "models_weights/grounding_dino_swin-t_pretrain_obj365_goldg.py",
    "models_weights/grounding_dino_swin-t_pretrain_obj365_goldg_*.pth",
    "work_dirs/polyp_config_v2/best_coco_bbox_mAP_epoch_5.pth",
    "work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth",
    "work_dirs/pancreas_unet/best.pt",
]

OUTPUT_PIPELINES = {
    "kvasir_implementation": [
        "oracle_baseline_kvasir",
        "calibrate_dino_threshold_kvasir",
        "test_kvasir",
    ],
    "msd_implementation": [
        "dino_calibration",
        "dino_medsam_cascade",
        "dino_medsam_gemini",
        "resnet18_recall",
        "three_slice_context",
        "resnet50_wide_crop",
    ],
}

STANDARD_OUTPUT_SUBDIRS = [
    "metrics",
    "figures",
    "masks",
    "checkpoints",
    "datasets",
    "logs",
    "cache",
    "runs",
]


def _drive_from_symlink() -> Path | None:
    for local_name in ("outputs", "data", "work_dirs"):
        local_path = REPO_ROOT / local_name
        if local_path.is_symlink():
            resolved = local_path.resolve()
            if resolved.name == local_name:
                return resolved.parent
    return None


def drive_root() -> Path:
    """Return the project Drive root, or the repo root outside Colab."""
    env_root = os.environ.get("INF8225_DRIVE_ROOT")
    if env_root:
        return Path(env_root)

    symlink_root = _drive_from_symlink()
    if symlink_root is not None:
        return symlink_root

    for candidate in DRIVE_CANDIDATES:
        if candidate.exists():
            return candidate

    return REPO_ROOT


def outputs_root(create: bool = True) -> Path:
    """Return the persistent outputs root."""
    env_root = os.environ.get("INF8225_OUTPUTS_ROOT")
    root = Path(env_root) if env_root else drive_root() / "outputs"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def output_dir(
    implementation: str,
    pipeline: str,
    *parts: str,
    create: bool = True,
) -> Path:
    """Return an output path under outputs/<implementation>/<pipeline>/."""
    path = outputs_root(create=create) / implementation / pipeline
    for part in parts:
        path = path / part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _pipeline_manifest() -> dict[str, list[str]]:
    return {
        implementation: [
            f"outputs/{implementation}/{pipeline}"
            for pipeline in pipelines
        ]
        for implementation, pipelines in OUTPUT_PIPELINES.items()
    }


def write_manifest(root: Path | None = None) -> Path:
    """Write a compact manifest documenting the expected Drive layout."""
    root = Path(root) if root is not None else drive_root()
    manifest = {
        "project": "Projet_Medsam",
        "layout_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": DATA_DIRS,
        "work_dirs": WORK_DIRS,
        "required_files": REQUIRED_FILES,
        "outputs": _pipeline_manifest(),
        "standard_output_subdirs": STANDARD_OUTPUT_SUBDIRS,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_drive_layout(root: Path | None = None) -> Path:
    """Create the expected Drive folder tree and return its root."""
    root = Path(root) if root is not None else drive_root()

    for rel in [*DATA_DIRS, *WORK_DIRS]:
        (root / rel).mkdir(parents=True, exist_ok=True)

    for implementation, pipelines in OUTPUT_PIPELINES.items():
        for pipeline in pipelines:
            pipeline_root = root / "outputs" / implementation / pipeline
            pipeline_root.mkdir(parents=True, exist_ok=True)
            for subdir in STANDARD_OUTPUT_SUBDIRS:
                (pipeline_root / subdir).mkdir(parents=True, exist_ok=True)

    write_manifest(root)
    return root
