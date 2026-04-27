# Scripts

Standalone utilities that are not part of any single iteration.

## Layout

| Folder | Purpose |
|--------|---------|
| `data/` | One-off dataset preparation helpers (MSD slice extraction, COCO conversion, label-map creation, patch utilities). |
| `train/` | Reserved for cross-iteration training helpers (currently empty; iteration-specific trainers live under `experiments/msd/<iter>/train_classifier.py`). |
| `evaluate/` | Standalone runners and legacy testers kept for reproducibility. |
| `figures/` | Reserved for figure-generation scripts that do not belong to a notebook. |

## Notable scripts

- `scripts/data/split_msd_annotations.py` — re-splits the COCO JSON of MSD
  pancreas into train/val/test with balanced tumor/healthy slices.
- `scripts/data/create_label_map.py` — builds the label map JSON
  consumed by `RandomSamplingNegPos` during DINO training.
- `scripts/data/extract_patches.py` — dataset-prep helper used during the
  early experiments.
- `scripts/evaluate/legacy_test_msd_no_resnet.py` — the original
  cascade-only evaluator from before the ResNet verifier; kept as a
  reproducibility record. The current cascade lives at
  `experiments/msd/dino_medsam_cascade/evaluation.py`.
- `scripts/evaluate/msd_runtime.py` — runtime helpers for the agentic
  segmentation experiments (preserved for completeness).
