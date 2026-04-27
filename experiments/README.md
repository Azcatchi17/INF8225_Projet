# Experiments

Source code (no notebooks) for every experimental iteration. Notebooks
that orchestrate these scripts on Colab live under `../notebooks/`.

## Layout

```
experiments/
├── kvasir/                          # Polyp segmentation proof-of-concept
└── msd/                             # MSD-Pancreas tumour pipeline
    ├── _shared/                     # Common utilities (proposal logic)
    ├── dino_medsam_cascade/         # Iter 1: DINO + MedSAM with anatomical cascade
    ├── resnet18_recall/             # Iter 2: ResNet-18 ensemble + hard negative mining
    ├── three_slice_context/         # Iter 3: 3-slice channel input
    └── resnet50_wide_crop/          # Iter 4: ResNet-50 pretrained + crop_margin=30
```

The chronological order of MSD iterations corresponds to the ablation
table reported in the paper (`report/main.tex`, Table 1). Each iteration
produces its own checkpoints and threshold artefact so they can coexist
without interfering.

## Running an iteration

All scripts are designed to be invoked as Python modules from the repo
root, after the heavy assets have been symlinked into `data/` and
`work_dirs/` (see `colab/setup.py`).

```bash
# From the repo root
python -m experiments.msd.resnet50_wide_crop.extract_hard_negatives
python -m experiments.msd.resnet50_wide_crop.train_classifier
python -m experiments.msd.resnet50_wide_crop.calibrate_threshold
python -m experiments.msd.resnet50_wide_crop.evaluate
```

The matching Colab notebooks live in
`notebooks/msd/resnet50_wide_crop/`.

## Module dependency graph

`_shared/proposal_strategy.py` is the single source of truth for DINO
candidate filtering, NMS, and ResNet ensemble scoring. Every iteration
imports from it.

`three_slice_context.score` and `three_slice_context.slice_stack` are
shared by iterations 3 and 4 (iter 4 reuses the 3-slice stacking logic
unchanged, only the backbone and crop margin differ).
