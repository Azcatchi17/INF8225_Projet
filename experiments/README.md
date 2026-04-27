# Experiments

Source code for every experimental iteration, plus the Colab bootstrap
helper used by the notebooks. Lightweight orchestration notebooks live
under `../notebooks/`; the archived full MSD improved-pipeline run lives
next to its iteration code.

## Layout

```
experiments/
├── colab_setup.py                   # Colab bootstrap and Drive symlinks
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
`work_dirs/` (see `experiments/colab_setup.py`).

```bash
# From the repo root
python -m experiments.msd.resnet50_wide_crop.extract_hard_negatives
python -m experiments.msd.resnet50_wide_crop.train_classifier
python -m experiments.msd.resnet50_wide_crop.calibrate_threshold
python -m experiments.msd.resnet50_wide_crop.evaluate
```

The matching Colab notebooks live in
`notebooks/msd/resnet50_wide_crop/`.

The preserved full improved-pipeline notebook is stored at
`experiments/msd/dino_medsam_cascade/improved_pipeline.ipynb`.

## Module dependency graph

`_shared/proposal_strategy.py` is the single source of truth for DINO
candidate filtering, NMS, and ResNet ensemble scoring. Every iteration
imports from it.

`three_slice_context.score` and `three_slice_context.slice_stack` are
shared by iterations 3 and 4 (iter 4 reuses the 3-slice stacking logic
unchanged, only the backbone and crop margin differ).
