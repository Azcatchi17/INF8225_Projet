# Notebooks

Colab-friendly orchestration notebooks. Each notebook clones the repo
on its target machine, syncs the heavy assets from Drive, and invokes
one iteration script via `python -m experiments.msd.<iter>.<step>`.

## Layout

```
notebooks/
├── kvasir/                     # Polyp proof-of-concept notebooks
├── msd/                        # MSD-Pancreas iterations (one folder per iter)
│   ├── dino_medsam_cascade/    # Iter 1: baseline cascade
│   ├── resnet18_recall/        # Iter 2: ResNet-18 ensemble
│   ├── three_slice_context/    # Iter 3: 3-slice channel input
│   └── resnet50_wide_crop/     # Iter 4: ResNet-50 + crop=30 (final, with 05_publication_figures.ipynb)
└── exploratory/                # Older notebooks kept for reproducibility
```

## Per-iteration files (MSD)

Each iteration folder under `notebooks/msd/` contains the same numbered
sequence:

```
00_calibrate_detector.ipynb     # only iter 2 (initial DINO threshold sweep)
01_extract_hard_negatives.ipynb
02_train_classifier.ipynb
03_calibrate_threshold.ipynb
04_evaluate.ipynb
05_publication_figures.ipynb    # only the final iteration
```

The 1:1 mapping between notebook and source script makes it easy to
audit what every cell does — the Python source lives in
`experiments/msd/<iter>/`.

## Exploratory notebooks

These predate the academic restructure and are kept for traceability:

- `agentic_segmentation_msd.ipynb` — the failed Gemini-agent loop.
- `convert_to_coco.ipynb`, `create_pancreas_dataset.ipynb` — early
  dataset preparation.
- `initial_grounding_dino_test.ipynb`, `test_gd*.ipynb` — early DINO
  fine-tuning sandboxes.
- `segment_msd_oracle.ipynb` — MedSAM oracle benchmark on MSD.
