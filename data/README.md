# Data

Heavy assets live here at runtime but are gitignored. Only small
result CSVs (under `results/`) are tracked.

## Layout

```
data/
├── README.md            # this file (tracked)
├── raw/                 # gitignored: original MSD volumes, Kvasir images
├── processed/           # gitignored: 2D slices, COCO JSON, masks
├── results/             # tracked: small CSVs from evaluation runs
└── classifier_dataset_*/  # gitignored: per-iteration patch dumps
```

## Datasets

- **MSD-Pancreas** (Antonelli et al., 2022): 281 3D CT volumes with
  multi-class masks (background, pancreas, tumor). We sample 2D axial
  slices and split 800 train / 100 val / 100 test, balanced 50/50
  between tumor-positive and healthy slices.
- **Kvasir-SEG** (Jha et al., 2019): 1000 colon endoscopy images with
  polyp masks and bounding boxes. Used as a proof-of-concept for the
  detection+segmentation chain.

## How to populate

On Colab the `experiments/colab_setup.py` helper symlinks both datasets from a
shared Google Drive folder. Locally, place the data manually under
`data/MSD_pancreas/` and `data/Kvasir-SEG/` following the structure
expected by the COCO JSON files.

## What lives in `results/`

Test-set CSVs produced by each iteration's `evaluate.py`:

- `dice_final_report_resnet18_recall.csv`
- `dice_final_report_three_slice_context.csv`
- `dice_final_report_resnet50_wide_crop.csv` (final pipeline)

Plus calibration sweeps:
- `calibration_threshold_*.csv`
- `threshold_sweep_*.csv`
