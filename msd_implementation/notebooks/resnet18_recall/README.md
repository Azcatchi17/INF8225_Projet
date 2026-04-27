# Colab MSD Recall Pipeline

This folder mirrors the TamIA pipeline steps `00` through `04` as Colab notebooks.
Each notebook runs the same root Python script used by the corresponding TamIA Slurm job:

| Notebook | Root script |
|---|---|
| `00_calibrate_dino.ipynb` | `calibrate_dino.py` |
| `01_extract_hard_negatives.ipynb` | `extract_hard_negatives.py` |
| `02_train_resnet.ipynb` | `train_resnet_kfold_recall.py` |
| `03_calibrate_threshold.ipynb` | `calibrate_threshold.py` |
| `04_test_recall.ipynb` | `test_gd_msd_final_recall.py` |

Recommended order:

1. Open `00_calibrate_dino.ipynb` in Colab with GPU runtime.
2. Run all cells.
3. Continue with notebooks `01` to `04` in order.

Notes:

- The setup cell clones/pulls `GIT_REF = "main"` into `/content/INF8225_Projet` by default.
- `colab/setup.py` mounts Drive, installs the OpenMMLab stack, and symlinks heavy assets.
- Keep `INSTALL_DEPS=True` for the first notebook in a fresh runtime. You can set it to `False` in later notebooks if the same runtime is still alive.
- The hard-negative dataset must be regenerated with notebook `01` for the recall strategy; do not reuse an older `classifier_dataset_hard` extracted with the previous top-1 pipeline.
