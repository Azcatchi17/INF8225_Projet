# Colab MSD Recall — variante 3-slice v2

Pipeline parallèle à `colab_three_slice/` avec deux changements :

1. **Backbone ResNet-50** ImageNet pretrained (au lieu de ResNet-18) → +14M params, features pré-entraînées plus riches.
2. **`crop_margin=30`** (au lieu de 10) → le classifieur voit le pancréas autour de la box, indispensable pour distinguer "tumeur dans pancréas" vs "parenchyme normal".

Le 3-slice (canaux `(prev, curr, next)`) est conservé : il a déjà prouvé son intérêt sur les FP en v1.

## Notebooks

| Notebook | Script appelé | Sortie |
|---|---|---|
| `01_extract_v2.ipynb` | `experiments.three_slice_v2.extract_v2` | `data/classifier_dataset_3slice_v2/{train,val}/{0,1}/*.png` |
| `02_train_v2.ipynb` | `experiments.three_slice_v2.train_v2` | `resnet50_3slice_fold_{1..5}.pth` (sur Drive) |
| `03_calibrate_v2.ipynb` | `experiments.three_slice_v2.calibrate_v2` | `optimal_threshold_3slice_v2.{txt,json}` |
| `04_test_v2.ipynb` | `experiments.three_slice_v2.test_v2` | `data/results/dice_final_report_msd_recall_3slice_v2.csv` |

## Ordre recommandé

1. `01_extract_v2.ipynb` (~10-15 min, IO-bound, le crop large quadruple à peu près le poids des PNG par rapport à la v1)
2. `02_train_v2.ipynb` (~120 min sur T4 — ResNet-50 est ~3× plus lent que ResNet-18)
3. `03_calibrate_v2.ipynb` (~5 min)
4. `04_test_v2.ipynb` (~5 min)

## Notes

- Le DINO n'est PAS réentraîné, on réutilise `work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth`.
- Les checkpoints v2 (`resnet50_3slice_fold_*.pth`) cohabitent avec ceux de v1 (`resnet_3slice_fold_*.pth`) et 2D (`resnet_fold_*.pth`).
- `BATCH_SIZE` est passé de 32 à 16 pour ResNet-50 (sinon OOM possible sur T4 avec batchnorm + autograd).
- Si la calibration choisit un seuil trop conservateur (recall test < 0,7), recalibrer avec `MIN_RECALL=0.95` dans `calibrate_v2.py`.

## Comparaison

Une fois `04` terminé, compare les 3 CSV :
- `dice_final_report_msd_recall.csv` (2D baseline)
- `dice_final_report_msd_recall_3slice.csv` (3-slice v1)
- `dice_final_report_msd_recall_3slice_v2.csv` (3-slice v2)
