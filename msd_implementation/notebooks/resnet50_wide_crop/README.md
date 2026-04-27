# Colab MSD Recall — variante 3-slice v2

Pipeline parallèle à `colab_three_slice/` avec deux changements :

1. **Backbone ResNet-50** ImageNet pretrained (au lieu de ResNet-18) → +14M params, features pré-entraînées plus riches.
2. **`crop_margin=30`** (au lieu de 10) → le classifieur voit le pancréas autour de la box, indispensable pour distinguer "tumeur dans pancréas" vs "parenchyme normal".

Le 3-slice (canaux `(prev, curr, next)`) est conservé : il a déjà prouvé son intérêt sur les FP en v1.

## Notebooks

| Notebook | Script appelé | Sortie |
|---|---|---|
| `01_extract_hard_negatives.ipynb` | `msd_implementation.pipelines.resnet50_wide_crop.extract_hard_negatives` | `outputs/msd_implementation/resnet50_wide_crop/datasets/classifier_dataset_resnet50_wide_crop/{train,val}/{0,1}/*.png` |
| `02_train_classifier.ipynb` | `msd_implementation.pipelines.resnet50_wide_crop.train_classifier` | `outputs/msd_implementation/resnet50_wide_crop/checkpoints/resnet50_wide_crop_fold_{1..5}.pth` |
| `03_calibrate_threshold.ipynb` | `msd_implementation.pipelines.resnet50_wide_crop.calibrate_threshold` | `outputs/msd_implementation/resnet50_wide_crop/metrics/optimal_threshold_resnet50_wide_crop.{txt,json}` |
| `04_evaluate.ipynb` | `msd_implementation.pipelines.resnet50_wide_crop.evaluate` | `outputs/msd_implementation/resnet50_wide_crop/metrics/dice_final_report_resnet50_wide_crop.csv` |

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
- `outputs/msd_implementation/resnet18_recall/metrics/dice_final_report_resnet18_recall.csv`
- `outputs/msd_implementation/three_slice_context/metrics/dice_final_report_three_slice_context.csv`
- `outputs/msd_implementation/resnet50_wide_crop/metrics/dice_final_report_resnet50_wide_crop.csv`
