# Colab MSD Recall — variante 3-slice

Pipeline parallèle à `colab_msd_recall/`, où chaque patch utilisé par le ResNet est un stack 3 canaux `(slice-1, slice, slice+1)` au lieu d'une grayscale 2D répliquée. Le but est de donner au classifieur du contexte spatial sans introduire de regroupement par patient à l'inférence.

## Notebooks

| Notebook | Script appelé | Sortie |
|---|---|---|
| `01_extract_3slice.ipynb` | `experiments.three_slice.extract_hard_negatives_3slice` | `data/classifier_dataset_3slice/{train,val}/{0,1}/*.png` |
| `02_train_3slice.ipynb` | `experiments.three_slice.train_resnet_3slice` | `resnet_3slice_fold_{1..5}.pth` (sur Drive) |
| `03_calibrate_3slice.ipynb` | `experiments.three_slice.calibrate_threshold_3slice` | `optimal_threshold_3slice.{txt,json}` |
| `04_test_3slice.ipynb` | `experiments.three_slice.test_recall_3slice` | `data/results/dice_final_report_msd_recall_3slice.csv` |

## Ordre recommandé

1. Ouvre `01_extract_3slice.ipynb` dans Colab avec une GPU runtime, exécute toutes les cellules.
2. Continue avec `02`, `03`, puis `04` dans le même runtime.

## Notes

- Les checkpoints 3-slice cohabitent avec les `resnet_fold_*.pth` 2D existants (préfixe différent).
- Le détecteur DINO n'est PAS réentraîné — on réutilise `work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth`. Pas de notebook `00` car la calibration DINO n'a pas changé.
- Les voisins de slices manquants (s±1 absent du dataset) sont remplacés par s±2 puis s±3, et en dernier recours par la slice courante elle-même. Dans ce cas extrême, le comportement dégénère vers le pipeline 2D original.
- Le seuil ResNet calibré peut être différent du 2D (typiquement 0.30–0.45 suivant la qualité des features 3-slice apprises) — c'est attendu, ne pas réutiliser `optimal_threshold.txt` du 2D.

## Comparaison

Une fois `04` terminé, compare `data/results/dice_final_report_msd_recall.csv` (2D) et `data/results/dice_final_report_msd_recall_3slice.csv` (3-slice) — par exemple avec un script local ou un notebook séparé.
