# Iteration 2 — ResNet-18 Recall Classifier

Adds a learned classifier between DINO proposals and MedSAM
segmentation. Five ResNet-18 models are trained with seed variation on
crops mined aggressively at low DINO threshold (0.01, top-k=12) to
expose the classifier to the exact distribution of false positives it
needs to reject.

## Pipeline (run in order)

| Step | Module | Output |
|------|--------|--------|
| 0 | `calibrate_detector` | DINO threshold sweep on validation |
| 1 | `extract_hard_negatives` | `outputs/msd_implementation/resnet18_recall/datasets/classifier_dataset_resnet18/{train,val}/{0,1}/*.png` |
| 2 | `train_classifier` | `resnet18_recall_fold_{1..5}.pth` |
| 3 | `calibrate_threshold` | `outputs/msd_implementation/resnet18_recall/metrics/optimal_threshold_resnet18.{txt,json}` |
| 4 | `evaluate` | `outputs/msd_implementation/resnet18_recall/metrics/dice_final_report_resnet18_recall.csv` |

## Running from the repo root

```bash
python -m msd_implementation.pipelines.resnet18_recall.calibrate_detector
python -m msd_implementation.pipelines.resnet18_recall.extract_hard_negatives
python -m msd_implementation.pipelines.resnet18_recall.train_classifier
python -m msd_implementation.pipelines.resnet18_recall.calibrate_threshold
python -m msd_implementation.pipelines.resnet18_recall.evaluate
```

The same steps in Colab format are under
`msd_implementation/notebooks/resnet18_recall/`.

## Why this iteration matters

The cascade in iteration 1 saturates around F1 = 0.74; the spatial
prior alone cannot tell apart a tumor and its surrounding parenchyma.
A learned classifier introduces a texture-based signal that the
detector lacks. However its baseline ResNet-18 trained from scratch on
~1500 patches reaches the same F1 as the cascade: precision goes up,
sensitivity comes down, the trade-off shifts but does not improve. The
key analysis is in `msd_implementation/pipelines/common/RECALL_STRATEGY.md` and
motivates iterations 3 and 4.
