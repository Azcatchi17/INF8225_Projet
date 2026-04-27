# MSD-Pancreas Iterations

Successive iterations of the detection-classification-segmentation
pipeline applied to the Medical Segmentation Decathlon pancreas task.
Each iteration is motivated by the failure modes observed at the
previous step.

| Order | Folder | Title | Key result |
|-------|--------|-------|------------|
| 1 | `dino_medsam_cascade/` | DINO + MedSAM with anatomical cascade | F1 = 0.74 |
| 2 | `resnet18_recall/` | ResNet-18 ensemble with hard negative mining | F1 = 0.69 (precision +) |
| 3 | `three_slice_context/` | + 3-slice channel input | F1 = 0.68 (specificity 0.96) |
| 4 | `resnet50_wide_crop/` | ResNet-50 pretrained + crop_margin=30 | **F1 = 0.78** |

`pipelines/common/proposal_strategy.py` contains the candidate generation
and ensemble scoring logic reused across iterations 2-4. Iteration 1 is
self-contained and does not depend on the shared module.

## Running

All iterations expect the repo root as working directory:

```bash
cd /path/to/INF8225_Projet
python -m msd_implementation.pipelines.<iteration>.<step>
```

Steps for the recall-oriented iterations (2-4):

```text
1. extract_hard_negatives   # mine patches into data/classifier_dataset_*/
2. train_classifier         # train 5x ResNet folds
3. calibrate_threshold      # F2 sweep on validation
4. evaluate                 # final test report
```

Iteration 1 has a single `evaluation.py` runner that performs the full
detect-segment-evaluate loop in one pass.

## Artefacts produced (per iteration)

| Iteration | Patches | Checkpoints | Threshold | Test CSV |
|-----------|---------|-------------|-----------|----------|
| 1 | _none_ | _none_ | _none_ | inline |
| 2 | `data/classifier_dataset_resnet18/` | `resnet18_recall_fold_{1..5}.pth` | `optimal_threshold_resnet18.{txt,json}` | `dice_final_report_resnet18_recall.csv` |
| 3 | `data/classifier_dataset_three_slice/` | `three_slice_fold_{1..5}.pth` | `optimal_threshold_three_slice_context.{txt,json}` | `dice_final_report_three_slice_context.csv` |
| 4 | `data/classifier_dataset_resnet50_wide_crop/` | `resnet50_wide_crop_fold_{1..5}.pth` | `optimal_threshold_resnet50_wide_crop.{txt,json}` | `dice_final_report_resnet50_wide_crop.csv` |
