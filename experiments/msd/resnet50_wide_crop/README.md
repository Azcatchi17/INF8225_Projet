# Iteration 4 — Pretrained ResNet-50 + Wide Crop (Final)

The final pipeline. Two changes versus iteration 3, both motivated by
the failure analysis of the previous step.

1. **ResNet-50 ImageNet pretrained** replaces ResNet-18 from scratch.
   25M parameters instead of 11M, but more importantly the
   pretrained weights bring low-level texture and edge features that
   the classifier could not learn on 1500 medical patches.
2. **`crop_margin = 30 px`** instead of 10. A tumour vs. healthy
   parenchyma decision relies on the *contrast* between the two; with
   a 10 px margin the classifier saw almost no surrounding pancreas, at
   30 px about 20% of the patch is parenchyma, enough to compare.

The 3-slice channel layout from iteration 3 is preserved (it filters
isolated FPs without cost). All other hyper-parameters (mining schedule,
loss, augmentations, ensemble size) are unchanged.

## Pipeline

```bash
python -m experiments.msd.resnet50_wide_crop.extract_hard_negatives
python -m experiments.msd.resnet50_wide_crop.train_classifier
python -m experiments.msd.resnet50_wide_crop.calibrate_threshold
python -m experiments.msd.resnet50_wide_crop.evaluate
```

`BATCH_SIZE` drops from 32 to 16 to fit the heavier backbone on a T4.

## Final result on the test split

```
TP=38  FP=10  TN=40  FN=12
Sensitivity 0.76  Specificity 0.80  Precision 0.79  F1 0.78
AUC 0.82          AP 0.88            DICE moy. (TP) 0.50
```

`+9` points of F1 over the iter 1 cascade and over the iter 2/iter 3
classifier. Detailed analysis is in `report/main.tex` (sections 5.6 and
6).

## Reusing iteration 3

Iteration 4's `score.py` re-exports `score_candidates_3slice` from
iteration 3, and uses `slice_stack` from there as well. This keeps the
3-slice logic in one place; only the backbone and crop margin differ.

## Publication figures

`notebooks/msd/resnet50_wide_crop/05_publication_figures.ipynb`
generates the 6 figures used in the paper from
`data/results/dice_final_report_resnet50_wide_crop.csv`. Outputs land
in `report/figures/`.
