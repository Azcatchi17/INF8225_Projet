# Iteration 3 — 3-Slice Channel Context

The patch passed to the classifier becomes a stack of three adjacent
slices `(slice-1, slice, slice+1)` packed as the RGB channels of a
single image. The hypothesis is that real tumours produce coherent
texture across the three channels (3D continuity) while DINO
hallucinations are sporadic and only appear on a single slice.

When a strict neighbour is missing in the dataset (annotated MSD slices
are not contiguous), `slice_stack.find_best_neighbor` falls back to
±2 then ±3 before duplicating the current slice as the last resort,
yielding a graceful degradation toward the 2D pipeline.

## Modules

| File | Purpose |
|------|---------|
| `slice_stack.py` | Slice neighbour lookup and 3-channel crop builder. |
| `score.py` | Scoring function that crops the same box from the stacked image and runs the ensemble. |
| `extract_hard_negatives.py`, `train_classifier.py`, `calibrate_threshold.py`, `evaluate.py` | Same role as in iter 2, with 3-slice patches throughout. |

## Pipeline

```bash
python -m msd_implementation.pipelines.three_slice_context.extract_hard_negatives
python -m msd_implementation.pipelines.three_slice_context.train_classifier
python -m msd_implementation.pipelines.three_slice_context.calibrate_threshold
python -m msd_implementation.pipelines.three_slice_context.evaluate
```

## What changed vs iter 2

| Aspect | iter 2 | iter 3 |
|--------|--------|--------|
| Patch channels | grayscale replicated 3x | (prev, curr, next) slices |
| Backbone | ResNet-18 from scratch | same |
| Crop margin | 10 px | 10 px |
| Dataset folder | `classifier_dataset_resnet18` | `classifier_dataset_three_slice` |

## Outcome

Specificity jumps from 0.82 to 0.96 (FP drops from 9 to 2), but the
sensitivity falls from 0.62 to 0.54. The temporal context successfully
filters isolated hallucinations but does not help patients whose tumour
has a weak signature in 2D — those tumours have an equally weak
signature in 3D. F1 stays at 0.68. Diagnostic and remedy in iteration 4.
