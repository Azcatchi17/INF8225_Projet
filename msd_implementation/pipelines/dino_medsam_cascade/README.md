# Iteration 1 — DINO + MedSAM with Anatomical Cascade

The simplest pipeline: a fine-tuned `Grounding DINO` predicts both
pancreas and tumor proposals in a single forward pass; we keep the
top tumor proposal whose box overlaps an inflated neighbourhood of the
top pancreas proposal, then segment it with `MedSAM`.

Module structure:

| File | Purpose |
|------|---------|
| `cascade_detector.py` | Wrapper around DINO that splits pancreas/tumor proposals and applies the spatial cascade rule. |
| `postprocess.py` | Bounding-box padding, largest-connected-component extraction, area filtering. |
| `evaluation.py` | Full evaluation loop on a COCO-style split with optional post-processing flags. |

## Running

```bash
python -c "from msd_implementation.pipelines.dino_medsam_cascade.evaluation import run_baseline_v2; \
           df, summary = run_baseline_v2(); print(summary)"
```

The notebook `msd_implementation/notebooks/dino_medsam_cascade/baseline.ipynb` provides
the same call wired in a Colab-friendly cell layout.

`msd_implementation/notebooks/dino_medsam_cascade/improved_pipeline.ipynb` preserves
the full MSD Colab run used for the UNet-gated DINO + MedSAM ablation
and threshold calibration.

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pancreas_thr` | 0.30 | Min DINO score for the pancreas branch. |
| `tumor_thr` | 0.01 | Permissive on tumors (the cascade does the filtering). |
| `pancreas_margin_px` | 20 | Inflation of the pancreas box before overlap test. |
| `min_overlap_ratio` | 0.10 | Required overlap between tumor box and inflated pancreas. |
| `pad_frac` | 0.0 | Optional box padding before MedSAM. |
| `keep_largest`, `min_area_px`, `max_area_frac` | _off_ | Optional mask post-processing. |
