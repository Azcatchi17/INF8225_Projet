# Configs

Detector and model configurations organised by component.

## Layout

```
configs/
├── grounding_dino/   # MMDet-style configs for Grounding DINO
├── medsam/           # MedSAM-related configuration (currently empty)
└── resnet/           # ResNet ensemble hyper-parameters (currently empty)
```

## Grounding DINO configs

| File | Used by |
|------|---------|
| `swin_t_base.py` | Base Swin-T configuration shared by all DINO fine-tunings. |
| `pancreas_tumor.py` | MSD-Pancreas fine-tuning (classes: pancreas, tumor) — referenced by all iterations under `experiments/msd/`. |
| `polyp_kvasir.py` | Kvasir-SEG polyp detector (single class). |
| `polyp_kvasir_v2.py` | Second iteration of the Kvasir detector with adjusted hyper-parameters. |

The trained checkpoints for these configs live under `work_dirs/`
(gitignored) and are typically symlinked from a Drive folder via
`notebooks/colab_setup.py`. Iteration scripts reference the runtime config copy
that MMDet writes alongside the checkpoint:
```python
init_detector("work_dirs/tumor_config_v3/tumor_config_v3.py", ckpt_path)
```
