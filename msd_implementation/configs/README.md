# MSD Configs

Configuration files specific to the MSD-Pancreas experiments.

`grounding_dino/pancreas_tumor.py` is the MMDetection/Grounding DINO
fine-tuning config for the pancreas and tumor classes. Runtime notebooks
load this versioned config directly and pair it with the Drive checkpoint at
`work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_25.pth`.
