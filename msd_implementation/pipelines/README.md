# MSD Pipelines

Reusable Python modules for the MSD-Pancreas experiments.

```
pipelines/
├── common/                  # shared DINO proposal and ensemble scoring logic
├── dino_medsam_cascade/     # iter 1: anatomical DINO + MedSAM cascade
├── dino_medsam_gemini/      # archived DINO/MedSAM/Gemini support code
├── resnet18_recall/         # iter 2: hard-negative ResNet-18 verifier
├── three_slice_context/     # iter 3: 3-slice context verifier
└── resnet50_wide_crop/      # iter 4: final wide-crop ResNet-50 verifier
```

Run modules from the repository root:

```bash
python -m msd_implementation.pipelines.resnet50_wide_crop.evaluate
```
