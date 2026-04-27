# MSD Notebooks

Colab orchestration notebooks for the MSD-Pancreas implementation. Each
notebook bootstraps the repository with `colab/setup.py`, syncs heavy
assets from Drive, then calls the matching source module under
`msd_implementation/pipelines/`.

```
notebooks/
├── dino_medsam_cascade/      # iter 1 and archived improved pipeline
├── resnet18_recall/          # iter 2
├── three_slice_context/      # iter 3
└── resnet50_wide_crop/       # iter 4 and publication figures
```
