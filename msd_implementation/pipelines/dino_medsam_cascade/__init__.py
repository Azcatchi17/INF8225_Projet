"""Iteration 1 - DINO + MedSAM with anatomical cascade.

Reproduces the pancreas->tumor cascade baseline. The detector
(`Grounding DINO`) emits both pancreas and tumor proposals; we keep the
highest-scoring tumor proposal whose bounding box overlaps a dilated
neighbourhood of the highest-scoring pancreas proposal, then send it to
`MedSAM` for segmentation.

Submodules are imported lazily so that `import msd_implementation` does
not pull in heavy runtime dependencies (torch, mmdet, MedSAM).
"""
