# Kvasir Experiments

Proof-of-concept experiments on Kvasir-SEG, used to validate the
DINO + MedSAM chain on a favourable case (high-contrast polyps with
crisp boundaries) before tackling the harder MSD-Pancreas task.

The associated notebooks live under `notebooks/kvasir/`:

- `segment_oracle.ipynb` — MedSAM oracle DICE using the ground-truth
  bounding boxes shipped with Kvasir-SEG. Establishes the segmenter's
  upper bound (DICE = 0.954 on 1000 images).
- `analyze_dataset.ipynb` — dataset statistics and image inspection.

The detector configurations for Kvasir live under
`configs/grounding_dino/polyp_kvasir.py` and `polyp_kvasir_v2.py`.
