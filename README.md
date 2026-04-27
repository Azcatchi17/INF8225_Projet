# INF8225 Project — Pancreatic Tumor Segmentation with Open-Vocabulary Detection

Code accompanying the report **"From Polyp Detection to Pancreatic
Tumor Segmentation: Limits of Open-Vocabulary Detectors and the Role of
a Learned Downstream Classifier"** (INF8225, Polytechnique Montréal,
2026).

We compose a fine-tuned `Grounding DINO` detector with the `MedSAM`
promptable segmenter, then add a learned ResNet-50 verifier in second
stage to recover the specificity that the open-vocabulary detector
loses on out-of-distribution medical classes. The pipeline is evaluated
on `Kvasir-SEG` (proof of concept) and on `Medical Segmentation
Decathlon, Pancreas` (target task).

## Headline numbers

| Pipeline (test split, 100 images, 50/50) | Sens. | Spec. | F1 |
|------------------------------------------|-------|-------|-----|
| DINO + MedSAM (no verifier) | 0.98 | 0.18 | 0.70 |
| + Anatomical cascade pancreas→tumor (iter 1) | 0.94 | 0.40 | 0.74 |
| + ResNet-18 ensemble (iter 2) | 0.62 | 0.82 | 0.69 |
| + 3-slice channel context (iter 3) | 0.54 | 0.96 | 0.68 |
| **+ ResNet-50 pretrained + wide crop (iter 4)** | **0.76** | **0.80** | **0.78** |

`MedSAM` oracle DICE on tumor-positive slices = 0.88, ie. the
segmenter is not the bottleneck.

## Repository layout

```
INF8225_Projet/
├── README.md                  # this file
├── LICENSE                    # MIT
├── CITATION.cff               # how to cite this work
├── requirements.txt
├── requirements-colab.txt      # Colab-only extras installed by the bootstrap
├── .gitignore
│
├── src/inf8225_project/       # reserved for shared library code (TBD)
│
├── experiments/               # experiment code and reproducible run helpers
│   ├── colab_setup.py          # Colab bootstrap used by notebooks
│   ├── kvasir/
│   └── msd/
│       ├── _shared/
│       ├── dino_medsam_cascade/      # iter 1
│       │   └── improved_pipeline.ipynb
│       ├── resnet18_recall/          # iter 2
│       ├── three_slice_context/      # iter 3
│       └── resnet50_wide_crop/       # iter 4 (final)
│
├── notebooks/                 # Colab orchestration notebooks
│   ├── kvasir/
│   └── msd/
│       ├── dino_medsam_cascade/
│       ├── resnet18_recall/
│       ├── three_slice_context/
│       └── resnet50_wide_crop/
│
├── configs/                   # detector configs (Grounding DINO etc.)
│
├── data/                      # symlinked from Drive in Colab
│   ├── README.md
│   ├── raw/                   # gitignored
│   ├── processed/             # mostly gitignored
│   └── results/               # small CSVs only
│
├── models/                    # gitignored, populated at training time
│
├── outputs/                   # gitignored, run artefacts
│
├── report/                    # IJCAI-style write-up
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
└── tests/                     # unit smoke tests
```

`experiments/` contains the reusable iteration code. `notebooks/`
contains lightweight orchestration notebooks that invoke the iteration
scripts via `python -m`. Read
`experiments/msd/README.md` for the chronological story of the four
iterations.

## Quick start

```bash
git clone https://github.com/moradBMH/INF8225_Projet.git
cd INF8225_Projet
pip install -r requirements.txt

# Open the publication-ready notebook on Colab
# notebooks/msd/resnet50_wide_crop/04_evaluate.ipynb
```

Heavy assets (DINO, MedSAM, MSD pancreas dataset) are not committed.
On Colab, `experiments/colab_setup.py` symlinks them from a Drive folder; locally
you have to provide them yourself.

## Reproducing the paper's table

`notebooks/msd/resnet50_wide_crop/05_publication_figures.ipynb`
regenerates Figures 1-3 from the test CSV. The exact numbers reported
in the paper come from this CSV
(`data/results/dice_final_report_resnet50_wide_crop.csv`).

## Citation

If you use this code, please cite the report — see `CITATION.cff`.

## License

MIT, see `LICENSE`.
