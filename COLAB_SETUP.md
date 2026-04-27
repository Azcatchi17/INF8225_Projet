# Running the project on Google Colab

Minimal setup to run the notebooks of this repo on Colab with a GPU runtime, using Google Drive as the source for the heavy files (dataset, checkpoints).

## 1. Prepare your Google Drive (one time)

1. Open the shared folder: <https://drive.google.com/drive/folders/1BAcGyja2SHP3t2OFOleN2cND4QlXb3fW>
2. Right-click the folder → **Organize → Add shortcut → My Drive**. Place the shortcut at the root of **My Drive** so Colab finds it at `/content/drive/MyDrive/Projet_Medsam`.

Expected layout inside the Drive folder:

```
Projet_Medsam/                                         ← Drive folder
├── data/
│   └── Kvasir-SEG/
│       ├── images/              (1000 .jpg)
│       ├── masks/               (1000 .jpg)
│       ├── kavsir_bboxes.json
│       ├── train.json  val.json  test.json
│       └── polyp_label_map.json
├── work_dirs/
│   └── polyp_config/
│       ├── polyp_config.py
│       └── best_coco_bbox_mAP_epoch_2.pth     ← fine-tuned DINO
├── work_dir/
│       └── MedSAM/
│           └── medsam_vit_b.pth
└── grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth   (optional, training only)
```

If your shortcut lands at a different path, pass it explicitly:
```python
from notebooks.colab_setup import setup
setup(drive_folder="/content/drive/MyDrive/some/other/path")
```

## 2. Pick a GPU runtime

**Runtime → Change runtime type → T4 GPU** (free) is enough for inference in all notebooks. Training Grounding DINO from scratch needs more memory — use Colab Pro (A100) or a local GPU.

## 3. Open a notebook

Notebooks with a Colab bootstrap cell already wired:

| Notebook | Purpose | Needs GPU? | Colab-ready |
|---|---|---|---|
| `notebooks/kvasir/segment_oracle.ipynb` | Kvasir MedSAM oracle segmentation | **yes** | yes |
| `notebooks/msd/dino_medsam_cascade/baseline.ipynb` | MSD DINO + MedSAM cascade baseline | **yes** | yes |
| `notebooks/msd/dino_medsam_cascade/improved_pipeline.ipynb` | archived MSD improved-pipeline run | **yes** | yes |
| `notebooks/msd/resnet50_wide_crop/04_evaluate.ipynb` | final MSD ResNet-50 verifier evaluation | **yes** | yes |
| `notebooks/msd/resnet50_wide_crop/05_publication_figures.ipynb` | publication figures from final CSV | CPU OK | yes |
| `notebooks/exploratory/convert_to_coco.ipynb` | COCO split generation (run once) | CPU OK | — |

The bootstrap cell at the top of each Colab-ready notebook:
1. Clones this repo into `/content/INF8225_Projet` (if not already there) and `cd`s into it.
2. Pins Colab to a known-good PyTorch stack supported by OpenMMLab:
   - GPU runtime: `torch==2.4.0`, `torchvision==0.19.0`, `torchaudio==2.4.0` from the `cu121` wheels
   - CPU runtime: `torch==2.3.1`, `torchvision==0.18.1`, `torchaudio==2.3.1`
3. Pins the scientific stack to known-good versions for Colab Py 3.12:
   - `numpy==1.26.4`
   - `scipy==1.13.1`
   - `matplotlib==3.8.4`
4. Installs `mmengine==0.10.7`, `mmcv==2.2.0`, and `mmdet==3.3.0` from prebuilt wheels only.
5. Installs the small extras from `requirements-colab.txt` (`transformers`, `nltk`, `pycocotools`, ...).
6. Mounts Drive, locates the project folder, creates symlinks so the notebook code (which uses paths like `data/Kvasir-SEG/...` and `work_dirs/polyp_config/...`) finds everything on Drive unchanged.

## Repairing a stale Colab runtime

If a previous session left `numpy` / `scipy` in a bad state and the notebook still fails on
`from mmdet.apis import init_detector, inference_detector`, force a clean reinstall once:

```python
from notebooks.colab_setup import setup
setup(reinstall=True)
```

If `torch` was already imported before the bootstrap cell ran, restart the runtime first, then rerun the first cell.

## 4. Outputs persist on Drive automatically

Because `data/` and `work_dirs/` are symlinks to Drive, every write from a notebook — new CSVs in `data/results/`, visualizations in `data/outputs*/`, training checkpoints in `work_dirs/polyp_config/` — lands on your Drive. Nothing is lost when the Colab VM is recycled.

## Running locally

Setup is a no-op when `google.colab` is not importable, so the same notebooks still run locally with your existing venv/conda env and the repo's native paths.
