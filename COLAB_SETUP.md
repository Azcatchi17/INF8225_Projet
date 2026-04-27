# Running the project on Google Colab

Minimal setup to run the notebooks of this repo on Colab with a GPU runtime, using Google Drive as the source for the heavy files (dataset, checkpoints).

## 1. Prepare your Google Drive (one time)

1. Open the shared folder: <https://drive.google.com/drive/folders/1BAcGyja2SHP3t2OFOleN2cND4QlXb3fW>
2. Right-click the folder → **Organize → Add shortcut → My Drive**. Place the shortcut at the root of **My Drive** so Colab finds it at `/content/drive/MyDrive/Projet_Medsam`.

Expected layout inside the Drive folder:

```
Projet_Medsam/                                         ← Drive folder
├── manifest.json
├── data/
│   ├── Kvasir-SEG/
│   └── MSD_pancreas/
├── work_dirs/
│   ├── polyp_config/
│   ├── polyp_config_v2/
│   ├── tumor_config_v3/
│   └── pancreas_unet/
├── work_dir/
│   └── MedSAM/
│       └── medsam_vit_b.pth
├── models_weights/
│   ├── grounding_dino_swin-t_pretrain_obj365_goldg.py
│   └── grounding_dino_swin-t_pretrain_obj365_goldg_*.pth
└── outputs/
    ├── kvasir_implementation/
    │   ├── oracle_baseline_kvasir/
    │   ├── calibrate_dino_threshold_kvasir/
    │   └── test_kvasir/
    └── msd_implementation/
        ├── dino_calibration/
        ├── dino_medsam_cascade/
        ├── dino_medsam_gemini/
        ├── resnet18_recall/
        ├── three_slice_context/
        └── resnet50_wide_crop/
```

If your shortcut lands at a different path, pass it explicitly:
```python
from colab.setup import setup
setup(drive_folder="/content/drive/MyDrive/some/other/path")
```

## 2. Pick a GPU runtime

**Runtime → Change runtime type → T4 GPU** (free) is enough for inference in all notebooks. Training Grounding DINO from scratch needs more memory — use Colab Pro (A100) or a local GPU.

## 3. Open a notebook

Notebooks with a Colab bootstrap cell already wired:

| Notebook | Purpose | Needs GPU? | Colab-ready |
|---|---|---|---|
| `grounding_dino.ipynb` | detection inference with fine-tuned DINO | **yes** | yes |
| `kvasir_implementation/oracle_baseline_kvasir.ipynb` | Kvasir MedSAM oracle baseline | **yes** | yes |
| `kvasir_implementation/test_kvasir.ipynb` | Kvasir DINO + MedSAM zero-shot/fine-tuned evaluation | **yes** | yes |
| `msd_implementation/notebooks/resnet18_recall/*.ipynb` | MSD recall-oriented ResNet-18 pipeline | **yes** | yes |
| `msd_implementation/notebooks/three_slice_context/*.ipynb` | MSD 3-slice ResNet-18 pipeline | **yes** | yes |
| `msd_implementation/notebooks/resnet50_wide_crop/*.ipynb` | MSD final ResNet-50 wide-crop pipeline and figures | **yes** | yes |
| `msd_implementation/notebooks/dino_medsam_cascade/*.ipynb` | MSD DINO + MedSAM cascade baselines | **yes** | yes |

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
5. Installs the small extras from `colab/requirements-colab.txt` (`transformers`, `nltk`, `pycocotools`, ...).
6. Mounts Drive, locates the project folder, creates `manifest.json` and the `outputs/` tree, then symlinks `data/`, `work_dirs/`, `outputs/`, `MedSAM/work_dir/`, and the Grounding DINO weight into the repo.

## Repairing a stale Colab runtime

If a previous session left `numpy` / `scipy` in a bad state and the notebook still fails on
`from mmdet.apis import init_detector, inference_detector`, force a clean reinstall once:

```python
from colab.setup import setup
setup(reinstall=True)
```

If `torch` was already imported before the bootstrap cell ran, restart the runtime first, then rerun the first cell.

## 4. Outputs persist on Drive automatically

Because `data/`, `work_dirs/`, and `outputs/` are symlinks to Drive, every notebook write lands under the same academic layout. Generated CSVs, figures, masks, caches, runs, and checkpoints go to `outputs/<implementation>/<pipeline>/...`. Nothing is lost when the Colab VM is recycled.

## Running locally

Setup is a no-op when `google.colab` is not importable, so the same notebooks still run locally with your existing venv/conda env and the repo's native paths.
