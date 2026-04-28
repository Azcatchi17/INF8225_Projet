# INF8225 — Segmentation de tumeurs pancréatiques sur CT

Projet de session INF8225 (Polytechnique Montréal). Pipeline itératif de détection et segmentation de tumeurs pancréatiques sur le dataset MSD Pancreas, combinant Grounding DINO, MedSAM et des classifieurs ResNet.

## Prérequis

### 1. Accès au Google Drive partagé

Le projet utilise un **Drive partagé** (`Shared Drive`) comme source unique pour les données, poids de modèles et résultats.

> **Le dossier doit apparaître dans vos Drives partagés sous le nom exact `Projet_Medsam`.**

Pour y accéder :
1. Ouvrez le lien du Drive partagé : <https://drive.google.com/drive/folders/1YtHe53jMv_iCys98jEVSTVTWMeBkopb2?usp=share_link>
2. Vérifiez que le dossier est visible dans **Google Drive → Drives partagés → Projet_Medsam**.
3. Sur Colab, le dossier sera monté automatiquement à `/content/drive/Shareddrives/Projet_Medsam`.

Si votre dossier est à un chemin différent (ex. raccourci dans Mon Drive), passez-le explicitement :
```python
from colab.setup import setup
setup(drive_folder="/content/drive/MyDrive/Projet_Medsam")
```

### 2. Structure attendue du Drive

```
Projet_Medsam/
├── data/
│   ├── Kvasir-SEG/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── train.json, val.json, test.json
│   │   ├── kavsir_bboxes.json
│   │   └── polyp_label_map.json
│   └── MSD_pancreas/
│       ├── train/images/, train/masks/
│       ├── val/images/, val/masks/
│       ├── test/images/, test/masks/
│       ├── annotations.json
│       └── train.json, val.json, test.json
├── work_dir/
│   └── MedSAM/
│       └── medsam_vit_b.pth
├── models_weights/
│   ├── grounding_dino_swin-t_pretrain_obj365_goldg.py
│   └── grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth
└── work_dirs/
    ├── polyp_config_v2/
    │   └── best_coco_bbox_mAP_epoch_5.pth
    ├── tumor_config_v3/
    │   └── best_coco_bbox_mAP_epoch_25.pth
    └── pancreas_unet/
        └── best.pt
```

### 3. Runtime Colab

**Runtime → Change runtime type → T4 GPU** (gratuit) suffit pour l'inférence. L'entraînement de Grounding DINO nécessite un A100 (Colab Pro).

## Lancer un notebook

Chaque notebook contient une cellule de bootstrap en première position qui :
1. Clone le repo dans `/content/INF8225_Projet` (branche `temp`)
2. Installe un stack PyTorch + OpenMMLab compatible (torch 2.4.0, mmdet 3.3.0)
3. Monte le Drive et crée des symlinks vers les données et poids
4. Configure `sys.path` pour les imports

Il suffit d'exécuter les cellules dans l'ordre.

## Structure du repo

```
INF8225_Projet/
├── colab/
│   ├── setup.py              # Bootstrap Colab (install deps, symlinks)
│   └── drive_paths.py        # Layout Drive, helpers output_dir()
├── kvasir_implementation/     # Pipeline Kvasir-SEG (polypes)
│   ├── polyp_config_v2.py    # Config Grounding DINO fine-tuné polypes
│   ├── oracle_baseline_kvasir.ipynb
│   └── test_kvasir.ipynb
├── msd_implementation/        # Pipeline MSD Pancreas (tumeurs)
│   ├── configs/grounding_dino/
│   │   └── pancreas_tumor.py # Config DINO fine-tuné tumeurs
│   ├── pipelines/
│   │   ├── dino_medsam_cascade/   # Cascade pancréas→tumeur
│   │   ├── dino_medsam_gemini/    # Pipeline agentique DINO+MedSAM+Gemini
│   │   ├── resnet18_recall/       # Itération 2 : ResNet-18 hard negatives
│   │   ├── three_slice_context/   # Itération 3 : contexte 3 coupes
│   │   ├── resnet50_wide_crop/    # Itération 4 : ResNet-50 + marge élargie
│   │   └── common/               # Utilitaires partagés
│   └── notebooks/
│       ├── dino_medsam_cascade/   # Notebooks baseline + improved
│       ├── resnet18_recall/       # 00-04 : calibrate → extract → train → threshold → eval
│       ├── three_slice_context/   # 01-04
│       └── resnet50_wide_crop/    # 01-05 (+ figures publication)
├── models_weights/            # Config DINO pré-entraîné (symlinké depuis Drive)
├── report/                    # Rapport LaTeX (format IJCAI)
└── MedSAM/                    # Sous-module MedSAM
```

## Pipelines MSD — ordre d'exécution

Chaque itération a ses notebooks numérotés à exécuter dans l'ordre :

| # | Notebook | Description |
|---|----------|-------------|
| 00 | `calibrate_detector` | Calibration seuil DINO (resnet18 uniquement) |
| 01 | `extract_hard_negatives` | Extraction des patchs TP/FP avec DINO à seuil bas |
| 02 | `train_classifier` | Entraînement ensemble 5× ResNet (ImageNet pretrained) |
| 03 | `calibrate_threshold` | Sweep F2 sur val → seuil optimal |
| 04 | `evaluate` | Évaluation finale sur test (Dice, F1, confusion matrix) |

### Itérations

| Itération | Dossier | Modèle | Particularité |
|-----------|---------|--------|---------------|
| 0 | `dino_medsam_cascade/baseline` | DINO + MedSAM | Cascade pancréas→tumeur, pas de classifieur |
| 1 | `dino_medsam_cascade/improved` | + UNet pancréas + gating | Pipeline à 3 portes (G1/G2/G3) |
| 2 | `resnet18_recall/` | ResNet-18 | Hard negative mining, OHEM-inspired |
| 3 | `three_slice_context/` | ResNet-18 3-slice | Contexte inter-coupes (prev/curr/next → RGB) |
| 4 | `resnet50_wide_crop/` | ResNet-50 3-slice | crop_margin=30, meilleur F1 (0.78) |

## Résultats (test, 100 images : 50 tumor / 50 sain)

| Pipeline | Sens. | Spéc. | F1 | Dice tumor | TP | FP | TN | FN |
|----------|-------|-------|----|------------|----|----|----|----|
| Baseline cascade | 0.98 | 0.20 | 0.66 | 0.58 | 49 | 40 | 10 | 1 |
| ResNet-18 | 0.54 | 0.72 | 0.56 | 0.42 | 27 | 14 | 36 | 23 |
| 3-slice ResNet-18 | 0.54 | 0.76 | 0.58 | 0.41 | 27 | 12 | 38 | 23 |
| ResNet-50 wide crop | 0.76 | 0.80 | **0.78** | 0.55 | 38 | 10 | 40 | 12 |

## Dépannage

### `numpy.dtype size changed` ou `mmdet` import error
```python
from colab.setup import setup
setup(reinstall=True)
```
Puis **Runtime → Restart session** et relancez la première cellule.

### `torch was imported before setup()`
Redémarrez le runtime Colab, puis exécutez uniquement la cellule de bootstrap en premier.

### Fichier manquant sur Drive
Vérifiez que le dossier `Projet_Medsam` est bien dans vos **Drives partagés** (pas Mon Drive). Le script affiche les fichiers manquants au démarrage.
