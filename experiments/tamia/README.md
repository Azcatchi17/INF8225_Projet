# Mode d emploi MSD_RECALL_STRATEGY sur TamIA

Adaptation SLURM des scripts du fichier racine `MSD_RECALL_STRATEGY.md` pour le cluster **TamIA** (Alliance Canada / PAICE).

L objectif est de pouvoir derouler ce pipeline en entier sur TamIA :

```bash
python calibrate_dino.py
python extract_hard_negatives.py
python train_resnet_kfold_recall.py
python calibrate_threshold.py
python test_gd_msd_final_recall.py
```

sans modifier aucun des scripts python existants, en chainant les etapes via `sbatch --dependency`.

---

## 0. Rappel TamIA

- TamIA est equipe de **H100 (80G)** et **H200 (141G)**.
- L allocation est **par noeud, pas par CPU** : meme un job a 1 GPU mobilise un noeud entier. On met donc `--mem=0` pour prendre toute la RAM.
- Le compte n est pas un compte DRAC classique (`def-`/`rrg-`) mais un compte **AIP** : `--account=aip-<nom_du_prof>`. Ton prof doit t ajouter a son allocation AIP avant le premier sbatch.
- Les noeuds de **calcul** n ont pas d internet : tous les `pip install` se font sur le **noeud de login**.
- Ref : https://docs.alliancecan.ca/wiki/TamIA

---

## 1. Arborescence creee par cette adaptation

```
experiments/tamia/
├── README.md                        (ce fichier)
├── config.sh                        pre-rempli (compte aip-azouaq, user dchikhi, GPU h100)
├── env.sh                           sourced par chaque job (modules + venv + symlinks)
├── setup_env.sh                     a lancer UNE FOIS sur le noeud de login
├── stage_assets.sh                  stage dataset + checkpoints sur $SCRATCH (via gdown)
├── submit_all.sh                    orchestrateur (sbatch avec dependences)
├── slurm/
│   ├── 00_calibrate_dino.sbatch
│   ├── 01_extract_hard_negatives.sbatch
│   ├── 02_train_resnet.sbatch
│   ├── 03_calibrate_threshold.sbatch
│   └── 04_test_recall.sbatch
└── logs/                            sorties SLURM (.out / .err)
```

Rien dans `agentic/`, rien a la racine du repo. Les scripts python originaux restent intouches.

Arborescence sur le cluster apres setup :

```
$HOME/INF8225_Projet/                   # ton clone git initial
$SCRATCH/INF8225_Projet/                # copie pour les runs (outputs, crops...)
$SCRATCH/INF8225_assets/                # gros fichiers (dataset, ckpts)
$HOME/envs/msd_recall/                  # venv persistent
```

Au debut de chaque job, `env.sh` cree des symlinks pour que le cwd de $SCRATCH/INF8225_Projet ressemble au layout attendu :

```
$SCRATCH/INF8225_Projet/
├── data/MSD_pancreas            -> $SCRATCH/INF8225_assets/MSD_pancreas
├── work_dirs/tumor_config_v3    -> $SCRATCH/INF8225_assets/work_dirs/tumor_config_v3
└── MedSAM/work_dir              -> $SCRATCH/INF8225_assets/MedSAM_work_dir
```

Les fichiers generes (resnet_fold_*.pth, optimal_threshold.{txt,json}, threshold_run_*.txt, data/results/*.csv, data/classifier_dataset_hard/*) atterrissent directement dans `$SCRATCH/INF8225_Projet` et survivent entre les jobs.

---

## 2. Setup initial (une seule fois)

### 2.1 Cloner le repo

```bash
ssh <user>@tamia.alliancecan.ca          # login
cd $HOME
git clone https://github.com/moradBMH/INF8225_Projet.git   # ou ton fork
cd INF8225_Projet
```

### 2.2 Verifier config.sh

`experiments/tamia/config.sh` est **pre-rempli et versionne** dans le repo. Verifie juste que les valeurs correspondent a ton setup :

| Variable | Valeur actuelle | Si different |
|---|---|---|
| `PI_NAME` | `azouaq` | edite si ton allocation AIP n est pas `aip-azouaq` |
| `CLUSTER_USER` | `dchikhi` | mets ton username TamIA (utilise pour rsync exterieur) |
| `GPU_TYPE` | `h100` | `h200` si tu vises specifiquement les 141G |
| `GDRIVE_FOLDER_URL` | folder Morad | OK tant que le folder reste partage en lecture |

Aucun edit n est necessaire si tu executes depuis le compte `dchikhi` avec l allocation `aip-azouaq`.

### 2.3 Construire le venv et copier le repo sur $SCRATCH

```bash
bash experiments/tamia/setup_env.sh
```

Ce script :

1. charge les modules TamIA (`StdEnv/2023 python/3.11 cuda/12.1 cudnn opencv/4.10.0`),
2. `rsync` le repo (sans `data/`, `work_dirs/`, checkpoints) vers `$SCRATCH/INF8225_Projet`,
3. cree le venv `$HOME/envs/msd_recall`,
4. installe torch/torchvision depuis le **wheelhouse Alliance** (`pip install --no-index`),
5. installe `mmengine==0.10.7`, `mmcv==2.1.0`, `mmdet==3.3.0` via `mim` (internet login node),
6. installe `transformers==4.33.0` (la version que le text encoder DINO attend),
7. installe MedSAM en editable depuis `MedSAM/` du repo,
8. cree `experiments/tamia/.gitignore` pour ne pas committer les logs SLURM.

### 2.4 Stager les assets

```bash
bash experiments/tamia/stage_assets.sh
```

**Par defaut** (et c est le cas cible si tu clones juste depuis GitHub sans assets locaux), le script utilise `GDRIVE_FOLDER_URL` de `config.sh` et tire tout le folder Google Drive via `gdown` :

```
Drive folder (prerempli dans config.sh)
  ├── MSD_pancreas/                             -> $TAMIA_ASSETS/MSD_pancreas/
  ├── best_coco_bbox_mAP_epoch_25.pth           -> $TAMIA_ASSETS/work_dirs/tumor_config_v3/
  ├── tumor_config_v3.py    (optionnel)         -> idem (fallback: copie depuis le repo)
  └── medsam_vit_b.pth                          -> $TAMIA_ASSETS/MedSAM_work_dir/MedSAM/
```

Le folder Drive doit etre **partage en lecture** avec le compte Google de celui qui execute (le plus simple : "Anyone with the link").

**Sources alternatives** (pour overrider fichier par fichier) via `config.sh` :

| Format | Exemple |
|---|---|
| Chemin local | `MSD_SOURCE=/home/user/MSD_pancreas` |
| Http(s) | `MSD_SOURCE=https://.../MSD.tar.gz` |
| S3 | `MSD_SOURCE=s3://bucket/MSD_pancreas` |
| Google Drive file | `DINO_CKPT_SOURCE=gdrive:<file_id>` |
| Google Drive folder | `GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/<id>` |

Si toutes les variables `*_SOURCE` et `GDRIVE_FOLDER_URL` sont vides, le script verifie simplement que les fichiers sont deja en place dans `$TAMIA_ASSETS/`.

### 2.5 Sanity check rapide

```bash
source experiments/tamia/env.sh         # hors job, juste pour valider
python -c "import mmdet, torch; print(mmdet.__version__, torch.cuda.is_available())"
ls data/MSD_pancreas/*.json
ls work_dirs/tumor_config_v3/
ls MedSAM/work_dir/MedSAM/
```

> Note : `env.sh` fait `exit 1` si le venv ou les assets manquent. Rien de destructif.

---

## 3. Lancer le pipeline

### 3.1 Tout en une commande (recommande)

```bash
bash experiments/tamia/submit_all.sh
```

Soumet les 5 jobs en chaine. Chaque job demarre apres succes (`afterok`) du precedent :

```
00_calibrate_dino    ┐
                     ├-> 01_extract_hard_negatives
                     │    ┐
                     │    ├-> 02_train_resnet
                     │    │    ┐
                     │    │    ├-> 03_calibrate_threshold
                     │    │    │    ┐
                     │    │    │    └-> 04_test_recall
```

Options :

```bash
bash experiments/tamia/submit_all.sh --from 02        # demarre a 02 (skip 00, 01)
bash experiments/tamia/submit_all.sh --only 04        # soumet juste 04
bash experiments/tamia/submit_all.sh --dry-run        # affiche sans soumettre
```

### 3.2 Soumettre une etape isolement

```bash
sbatch --account=aip-$PI_NAME --gres=gpu:h100:1 \
    experiments/tamia/slurm/00_calibrate_dino.sbatch
```

Les options `--account` et `--gres` sont redondantes si deja definies dans le header sbatch, mais elles permettent d overrider sans editer le fichier.

### 3.3 Suivre les jobs

```bash
squeue -u $USER
sacct -u $USER --starttime=today --format=JobID,JobName,State,Elapsed,ReqGRES
tail -f experiments/tamia/logs/00_calibrate_dino-*.out
```

---

## 4. Artefacts attendus

Apres un run complet sur `$SCRATCH/INF8225_Projet/` :

| Etape | Sortie principale |
|---|---|
| 00 | table TP/FN/FP dans `logs/00_*.out` |
| 01 | `data/classifier_dataset_hard/{train,val}/{0,1}/*.png` |
| 02 | `resnet_fold_{1..5}.pth`, `threshold_run_{1..5}.txt` |
| 03 | `optimal_threshold.txt`, `optimal_threshold.json`, `data/results/calibration_threshold_multi_candidate.csv` |
| 04 | `data/results/dice_final_report_msd_recall.csv` |

Interpretation des causes de FN dans le CSV final (colonne `fn_cause`) :

- `dino_or_spatial_filters` : aucun candidat n a survecu a DINO + filtre pancreas.
- `resnet_threshold` : un candidat existe mais le ResNet le rejette.
- `medsam_or_wrong_box` : un candidat est accepte mais MedSAM ne recouvre pas la tumeur.

Pour recuperer les CSV sur ta machine :

```bash
# depuis ta machine locale
rsync -avz tamia.alliancecan.ca:\$SCRATCH/INF8225_Projet/data/results/ ./data/results/
```

---

## 5. Ressources demandees par job

Valeurs par defaut (ajustables via `config.sh`) :

| Etape | GPU | Duree | Notes |
|---|---|---|---|
| 00 calibrate_dino | 1x H100 | 30 min | 100 images val, inference DINO seule |
| 01 extract_hard_negatives | 1x H100 | 90 min | DINO sur ~900 images train + 100 val |
| 02 train_resnet | 1x H100 | 3 h | 5 folds x 15 epochs, sequentiel |
| 03 calibrate_threshold | 1x H100 | 45 min | DINO + ensemble ResNet sur val |
| 04 test_recall | 1x H100 | 60 min | DINO + ResNet + MedSAM sur test |

Note facturation TamIA : chaque job reserve un noeud complet (H100 = 12.15 RGUs / noeud pour 1 GPU). Meme avec `--gres=gpu:h100:1`, tu paies le noeud. Si tu veux amortir, tu peux paralleliser 4 jobs sur un meme noeud via `srun --exclusive` et un orchestrateur custom ; pas fait par defaut ici pour garder le scenario simple.

---

## 6. Depannage

**`module: command not found`** : tu es peut-etre sur une VM non-Alliance. Les scripts sont concus pour TamIA / docs.alliancecan.ca et assument un Lmod module system.

**`virtualenv: command not found`** : ajoute `module load python/3.11` avant de creer l env, ou utilise `python -m venv` avec `--system-site-packages`.

**`mmcv` echoue a compiler** : re-execute `setup_env.sh` avec `CUDA_HOME` exporte explicitement. Sur Alliance : `export CUDA_HOME=$(dirname $(dirname $(which nvcc)))`.

**`sbatch: error: Invalid account`** : ton prof ne t a pas encore ajoute a l allocation AIP, ou `PI_NAME` dans `config.sh` est faux.

**Le job reste `PD` avec raison `(QOSMinCpuNotSatisfied)`** : TamIA alloue par noeud; retire `--cpus-per-task` et laisse `--mem=0`. Les headers sbatch livres ici sont deja conformes.

**Les checkpoints DINO/MedSAM manquent apres stage_assets.sh** : verifie que les variables `*_SOURCE` de `config.sh` pointent vers des chemins accessibles depuis le login node. Sinon copie les fichiers a la main via `rsync` vers `$TAMIA_ASSETS/`.

**Le script echoue sur `No such file: data/MSD_pancreas/val.json`** : `env.sh` n a pas reussi a creer les symlinks. Verifie que `$TAMIA_ASSETS/MSD_pancreas/val.json` existe.

---

## 7. Differences volontaires avec `MSD_RECALL_STRATEGY.md` racine

Aucune ligne des 5 scripts python du mode d emploi n est modifiee : `calibrate_dino.py`, `extract_hard_negatives.py`, `train_resnet_kfold_recall.py`, `calibrate_threshold.py`, `test_gd_msd_final_recall.py` tournent tels quels. Les seules adaptations sont :

- environnement execute sur `$SCRATCH` au lieu du cwd local,
- donnees et checkpoints references via symlinks,
- ordonnancement SLURM avec dependences,
- modules + venv Alliance avec `--no-index` pour les paquets du wheelhouse.

Si tu veux modifier le comportement d un script (seuils, nombres d epoques...), ouvre une PR sur la racine ou fais une copie dans un autre sous-dossier `experiments/` conformement a la regle du projet.
