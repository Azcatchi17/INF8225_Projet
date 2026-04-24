#!/usr/bin/env bash
# experiments/tamia/setup_env.sh
#
# A lancer UNE FOIS sur le noeud de login TamIA (les noeuds de calcul n ont pas
# internet). Cree :
#   - le venv $TAMIA_VENV (python 3.11, torch 2.5.1+cu121, mm*, MedSAM editable)
#   - le clone du repo sur $TAMIA_REPO (copie rsync depuis le cwd courant)
#
# Usage :
#   cd ~/INF8225_Projet               # ton clone initial dans $HOME
#   bash experiments/tamia/setup_env.sh
#
# Pre-requis : experiments/tamia/config.sh deja versionne avec les bonnes
# valeurs (PI_NAME, CLUSTER_USER, GPU_TYPE, GDRIVE_FOLDER_URL).

set -euo pipefail

THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_SRC="$( cd -- "$THIS_DIR/../.." &> /dev/null && pwd )"

if [[ ! -f "$THIS_DIR/config.sh" ]]; then
    echo "[setup] experiments/tamia/config.sh absent (devrait etre versionne)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$THIS_DIR/config.sh"

echo "[setup] PI_NAME=$PI_NAME  SLURM_ACCOUNT=$SLURM_ACCOUNT"
echo "[setup] TAMIA_REPO=$TAMIA_REPO"
echo "[setup] TAMIA_ASSETS=$TAMIA_ASSETS"
echo "[setup] TAMIA_VENV=$TAMIA_VENV"

# -----------------------------------------------------------------------------
# 1. Modules (StdEnv 2023 + cuda 12.2, pile standard Alliance Canada)
# -----------------------------------------------------------------------------
module purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn opencv/4.10.0

# -----------------------------------------------------------------------------
# 2. Copie du repo vers $SCRATCH (outputs iront la-bas, pas dans $HOME)
# -----------------------------------------------------------------------------
mkdir -p "$TAMIA_REPO" "$TAMIA_ASSETS" "$(dirname "$TAMIA_VENV")"

echo "[setup] rsync repo -> $TAMIA_REPO"
rsync -a --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='data/' \
    --exclude='work_dirs/' \
    --exclude='MedSAM/work_dir/' \
    --exclude='resnet_fold_*.pth' \
    --exclude='optimal_threshold.*' \
    --exclude='threshold_run_*.txt' \
    --exclude='experiments/tamia/logs/' \
    "$REPO_SRC/" "$TAMIA_REPO/"

# Assure que MedSAM (package editable) existe sur scratch puisque pip le
# reference par chemin local.
if [[ ! -d "$TAMIA_REPO/MedSAM" ]]; then
    echo "[setup] MedSAM absent du repo source : clone-le d abord ($REPO_SRC/MedSAM)" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 3. Venv
# -----------------------------------------------------------------------------
if [[ -d "$TAMIA_VENV" ]]; then
    echo "[setup] venv deja present, skip creation"
else
    echo "[setup] creation venv $TAMIA_VENV"
    virtualenv --no-download "$TAMIA_VENV"
fi
# shellcheck disable=SC1091
source "$TAMIA_VENV/bin/activate"
pip install --no-index --upgrade pip

# -----------------------------------------------------------------------------
# 4. Paquets disponibles dans le wheelhouse Alliance (pas d internet requis)
# -----------------------------------------------------------------------------
# torch 2.5 existe pour StdEnv/2023 + cuda 12.1. Les autres (numpy, scipy,
# scikit-image, scikit-learn, pandas, Pillow, tqdm, matplotlib, pycocotools,
# opencv-python, nibabel, simpleitk) sont presque tous dans le wheelhouse.
pip install --no-index \
    torch torchvision \
    numpy scipy pandas pillow tqdm matplotlib \
    scikit-image scikit-learn \
    pycocotools nibabel simpleitk \
    opencv-python

# -----------------------------------------------------------------------------
# 5. Paquets qui necessitent internet (mmengine, mmcv, mmdet, transformers...)
# -----------------------------------------------------------------------------
# Le noeud de login a internet. mim resout les dependances mmcv/mmdet et
# recupere la bonne roue CUDA.
pip install openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"

# transformers 4.33 est ce que la config DINO attend (BERT tokenizer) ; on le
# force explicitement pour eviter qu une version plus recente ne casse le text
# encoder.
pip install "transformers==4.33.0" "tokenizers==0.13.3"

# MedSAM est installe en editable depuis le sous-dossier clone.
pip install -e "$TAMIA_REPO/MedSAM"

# Quelques utilitaires legers non critiques
pip install "addict" "yapf" "terminaltables" "shapely" || true

# gdown : necessaire si stage_assets.sh doit tirer les assets depuis Google
# Drive (voie A du README). Harmless sinon.
pip install "gdown>=5.1.0"

# -----------------------------------------------------------------------------
# 6. .gitignore local pour les logs
# -----------------------------------------------------------------------------
GITIGNORE="$THIS_DIR/.gitignore"
if [[ ! -f "$GITIGNORE" ]]; then
    cat > "$GITIGNORE" <<'EOF'
logs/*.out
logs/*.err
EOF
    echo "[setup] .gitignore cree : $GITIGNORE"
fi

# -----------------------------------------------------------------------------
# 7. Diag
# -----------------------------------------------------------------------------
python - <<'PY'
import torch
import mmdet, mmcv, mmengine
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"mmdet={mmdet.__version__}  mmcv={mmcv.__version__}  mmengine={mmengine.__version__}")
PY

echo ""
echo "[setup] OK."
echo "[setup] Etape suivante :"
echo "          bash experiments/tamia/stage_assets.sh"
echo "        puis :"
echo "          bash experiments/tamia/submit_all.sh"
