# experiments/tamia/env.sh
#
# Source ce fichier au debut de chaque job SLURM pour :
#   - charger les modules TamIA (StdEnv, python, cuda, cudnn, opencv, gcc)
#   - activer le venv persistant construit par setup_env.sh
#   - garantir que le repertoire courant a les bons symlinks vers $SCRATCH
#     (data/MSD_pancreas, work_dirs/tumor_config_v3, MedSAM/work_dir, ...)
#
# Usage :
#   source experiments/tamia/env.sh
#
# Variables attendues : venant de config.sh (TAMIA_REPO, TAMIA_ASSETS, TAMIA_VENV).

set -euo pipefail

THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [[ ! -f "$THIS_DIR/config.sh" ]]; then
    echo "[env.sh] ERREUR: experiments/tamia/config.sh absent" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$THIS_DIR/config.sh"

# -----------------------------------------------------------------------------
# 1. Modules TamIA
# -----------------------------------------------------------------------------
# StdEnv/2023 est la pile recommandee. python/3.11 est celle compatible avec
# mmdet 3.3 + mmcv 2.1. cuda/12.2 est la version pairee avec StdEnv/2023 sur
# Alliance. opencv fourni par module pour eviter la compilation.
module purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn opencv/4.10.0

# -----------------------------------------------------------------------------
# 2. Venv
# -----------------------------------------------------------------------------
if [[ ! -d "$TAMIA_VENV" ]]; then
    echo "[env.sh] venv absent: $TAMIA_VENV" >&2
    echo "[env.sh] lance d abord: bash experiments/tamia/setup_env.sh" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$TAMIA_VENV/bin/activate"

# -----------------------------------------------------------------------------
# 3. CWD de travail (les scripts du mode d emploi utilisent des chemins relatifs)
# -----------------------------------------------------------------------------
if [[ ! -d "$TAMIA_REPO" ]]; then
    echo "[env.sh] repo absent: $TAMIA_REPO" >&2
    echo "[env.sh] lance d abord: bash experiments/tamia/setup_env.sh" >&2
    exit 1
fi
cd "$TAMIA_REPO"

# -----------------------------------------------------------------------------
# 4. Symlinks vers les assets lourds sur $SCRATCH
# -----------------------------------------------------------------------------
# calibrate_dino.py et ses copains cherchent :
#   data/MSD_pancreas/{train,val,test}.json + images/masks
#   work_dirs/tumor_config_v3/{tumor_config_v3.py, best_coco_bbox_mAP_epoch_25.pth}
#   MedSAM/work_dir/MedSAM/medsam_vit_b.pth
# On les pose via symlinks sans toucher le code.
link_if_missing() {
    local src="$1"
    local dst="$2"
    if [[ -e "$dst" || -L "$dst" ]]; then
        return 0
    fi
    if [[ ! -e "$src" ]]; then
        echo "[env.sh] asset manquant: $src" >&2
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
    echo "[env.sh] symlink cree: $dst -> $src"
}

link_if_missing "$TAMIA_ASSETS/MSD_pancreas"                         "$TAMIA_REPO/data/MSD_pancreas"
link_if_missing "$TAMIA_ASSETS/work_dirs/tumor_config_v3"             "$TAMIA_REPO/work_dirs/tumor_config_v3"
link_if_missing "$TAMIA_ASSETS/MedSAM_work_dir"                       "$TAMIA_REPO/MedSAM/work_dir"

# -----------------------------------------------------------------------------
# 5. Reglages runtime
# -----------------------------------------------------------------------------
# Les scripts utilisent des chemins relatifs; on s assure qu ils s ecrivent sur
# $SCRATCH (via le cwd) et pas dans le venv.
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Les scripts calibrate/test appellent mmdet / torch cuda; on print ce qu il
# faut pour diagnostiquer rapidement dans les .out.
echo "[env.sh] SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}  SLURM_NODELIST=${SLURM_NODELIST:-n/a}"
python -c "import torch; print(f'[env.sh] torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}')"
