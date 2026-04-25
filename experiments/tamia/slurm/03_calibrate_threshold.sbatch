#!/bin/bash
# experiments/tamia/slurm/03_calibrate_threshold.sbatch
#
# Etape 4 du mode d emploi MSD_RECALL_STRATEGY.md :
#   python calibrate_threshold.py
#
# Balaie les seuils ResNet sur val.json avec budget FP. Produit :
#   optimal_threshold.txt
#   optimal_threshold.json
#   data/results/calibration_threshold_multi_candidate.csv

#SBATCH --job-name=msd-03-calibrate-threshold
#SBATCH --output=experiments/tamia/logs/03_calibrate_threshold-%j.out
#SBATCH --error=experiments/tamia/logs/03_calibrate_threshold-%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120

set -euo pipefail

THIS_DIR="experiments/tamia"
# shellcheck disable=SC1091
source "$THIS_DIR/env.sh"

# Verifie l ensemble ResNet
missing=()
for i in 1 2 3 4 5; do
    if [[ ! -f "$TAMIA_REPO/resnet_fold_${i}.pth" ]]; then
        missing+=("resnet_fold_${i}.pth")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "[03] ERREUR: checkpoints ResNet manquants: ${missing[*]}" >&2
    echo "[03] relance 02_train_resnet d abord." >&2
    exit 2
fi

echo "[03] cwd=$PWD"
echo "[03] lance calibrate_threshold.py"

srun --unbuffered python calibrate_threshold.py

echo "[03] artefacts :"
[[ -f "$TAMIA_REPO/optimal_threshold.txt"  ]] && cat "$TAMIA_REPO/optimal_threshold.txt"
[[ -f "$TAMIA_REPO/optimal_threshold.json" ]] && head -c 600 "$TAMIA_REPO/optimal_threshold.json"
echo ""
echo "[03] done."
