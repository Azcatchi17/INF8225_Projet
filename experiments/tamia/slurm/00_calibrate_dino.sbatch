#!/bin/bash
# experiments/tamia/slurm/00_calibrate_dino.sbatch
#
# Etape 1 du mode d emploi MSD_RECALL_STRATEGY.md :
#   python calibrate_dino.py
#
# Balaie une grille de seuils DINO sur val.json et imprime la table TP/FN/FP.
# GPU : 1x H100 suffit largement (200 scans val, ~1s/scan).
#
# Soumission directe :
#   sbatch --export=ALL experiments/tamia/slurm/00_calibrate_dino.sbatch
#
# Ou via l orchestrateur :
#   bash experiments/tamia/submit_all.sh

#SBATCH --job-name=msd-00-calibrate-dino
#SBATCH --output=experiments/tamia/logs/00_calibrate_dino-%j.out
#SBATCH --error=experiments/tamia/logs/00_calibrate_dino-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120

set -euo pipefail

# Charge config + modules + venv + symlinks (config.sh donne aussi SLURM_ACCOUNT
# mais sbatch a deja lu le header ; si tu ajustes le compte, passe via
# sbatch -A $SLURM_ACCOUNT dans submit_all.sh).
THIS_DIR="experiments/tamia"
# shellcheck disable=SC1091
source "$THIS_DIR/env.sh"

echo "[00] cwd=$PWD"
echo "[00] python=$(which python)"
echo "[00] lance calibrate_dino.py"

srun --unbuffered python calibrate_dino.py

echo "[00] done."
