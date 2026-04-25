#!/bin/bash
# experiments/tamia/slurm/04_test_recall.sbatch
#
# Etape 5 du mode d emploi MSD_RECALL_STRATEGY.md :
#   python test_gd_msd_final_recall.py
#
# Evaluation finale sur test.json avec DINO + ResNet + MedSAM. Produit :
#   data/results/dice_final_report_msd_recall.csv
#
# MedSAM charge un ViT-B (~2G sur GPU) et on garde les 5 ResNet + DINO en
# meme temps. Le script utilise cuda:0, mais TamIA exige un noeud H100:4.

#SBATCH --job-name=msd-04-test-recall
#SBATCH --output=experiments/tamia/logs/04_test_recall-%j.out
#SBATCH --error=experiments/tamia/logs/04_test_recall-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120

set -euo pipefail

THIS_DIR="experiments/tamia"
# shellcheck disable=SC1091
source "$THIS_DIR/env.sh"

if [[ ! -f "$TAMIA_REPO/optimal_threshold.txt" && ! -f "$TAMIA_REPO/optimal_threshold.json" ]]; then
    echo "[04] WARNING: aucun optimal_threshold.* trouve. Le script utilisera 0.25 par defaut."
fi
if [[ ! -f "$TAMIA_REPO/MedSAM/work_dir/MedSAM/medsam_vit_b.pth" ]]; then
    echo "[04] ERREUR: MedSAM vit_b manquant. Lance stage_assets.sh." >&2
    exit 2
fi

echo "[04] cwd=$PWD"
echo "[04] lance test_gd_msd_final_recall.py"

srun --unbuffered python test_gd_msd_final_recall.py

echo "[04] resultats :"
[[ -f "$TAMIA_REPO/data/results/dice_final_report_msd_recall.csv" ]] && \
    ls -la "$TAMIA_REPO/data/results/dice_final_report_msd_recall.csv"

echo "[04] done."
