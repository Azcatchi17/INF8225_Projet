#!/bin/bash
# experiments/tamia/slurm/01_extract_hard_negatives.sbatch
#
# Etape 2 du mode d emploi MSD_RECALL_STRATEGY.md :
#   python extract_hard_negatives.py
#
# Extraction des crops positifs GT + hard negatives DINO pour re-entrainer le
# ResNet. Plus long que l etape 0 (2 splits : train + val).

#SBATCH --job-name=msd-01-extract-hard-negatives
#SBATCH --output=experiments/tamia/logs/01_extract_hard_negatives-%j.out
#SBATCH --error=experiments/tamia/logs/01_extract_hard_negatives-%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120

set -euo pipefail

THIS_DIR="experiments/tamia"
# shellcheck disable=SC1091
source "$THIS_DIR/env.sh"

echo "[01] cwd=$PWD"
echo "[01] lance extract_hard_negatives.py"

# Le script va creer data/classifier_dataset_hard/{train,val}/{0,1}/*.png
# directement sur $SCRATCH (cwd = $TAMIA_REPO).
srun --unbuffered python extract_hard_negatives.py

echo "[01] patches generes :"
find "$TAMIA_REPO/data/classifier_dataset_hard" -maxdepth 3 -type d -printf "  %p\n"
for split in train val; do
    for cls in 0 1; do
        dir="$TAMIA_REPO/data/classifier_dataset_hard/$split/$cls"
        if [[ -d "$dir" ]]; then
            count=$(find "$dir" -name "*.png" | wc -l)
            echo "  $split/$cls: $count"
        fi
    done
done

echo "[01] done."
