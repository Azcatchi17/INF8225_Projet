#!/bin/bash
# experiments/tamia/slurm/02_train_resnet.sbatch
#
# Etape 3 du mode d emploi MSD_RECALL_STRATEGY.md :
#   python train_resnet_kfold_recall.py
#
# Entraine 5 ResNet-18 (15 epoques chaque, sequentiel dans le script) sur les
# crops de data/classifier_dataset_hard/. Produit resnet_fold_{1..5}.pth +
# threshold_run_{1..5}.txt.
#
# Note TamIA : le script est sequentiel et utilise cuda:0. Le scheduler exige
# quand meme un noeud GPU complet (`--gpus-per-node=h100:4`).

#SBATCH --job-name=msd-02-train-resnet
#SBATCH --output=experiments/tamia/logs/02_train_resnet-%j.out
#SBATCH --error=experiments/tamia/logs/02_train_resnet-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:TERM@120

set -euo pipefail

THIS_DIR="experiments/tamia"
# shellcheck disable=SC1091
source "$THIS_DIR/env.sh"

# Verification sanity : les dossiers de crops doivent exister avant l entree
# dans train (sinon l erreur est ininterpretable).
for split in train val; do
    for cls in 0 1; do
        dir="$TAMIA_REPO/data/classifier_dataset_hard/$split/$cls"
        if [[ ! -d "$dir" ]] || [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
            echo "[02] ERREUR: $dir est vide. Lance 01_extract_hard_negatives d abord." >&2
            exit 2
        fi
    done
done

echo "[02] cwd=$PWD"
echo "[02] lance train_resnet_kfold_recall.py"

srun --unbuffered python train_resnet_kfold_recall.py

echo "[02] artefacts produits :"
ls -la "$TAMIA_REPO"/resnet_fold_*.pth 2>/dev/null || echo "  aucun .pth"
ls -la "$TAMIA_REPO"/threshold_run_*.txt 2>/dev/null || echo "  aucun threshold_run"

echo "[02] done."
