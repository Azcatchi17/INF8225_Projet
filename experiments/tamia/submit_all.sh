#!/usr/bin/env bash
# experiments/tamia/submit_all.sh
#
# Soumet les 5 jobs du mode d emploi MSD_RECALL_STRATEGY.md en chaine avec des
# dependances --dependency=afterok. Chaque etape ne demarre que si la
# precedente s est terminee sans erreur.
#
# Usage :
#   bash experiments/tamia/submit_all.sh
#   bash experiments/tamia/submit_all.sh --from 02    # demarre a l etape 02
#   bash experiments/tamia/submit_all.sh --only 04    # soumet juste 04
#   bash experiments/tamia/submit_all.sh --dry-run    # print les sbatch sans soumettre

set -euo pipefail

THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck disable=SC1091
source "$THIS_DIR/config.sh"

FROM="00"
ONLY=""
DRY=0

while (( $# > 0 )); do
    case "$1" in
        --from)    FROM="$2"; shift 2 ;;
        --only)    ONLY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--from STEP] [--only STEP] [--dry-run]
  STEP dans {00,01,02,03,04}
EOF
            exit 0
            ;;
        *) echo "option inconnue: $1" >&2; exit 1 ;;
    esac
done

STEPS=(
  "00|00_calibrate_dino.sbatch|$SLURM_TIME_DINO"
  "01|01_extract_hard_negatives.sbatch|$SLURM_TIME_EXTRACT"
  "02|02_train_resnet.sbatch|$SLURM_TIME_TRAIN"
  "03|03_calibrate_threshold.sbatch|$SLURM_TIME_CALIB"
  "04|04_test_recall.sbatch|$SLURM_TIME_TEST"
)

common_args=(
    "--account=$SLURM_ACCOUNT"
    "--gpus-per-node=${GPU_TYPE}:${GPUS_PER_NODE}"
    "--parsable"
)
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
    common_args+=("--mail-user=$SLURM_MAIL_USER" "--mail-type=${SLURM_MAIL_TYPE}")
fi

prev_jid=""
for spec in "${STEPS[@]}"; do
    IFS='|' read -r step sbatch_file time_limit <<< "$spec"

    if [[ -n "$ONLY" && "$step" != "$ONLY" ]]; then
        continue
    fi
    if (( 10#$step < 10#$FROM )) && [[ -z "$ONLY" ]]; then
        continue
    fi

    args=("${common_args[@]}" "--time=$time_limit")
    if [[ -n "$prev_jid" && -z "$ONLY" ]]; then
        args+=("--dependency=afterok:$prev_jid")
    fi

    sbatch_path="$THIS_DIR/slurm/$sbatch_file"
    if [[ "$DRY" -eq 1 ]]; then
        echo "sbatch ${args[*]} $sbatch_path"
        prev_jid="DRYRUN-$step"
    else
        jid=$(sbatch "${args[@]}" "$sbatch_path")
        echo "[submit] $step -> JID=$jid  (sbatch $sbatch_file)"
        prev_jid="$jid"
    fi
done

if [[ "$DRY" -eq 0 ]]; then
    echo ""
    echo "[submit] Jobs soumis. Suivi :"
    echo "         squeue -u \$USER"
    echo "         tail -f experiments/tamia/logs/*-\${SLURM_JOB_ID}.out"
fi
