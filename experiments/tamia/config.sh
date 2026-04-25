# experiments/tamia/config.sh
#
# Config TamIA pour l execution du mode d emploi MSD_RECALL_STRATEGY.md.
# Usernames et prof pre-remplis. Toute la config vient de :
#   - https://docs.alliancecan.ca/wiki/TamIA  (doc officielle, blocker anti-bot
#     au moment de la redaction -> certaines valeurs sont les defauts Alliance
#     Canada classiques, a confirmer au premier sbatch)
#   - https://docs.mila.quebec/technical_reference/clusters/external/

# =============================================================================
# 1. Compte SLURM TamIA
# =============================================================================
# Alloc AIP = aip-<pi_name>. Le PI doit t avoir ajoute a son allocation AIP
# avant que sbatch n accepte tes jobs.
PI_NAME="azouaq"
SLURM_ACCOUNT="aip-${PI_NAME}"

# Username TamIA (utilise pour scp/rsync depuis l exterieur, et pour les
# chemins $HOME / $SCRATCH qui sont deriv es automatiquement).
CLUSTER_USER="dchikhi"

# =============================================================================
# 2. Type de GPU demande
# =============================================================================
# Format Alliance Canada: --gres=gpu:<type>:<n>. Sur les autres clusters Alliance
# le type est le code du GPU (a100, v100l, ...). Pour TamIA c est h100 ou h200.
# H100 = ~12.15 RGU / noeud (source Mila docs). H200 = plus cher.
GPU_TYPE="h100"
GPUS_PER_NODE="1"

# =============================================================================
# 3. Arborescence TamIA
# =============================================================================
# Sur Alliance Canada :
#   $HOME                = /home/$CLUSTER_USER          (quota ~50G, backup)
#   $SCRATCH             = /scratch/$CLUSTER_USER       (~20T, pas de backup)
#   $PROJECT/def-<pi>/   = /project/.../def-<pi>        (quota partage, backup)
#
# Sur TamIA les equivalents sont typiquement les memes var env ($SCRATCH etc.).
# On n utilise ici que $HOME et $SCRATCH. L allocation AIP (aip-azouaq) donne
# potentiellement un /project/aip-azouaq/ partage mais pas obligatoire.
TAMIA_REPO="$SCRATCH/INF8225_Projet"
TAMIA_ASSETS="$SCRATCH/INF8225_assets"
TAMIA_VENV="$HOME/envs/msd_recall"

# =============================================================================
# 4. Sources des assets (Google Drive partage par Morad)
# =============================================================================
# Folder Drive `Projet_tamia` (a plat, pour eviter le rate-limit Drive sur les
# milliers de PNG individuels — les gros dossiers sont zippes) :
#   MSD_pancreas.zip                              (dataset, ~2 Go zippe)
#   best_coco_bbox_mAP_epoch_25.pth               (checkpoint DINO v3)
#   tumor_config_v3.py                            (config DINO)
#   medsam_vit_b.pth                              (checkpoint MedSAM)
#
# stage_assets.sh decompresse les zips automatiquement et dispatche par nom.
GDRIVE_FOLDER_URL="https://drive.google.com/drive/u/0/folders/1BgOn3074_GUrFu2qw5sGnXV0zCxaII73"

# Sources individuelles vides = on se repose entierement sur GDRIVE_FOLDER_URL.
MSD_SOURCE=""
DINO_CKPT_SOURCE=""
MEDSAM_CKPT_SOURCE=""

# =============================================================================
# 5. Temps alloue par job (prevu large, ajuste si attente en queue trop longue)
# =============================================================================
SLURM_TIME_DINO="00:30:00"
SLURM_TIME_EXTRACT="01:30:00"
SLURM_TIME_TRAIN="03:00:00"
SLURM_TIME_CALIB="00:45:00"
SLURM_TIME_TEST="01:00:00"

# =============================================================================
# 6. Notifications email (optionnel)
# =============================================================================
SLURM_MAIL_USER=""
SLURM_MAIL_TYPE="END,FAIL"
