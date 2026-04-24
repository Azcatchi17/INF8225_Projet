# experiments/tamia/config.example.sh
#
# Copie ce fichier en config.sh et remplis les valeurs specifiques a ton compte
# TamIA (Alliance Canada). Toutes les variables ici sont lues par env.sh et par
# les sbatch dans slurm/.
#
#   cp experiments/tamia/config.example.sh experiments/tamia/config.sh
#   nano experiments/tamia/config.sh
#
# config.sh est gitignore via experiments/tamia/.gitignore (cree par setup_env.sh).

# =============================================================================
# 1. Compte SLURM TamIA
# =============================================================================
# TamIA n utilise pas les allocations DRAC standard (def-/rrg-). Il utilise des
# allocations AIP : --account=aip-<prof>. Ton prof doit t avoir ajoute a son
# allocation AIP. Exemple : aip-bengioy, aip-pal.
PI_NAME="${PI_NAME:-bengioy}"
SLURM_ACCOUNT="aip-${PI_NAME}"

# =============================================================================
# 2. Type de GPU demande
# =============================================================================
# TamIA a des H100 (80G) et des H200 (141G). H100 est moins cher en RGUs.
# Si tu n as pas besoin de H200 reste sur h100.
GPU_TYPE="${GPU_TYPE:-h100}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

# =============================================================================
# 3. Arborescence TamIA
# =============================================================================
# $HOME             : lecture/ecriture, quota ~50G, backup.
# $SCRATCH          : grand (~20T), pas de backup, purge automatique. Donnees.
# ~/projects/...    : shared lab storage, utile pour les venvs persistants.
#
# On cree une copie du repo sur $SCRATCH pour pouvoir ecrire les outputs sans
# toucher $HOME. Les checkpoints lourds (DINO, MedSAM) et le dataset MSD vivent
# aussi sur $SCRATCH.
TAMIA_REPO="${TAMIA_REPO:-$SCRATCH/INF8225_Projet}"
TAMIA_ASSETS="${TAMIA_ASSETS:-$SCRATCH/INF8225_assets}"
TAMIA_VENV="${TAMIA_VENV:-$HOME/envs/msd_recall}"

# =============================================================================
# 4. Sources des assets a stager
# =============================================================================
# Deux chemins possibles :
#
# (a) Voie simple : UN seul folder Google Drive contenant tout (dataset + les
#     deux checkpoints). stage_assets.sh telecharge le folder avec `gdown` puis
#     dispatche les fichiers au bon endroit. C est ce qui est prerempli ci-
#     dessous et qui marche out-of-the-box en clonant depuis GitHub.
GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1Y__wKNXPZ9UpaXqCWM9VnN8GkvajomxE}"

# (b) Voie fine : tu fournis chaque source individuellement. Format accepte :
#       - chemin local (fichier ou dossier)   -> rsync / cp
#       - https://...                          -> wget
#       - s3://...                             -> aws s3 cp
#       - gdrive:<file_id>  OR gdrive:<folder_id> -> gdown
#     Si tu laisses ces trois variables vides ET GDRIVE_FOLDER_URL vide,
#     stage_assets.sh verifie juste que les fichiers sont deja en place dans
#     $TAMIA_ASSETS.
MSD_SOURCE="${MSD_SOURCE:-}"
DINO_CKPT_SOURCE="${DINO_CKPT_SOURCE:-}"
MEDSAM_CKPT_SOURCE="${MEDSAM_CKPT_SOURCE:-}"

# =============================================================================
# 5. Ressources par job (limite noeud sur TamIA)
# =============================================================================
# TamIA alloue par noeud, pas par CPU. Meme en demandant 1 GPU tu reserves un
# noeud complet. --mem=0 et --exclusive prennent donc toute la RAM et tous les
# CPUs. Tu peux overrider par job via --export dans submit_all.sh.
SLURM_TIME_DINO="${SLURM_TIME_DINO:-00:30:00}"
SLURM_TIME_EXTRACT="${SLURM_TIME_EXTRACT:-01:30:00}"
SLURM_TIME_TRAIN="${SLURM_TIME_TRAIN:-03:00:00}"
SLURM_TIME_CALIB="${SLURM_TIME_CALIB:-00:45:00}"
SLURM_TIME_TEST="${SLURM_TIME_TEST:-01:00:00}"

# =============================================================================
# 6. Notifications email (optionnel)
# =============================================================================
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-END,FAIL}"
