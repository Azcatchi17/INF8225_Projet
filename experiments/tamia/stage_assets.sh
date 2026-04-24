#!/usr/bin/env bash
# experiments/tamia/stage_assets.sh
#
# Prepare les assets lourds sur $TAMIA_ASSETS :
#   - dataset MSD_pancreas     -> $TAMIA_ASSETS/MSD_pancreas
#   - DINO v3 config + ckpt    -> $TAMIA_ASSETS/work_dirs/tumor_config_v3/
#   - MedSAM vit_b ckpt        -> $TAMIA_ASSETS/MedSAM_work_dir/MedSAM/medsam_vit_b.pth
#
# A lancer sur le noeud de login (internet requis pour Drive / http / s3).
#
# Deux modes :
#
#   (a) GDRIVE_FOLDER_URL non vide  (voie recommandee a partir d un clone Git
#       vierge) : le script telecharge tout le folder Google Drive via gdown
#       puis range les fichiers aux emplacements attendus. Gere les nested
#       dossiers du folder Drive.
#
#   (b) MSD_SOURCE / DINO_CKPT_SOURCE / MEDSAM_CKPT_SOURCE fournis
#       individuellement. Formats acceptes par source :
#           - chemin local (fichier ou dossier)
#           - http(s)://...
#           - s3://...
#           - gdrive:<file_id>    (fichier unique)
#           - gdrive:<folder_id>  (dossier entier)
#
# Les deux modes peuvent coexister : (a) pose le layout de base, (b) override
# au cas par cas ensuite.

set -euo pipefail

THIS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# shellcheck disable=SC1091
source "$THIS_DIR/config.sh"

# Active le venv pour avoir `gdown` sous la main (installe par setup_env.sh).
if [[ -d "$TAMIA_VENV" ]]; then
    # shellcheck disable=SC1091
    source "$TAMIA_VENV/bin/activate"
fi

DEST_MSD="$TAMIA_ASSETS/MSD_pancreas"
DEST_DINO_DIR="$TAMIA_ASSETS/work_dirs/tumor_config_v3"
DEST_MEDSAM_DIR="$TAMIA_ASSETS/MedSAM_work_dir/MedSAM"
DEST_MEDSAM_CKPT="$DEST_MEDSAM_DIR/medsam_vit_b.pth"

mkdir -p "$TAMIA_ASSETS" "$DEST_DINO_DIR" "$DEST_MEDSAM_DIR"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
require_gdown() {
    if ! command -v gdown >/dev/null 2>&1; then
        echo "[stage] ERREUR: gdown absent. Lance d abord experiments/tamia/setup_env.sh" >&2
        exit 1
    fi
}

extract_gdrive_id() {
    # Extrait un id depuis 'gdrive:<id>', 'https://drive.google.com/.../folders/<id>',
    # '.../file/d/<id>/...', ou 'id=<id>&...'.
    local raw="$1"
    raw="${raw#gdrive:}"
    if [[ "$raw" == http*://* ]]; then
        if [[ "$raw" =~ /folders/([A-Za-z0-9_-]+) ]]; then
            echo "${BASH_REMATCH[1]}"; return 0
        fi
        if [[ "$raw" =~ /file/d/([A-Za-z0-9_-]+) ]]; then
            echo "${BASH_REMATCH[1]}"; return 0
        fi
        local id_re='[?&]id=([A-Za-z0-9_-]+)'
        if [[ "$raw" =~ $id_re ]]; then
            echo "${BASH_REMATCH[1]}"; return 0
        fi
        echo "[stage] ERREUR: impossible d extraire un id Drive de: $raw" >&2
        return 1
    fi
    echo "$raw"
}

download_gdrive_folder() {
    local src="$1"
    local destination="$2"
    require_gdown
    local id
    id="$(extract_gdrive_id "$src")"
    mkdir -p "$destination"
    echo "[stage] gdown --folder id=$id -> $destination"
    # --remaining-ok : continue meme si Drive bride (>50 fichiers par folder).
    gdown --folder "https://drive.google.com/drive/folders/$id" \
          -O "$destination" --remaining-ok --quiet
}

download_gdrive_file() {
    local src="$1"
    local destination="$2"
    require_gdown
    local id
    id="$(extract_gdrive_id "$src")"
    mkdir -p "$(dirname "$destination")"
    echo "[stage] gdown id=$id -> $destination"
    gdown "https://drive.google.com/uc?id=$id" -O "$destination" --quiet
}

stage_one() {
    local label="$1"
    local source="$2"
    local destination="$3"

    if [[ -z "$source" ]]; then
        if [[ -e "$destination" ]]; then
            echo "[stage] $label: OK (present a $destination)"
            return 0
        fi
        echo "[stage] $label: absent ($destination) et aucune source fournie" >&2
        return 1
    fi

    if [[ "$source" == gdrive:* ]]; then
        # Folder si le nom de destination est un dossier, file sinon.
        if [[ "$destination" == */ || -d "$destination" ]]; then
            download_gdrive_folder "$source" "$destination"
        else
            download_gdrive_file "$source" "$destination"
        fi
    elif [[ "$source" == http*://* ]]; then
        echo "[stage] $label: wget $source -> $destination"
        mkdir -p "$(dirname "$destination")"
        wget -q --show-progress -O "$destination" "$source"
    elif [[ "$source" == s3://* ]]; then
        echo "[stage] $label: aws s3 cp $source -> $destination"
        mkdir -p "$(dirname "$destination")"
        aws s3 cp --recursive "$source" "$destination" 2>/dev/null \
            || aws s3 cp "$source" "$destination"
    elif [[ -d "$source" ]]; then
        echo "[stage] $label: rsync (dir) $source -> $destination"
        mkdir -p "$destination"
        rsync -a --info=progress2 "$source/" "$destination/"
    elif [[ -f "$source" ]]; then
        echo "[stage] $label: cp $source -> $destination"
        mkdir -p "$(dirname "$destination")"
        cp -n "$source" "$destination"
    else
        echo "[stage] $label: source introuvable ($source)" >&2
        return 1
    fi
}

# -----------------------------------------------------------------------------
# 1. Voie (a) : pull du folder Drive complet si demande
# -----------------------------------------------------------------------------
if [[ -n "${GDRIVE_FOLDER_URL:-}" ]]; then
    STAGING="$TAMIA_ASSETS/_gdrive_staging"
    rm -rf "$STAGING"
    mkdir -p "$STAGING"
    require_gdown

    folder_id="$(extract_gdrive_id "$GDRIVE_FOLDER_URL")"
    echo "[stage] (a) gdown du folder complet id=$folder_id -> $STAGING"
    gdown --folder "https://drive.google.com/drive/folders/$folder_id" \
          -O "$STAGING" --remaining-ok --quiet

    echo "[stage] contenu telecharge :"
    find "$STAGING" -maxdepth 4 \( -type f -o -type d \) -printf "  %p\n" | head -40

    # --- dispatch : dataset MSD_pancreas ---------------------------------------
    # On accepte plusieurs topologies :
    #   - <staging>/MSD_pancreas/
    #   - <staging>/data/MSD_pancreas/
    #   - <staging>/Projet_Medsam/data/MSD_pancreas/ (layout Drive d origine)
    msd_dir="$(find "$STAGING" -type d -name "MSD_pancreas" -print -quit || true)"
    if [[ -n "$msd_dir" && -d "$msd_dir" ]]; then
        echo "[stage] dispatch: MSD_pancreas depuis $msd_dir -> $DEST_MSD"
        rm -rf "$DEST_MSD"
        mv "$msd_dir" "$DEST_MSD"
    fi

    # --- dispatch : DINO config + checkpoint -----------------------------------
    dino_ckpt="$(find "$STAGING" -type f -name "best_coco_bbox_mAP_epoch_*.pth" -print -quit || true)"
    if [[ -n "$dino_ckpt" && -f "$dino_ckpt" ]]; then
        echo "[stage] dispatch: DINO ckpt $dino_ckpt -> $DEST_DINO_DIR/"
        mv "$dino_ckpt" "$DEST_DINO_DIR/"
    fi
    dino_cfg="$(find "$STAGING" -type f -name "tumor_config_v3.py" -print -quit || true)"
    if [[ -n "$dino_cfg" && -f "$dino_cfg" ]]; then
        echo "[stage] dispatch: DINO config $dino_cfg -> $DEST_DINO_DIR/"
        mv "$dino_cfg" "$DEST_DINO_DIR/"
    fi

    # --- dispatch : MedSAM ckpt ------------------------------------------------
    medsam_ckpt="$(find "$STAGING" -type f -name "medsam_vit_b.pth" -print -quit || true)"
    if [[ -n "$medsam_ckpt" && -f "$medsam_ckpt" ]]; then
        echo "[stage] dispatch: MedSAM ckpt $medsam_ckpt -> $DEST_MEDSAM_CKPT"
        mv "$medsam_ckpt" "$DEST_MEDSAM_CKPT"
    fi

    rm -rf "$STAGING"
fi

# -----------------------------------------------------------------------------
# 2. Voie (b) : sources individuelles (overrides / complements)
# -----------------------------------------------------------------------------
stage_one "msd_dataset"  "$MSD_SOURCE"        "$DEST_MSD/"

if [[ -n "$DINO_CKPT_SOURCE" ]]; then
    # Si c est un dossier ou un folder Drive, on copie tout; sinon fichier.
    if [[ "$DINO_CKPT_SOURCE" == */ || -d "$DINO_CKPT_SOURCE" ]]; then
        stage_one "dino_v3_dir"  "$DINO_CKPT_SOURCE" "$DEST_DINO_DIR/"
    else
        stage_one "dino_v3_ckpt" "$DINO_CKPT_SOURCE" \
                  "$DEST_DINO_DIR/best_coco_bbox_mAP_epoch_25.pth"
    fi
fi

# tumor_config_v3.py : fallback depuis le repo si absent apres dispatch.
if [[ ! -f "$DEST_DINO_DIR/tumor_config_v3.py" ]]; then
    if [[ -f "$TAMIA_REPO/tumor_config_v3.py" ]]; then
        cp "$TAMIA_REPO/tumor_config_v3.py" "$DEST_DINO_DIR/tumor_config_v3.py"
        echo "[stage] tumor_config_v3.py copie depuis le repo"
    fi
fi

stage_one "medsam_ckpt" "$MEDSAM_CKPT_SOURCE" "$DEST_MEDSAM_CKPT"

# -----------------------------------------------------------------------------
# 3. Resume + verifications finales
# -----------------------------------------------------------------------------
echo ""
echo "[stage] Resume :"
for f in \
    "$DEST_MSD/train.json" \
    "$DEST_MSD/val.json" \
    "$DEST_MSD/test.json" \
    "$DEST_DINO_DIR/tumor_config_v3.py" \
    "$DEST_MEDSAM_CKPT"; do
    if [[ -e "$f" ]]; then
        size=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  [OK]     $f  ($size)"
    else
        echo "  [ABSENT] $f"
    fi
done
# DINO ckpt : n importe quelle epoch
dino_any="$(ls -1 "$DEST_DINO_DIR"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1 || true)"
if [[ -n "$dino_any" ]]; then
    echo "  [OK]     $dino_any  ($(du -sh "$dino_any" | cut -f1))"
else
    echo "  [ABSENT] $DEST_DINO_DIR/best_coco_bbox_mAP_epoch_*.pth"
fi

echo ""
echo "[stage] OK. Les symlinks vers \$TAMIA_REPO seront crees par env.sh au"
echo "        premier run des jobs sbatch (experiments/tamia/slurm/*.sbatch)."
