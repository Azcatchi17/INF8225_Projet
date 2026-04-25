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
GDOWN_FOLDER_EXTRA=""
GDOWN_VERSION=""

require_gdown() {
    if ! command -v gdown >/dev/null 2>&1; then
        echo "[stage] ERREUR: gdown absent. Lance d abord experiments/tamia/setup_env.sh" >&2
        exit 1
    fi
    # Compat --remaining-ok :
    #   - gdown <  5.2  : flag n existe pas, limite dure a 50 fichiers / folder
    #   - gdown 5.2.x   : flag necessaire pour depasser 50 fichiers
    #   - gdown >= 6    : flag supprime car le comportement est devenu default
    if [[ -z "$GDOWN_VERSION" ]]; then
        GDOWN_VERSION="$(python -c 'import gdown; print(gdown.__version__)' 2>/dev/null || echo unknown)"
        if gdown --help 2>&1 | grep -q -- '--remaining-ok'; then
            GDOWN_FOLDER_EXTRA="--remaining-ok"
            echo "[stage] gdown $GDOWN_VERSION (--remaining-ok supporte)"
        else
            local gdown_major="${GDOWN_VERSION%%.*}"
            if [[ "$gdown_major" =~ ^[0-9]+$ ]] && (( gdown_major >= 6 )); then
                echo "[stage] gdown $GDOWN_VERSION (>=6 : download >50 fichiers en default, aucun flag requis)"
            else
                echo "[stage] WARNING: gdown $GDOWN_VERSION < 5.2 : limite a 50 fichiers / sous-dossier."
                echo "[stage]          Upgrade: pip install --upgrade --index-url https://pypi.org/simple 'gdown>=5.2.0'"
            fi
        fi
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
          -O "$destination" $GDOWN_FOLDER_EXTRA --quiet
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
          -O "$STAGING" $GDOWN_FOLDER_EXTRA --quiet

    # Drive rate-limit par fichier : pour les dossiers avec beaucoup de PNGs
    # (MSD_pancreas, classifier_dataset_hard) on s attend a des zips. On les
    # decompresse en place avant le dispatch par nom.
    while IFS= read -r -d '' archive; do
        echo "[stage] unzip $archive"
        unzip -q -o "$archive" -d "$(dirname "$archive")"
        rm -f "$archive"
    done < <(find "$STAGING" -type f -name "*.zip" -print0)
    while IFS= read -r -d '' archive; do
        echo "[stage] tar -xzf $archive"
        tar -xzf "$archive" -C "$(dirname "$archive")"
        rm -f "$archive"
    done < <(find "$STAGING" -type f \( -name "*.tar.gz" -o -name "*.tgz" \) -print0)

    echo "[stage] contenu telecharge (apres unzip) :"
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

    # classifier_dataset_hard : deliberement NON dispatche. Les crops du Drive
    # ont ete extraits sous l ancienne strategie (top-1, seuil DINO eleve). La
    # strategie recall actuelle change les bbox envoyees au ResNet (seuil 0.01,
    # top-5, pancreas_margin 35) -> il faut imperativement les re-generer via
    # l etape 01 (extract_hard_negatives.py) pour ne pas entrainer le ResNet
    # sur la mauvaise distribution. On ignore donc l eventuel dossier Drive.
    stale_hard="$(find "$STAGING" -type d -name "classifier_dataset_hard" -print -quit || true)"
    if [[ -n "$stale_hard" && -d "$stale_hard" ]]; then
        echo "[stage] classifier_dataset_hard trouve sur Drive -> IGNORE"
        echo "        (strategie recall : crops a regenerer via etape 01)"
    fi

    rm -rf "$STAGING"
fi

# -----------------------------------------------------------------------------
# 2. Voie (b) : sources individuelles (downloads fichier par fichier)
# -----------------------------------------------------------------------------
# Recommande pour TamIA : gdown traite chaque source individuellement, ce qui
# evite le bug de confirmation antivirus pour les fichiers > 100 Mo qui
# survient en mode folder-traversal (gdown 6.x).

fetch_to_file() {
    # Telecharge UNE source vers UN fichier de destination (creuse le dirname).
    local source="$1"
    local destination="$2"
    if [[ -z "$source" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "$destination")"
    if [[ "$source" == gdrive:* || "$source" == http*://drive.google.com/* ]]; then
        download_gdrive_file "$source" "$destination"
    elif [[ "$source" == http*://* ]]; then
        echo "[stage] wget $source -> $destination"
        wget -q --show-progress -O "$destination" "$source"
    elif [[ "$source" == s3://* ]]; then
        echo "[stage] aws s3 cp $source -> $destination"
        aws s3 cp "$source" "$destination"
    elif [[ -f "$source" ]]; then
        echo "[stage] cp $source -> $destination"
        cp -n "$source" "$destination"
    else
        echo "[stage] fetch_to_file: source non supportee: $source" >&2
        return 1
    fi
}

# --- MSD_pancreas : zip a telecharger puis dezipper -------------------------
if [[ -n "$MSD_SOURCE" ]]; then
    msd_zip="$TAMIA_ASSETS/_msd_pancreas.zip"
    msd_extract="$TAMIA_ASSETS/_msd_extract"

    echo "[stage] MSD_pancreas: download zip"
    fetch_to_file "$MSD_SOURCE" "$msd_zip"

    echo "[stage] MSD_pancreas: unzip -> $msd_extract"
    rm -rf "$msd_extract"
    mkdir -p "$msd_extract"
    unzip -q -o "$msd_zip" -d "$msd_extract"

    # Le zip peut contenir au choix : MSD_pancreas/, data/MSD_pancreas/, ...
    src="$(find "$msd_extract" -type d -name "MSD_pancreas" -print -quit || true)"
    if [[ -z "$src" ]]; then
        echo "[stage] ERREUR: MSD_pancreas/ introuvable dans le zip extrait" >&2
        ls -la "$msd_extract"
        exit 2
    fi
    rm -rf "$DEST_MSD"
    mv "$src" "$DEST_MSD"
    rm -rf "$msd_extract" "$msd_zip"
    echo "[stage] MSD_pancreas: ready at $DEST_MSD"
fi

# --- DINO checkpoint --------------------------------------------------------
fetch_to_file "$DINO_CKPT_SOURCE" "$DEST_DINO_DIR/best_coco_bbox_mAP_epoch_25.pth"

# --- DINO config (tumor_config_v3.py) ---------------------------------------
fetch_to_file "${DINO_CFG_SOURCE:-}" "$DEST_DINO_DIR/tumor_config_v3.py"
# Fallback : si DINO_CFG_SOURCE vide et que le repo a deja le fichier, copie.
if [[ ! -f "$DEST_DINO_DIR/tumor_config_v3.py" && -f "$TAMIA_REPO/tumor_config_v3.py" ]]; then
    cp "$TAMIA_REPO/tumor_config_v3.py" "$DEST_DINO_DIR/tumor_config_v3.py"
    echo "[stage] tumor_config_v3.py copie depuis le repo (fallback)"
fi

# --- MedSAM checkpoint ------------------------------------------------------
fetch_to_file "$MEDSAM_CKPT_SOURCE" "$DEST_MEDSAM_CKPT"

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
