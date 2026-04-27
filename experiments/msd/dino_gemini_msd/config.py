"""Central configuration for the DINO + MedSAM + Gemini MSD pipeline."""
from __future__ import annotations
from pathlib import Path

# --- Repo / Drive paths ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Pointage vers tes nouveaux modèles MSD
DINO_CONFIG = PROJECT_ROOT / "tumor_config_v3.py"
DINO_CHECKPOINT = PROJECT_ROOT / "work_dirs" / "tumor_config_v3" / "best_coco_bbox_mAP_epoch_25.pth"
MEDSAM_CHECKPOINT = PROJECT_ROOT / "MedSAM" / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

# Adaptation au dataset MSD
MSD_ROOT = PROJECT_ROOT / "data" / "MSD_pancreas"
# Note : tes chemins d'images incluent souvent 'train/' ou 'test/'
MSD_IMAGES = MSD_ROOT 
MSD_MASKS = MSD_ROOT
MSD_TEST_JSON = MSD_ROOT / "test.json"

AGENT_RUNS_DIR = PROJECT_ROOT / "data" / "agent_runs_msd"
AGENT_RESULTS_DIR = PROJECT_ROOT / "data" / "agent_results_msd"
AGENT_CACHE_DIR = PROJECT_ROOT / "data" / "agent_cache_msd"

# --- Agent loop thresholds ------------------------------------------------
MAX_ITER = 5
CONVERGENCE_IOU = 0.98      
DINO_SCORE_THRESHOLD = 0.06  # Seuil abaissé car DINO est plus timide sur le pancréas
DINO_TOP_K = 5              
EMPTY_AREA_PCT = 0.001      
OVERSIZED_AREA_PCT = 0.4    

# --- Trust region ---------------------------------------------------------
# On garde ces valeurs, mais elles seront plus dures à atteindre sur le pancréas
TRUST_BBOX_IOU = 0.85
TRUST_COMPACTNESS = 0.5
TRUST_MAX_COMPONENTS = 1

# --- Phase 2: MedSAM improvements ----------------------------------------
POSTPROCESS_MASK = True
POSTPROCESS_MIN_COMPONENT = 0.05
ENSEMBLE_TOP_K = 3

# --- Phase 3: pancreas-ROI gated pipeline --------------------------------
# G1 — pancreas ROI
PANCREAS_CKPT = PROJECT_ROOT / "work_dirs" / "pancreas_unet" / "best.pt"
PANCREAS_DILATION_PX = 35           # morpho-dilate the UNet mask to absorb border tumors
MIN_PANCREAS_AREA_PX = 200          # below this, declare "no pancreas on this slice"

# G2 — tumor detection filter
MIN_PANCREAS_OVERLAP = 0.05         # fraction of candidate box area that must fall inside pancreas mask
MIN_CENTER_IN_MASK = False          # require the box centroid to lie on the pancreas mask
CLIP_MASK_TO_PANCREAS_ROI = False   # final AND between MedSAM mask and pancreas ROI

# G2.5 — dual-prompt differential (optional)
DUAL_PROMPT_DISTRACTORS = [
    "normal pancreas tissue",
    "bowel gas",
    "abdominal organ",
]
DUAL_PROMPT_ALPHA = 0.5             # weight on max distractor score in differential

# G3 — tumor-presence threshold (CALIBRATE on val, then hard-code the result here)
TUMOR_DIFF_THRESHOLD = 0.25

# --- Gemma / Gemini --------------------------------------------------------
# Correction des identifiants (voir analyse ci-dessous)
GEMMA_MODEL_ID = "gemini-3-flash-preview"
GEMMA_API_KEY_ENV = "GEMINI_API_KEY"
GEMMA_IMAGE_SIDE = 512              
GEMMA_MAX_REQUESTS_PER_MIN = 25   
GEMMA_MAX_RETRIES = 12
GEMMA_RETRY_BASE_DELAY = 5.0
GEMMA_RETRY_MAX_DELAY = 45.0
GEMMA_RETRY_BUFFER_SEC = 2.0

# --- MedSAM / device ------------------------------------------------------
DEVICE = "cuda"  
MEDSAM_INPUT_SIDE = 1024
USE_BF16 = True

# --- Prompt pour MSD ------------------------------------------------------
DEFAULT_DINO_TEXT = "pancreas . tumor ."
