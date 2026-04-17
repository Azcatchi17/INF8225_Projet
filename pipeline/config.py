"""Central configuration for the agentic segmentation pipeline."""
from __future__ import annotations

from pathlib import Path

# --- Repo / Drive paths ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DINO_CONFIG = PROJECT_ROOT / "polyp_config.py"
DINO_CHECKPOINT = PROJECT_ROOT / "work_dirs" / "polyp_config" / "best_coco_bbox_mAP_epoch_2.pth"
MEDSAM_CHECKPOINT = PROJECT_ROOT / "MedSAM" / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

KVASIR_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"
KVASIR_IMAGES = KVASIR_ROOT / "images"
KVASIR_MASKS = KVASIR_ROOT / "masks"
KVASIR_TEST_JSON = KVASIR_ROOT / "test.json"

AGENT_RUNS_DIR = PROJECT_ROOT / "data" / "agent_runs"
AGENT_RESULTS_DIR = PROJECT_ROOT / "data" / "agent_results"
AGENT_CACHE_DIR = PROJECT_ROOT / "data" / "agent_cache"

# --- Agent loop thresholds ------------------------------------------------
MAX_ITER = 5
CONVERGENCE_IOU = 0.98      # stop if consecutive masks are almost identical
DINO_SCORE_THRESHOLD = 0.3  # keep only boxes above this confidence
DINO_TOP_K = 5              # candidates kept for replace_box action
EMPTY_AREA_PCT = 0.001      # below this → short-circuit add_positive
OVERSIZED_AREA_PCT = 0.4    # above this → short-circuit shrink_box

# --- Gemma -----------------------------------------------------------------
# Free tier by default; switch to "gemma-4-e4b-it" (HF: google/gemma-4-E4B-it)
# once the free version has validated the pipeline end-to-end.
GEMMA_MODEL_ID = "gemini-2.5-flash"
GEMMA_API_KEY_ENV = "GEMINI_API_KEY"
GEMMA_IMAGE_SIDE = 512            # downscale images sent to Gemma to save tokens
GEMMA_MAX_REQUESTS_PER_MIN = 5    # Gemini free-tier default for this model
GEMMA_MAX_RETRIES = 6
GEMMA_RETRY_BASE_DELAY = 1.0
GEMMA_RETRY_BUFFER_SEC = 1.0      # small cushion around server-provided delays

# --- MedSAM / device ------------------------------------------------------
DEVICE = "cuda"  # falls back to cpu at load time if unavailable
MEDSAM_INPUT_SIDE = 1024
USE_BF16 = True

# --- One-shot prompt used for the baseline comparison ---------------------
DEFAULT_DINO_TEXT = "polyp."
