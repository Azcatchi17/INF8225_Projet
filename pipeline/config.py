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

# --- Trust region (pre-stop) ---------------------------------------------
# If the current mask passes ALL these, skip the Gemini review and stop.
# Rationale: batch showed Gemini degrades already-good masks ~40% of the
# time. A mask with tight bbox agreement + single connected blob + decent
# compactness almost always IS the polyp.
TRUST_BBOX_IOU = 0.7
TRUST_COMPACTNESS = 0.45
TRUST_MAX_COMPONENTS = 1

# --- Gemma -----------------------------------------------------------------
# Gemini 3.1 Pro on Google AI Studio (paid tier, multimodal).
GEMMA_MODEL_ID = "gemini-3.1-pro"
GEMMA_API_KEY_ENV = "GEMINI_API_KEY"
GEMMA_IMAGE_SIDE = 512              # downscale images sent to Gemma to save tokens
GEMMA_MAX_REQUESTS_PER_MIN = 1000   # paid Tier 1 (adjust to match your quota)
GEMMA_MAX_RETRIES = 6
GEMMA_RETRY_BASE_DELAY = 1.0
GEMMA_RETRY_BUFFER_SEC = 1.0      # small cushion around server-provided delays

# --- MedSAM / device ------------------------------------------------------
DEVICE = "cuda"  # falls back to cpu at load time if unavailable
MEDSAM_INPUT_SIDE = 1024
USE_BF16 = True

# --- One-shot prompt used for the baseline comparison ---------------------
DEFAULT_DINO_TEXT = "polyp."
