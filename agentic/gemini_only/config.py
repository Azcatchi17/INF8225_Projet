"""Gemini-only variant config. Same as dino_gemini.config except separate output
dirs so runs don't collide with the DINO+Gemini baseline."""
from __future__ import annotations

from agentic.dino_gemini.config import *  # noqa: F401,F403 — re-export shared constants
from agentic.dino_gemini import config as _base

AGENT_RUNS_DIR = _base.PROJECT_ROOT / "data" / "agent_runs_vlm"
AGENT_RESULTS_DIR = _base.PROJECT_ROOT / "data" / "agent_results_vlm"
AGENT_CACHE_DIR = _base.PROJECT_ROOT / "data" / "agent_cache_vlm"

# VLM detector thresholds (replace DINO's)
VLM_SCORE_THRESHOLD = 0.3
VLM_TOP_K = 5
