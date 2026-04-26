"""3-slice v2 scoring helper.

Same logic as ``experiments.three_slice.score_3slice.score_candidates_3slice``
but kept here so the v2 pipeline imports a single namespace. The scoring
function itself does not depend on the backbone (ResNet-50 vs ResNet-18) -
that choice happens when loading the ensemble in the calibrate/test
scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.three_slice.score_3slice import score_candidates_3slice  # noqa: F401

__all__ = ["score_candidates_3slice"]
