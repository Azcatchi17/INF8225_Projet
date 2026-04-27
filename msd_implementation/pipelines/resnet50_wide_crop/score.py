"""3-slice v2 scoring helper.

Same logic as ``msd_implementation.pipelines.three_slice_context.score_3slice.score_candidates_3slice``
but kept here so the v2 pipeline imports a single namespace. The scoring
function itself does not depend on the backbone (ResNet-50 vs ResNet-18) -
that choice happens when loading the ensemble in the calibrate/test
scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msd_implementation.pipelines.three_slice_context.score import score_candidates_3slice  # noqa: F401

__all__ = ["score_candidates_3slice"]
