"""Dataclasses describing a run's state and per-iteration results."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    score: float

    def to_dict(self) -> dict:
        return {"xyxy": list(self.xyxy), "score": float(self.score)}


@dataclass
class MaskMetrics:
    area_pct: float
    n_components: int
    largest_component_pct: float
    compactness: float
    bbox_agreement_iou: float
    is_empty: bool
    is_oversized: bool
    dice: Optional[float] = None  # only set when GT is available

    def to_dict(self) -> dict:
        return asdict(self)

    def to_gemma_dict(self) -> dict:
        """Same as to_dict, but without DICE (Gemma must not see the GT signal)."""
        d = asdict(self)
        d.pop("dice", None)
        return d


@dataclass
class GemmaAction:
    action: str
    params: dict
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationResult:
    iteration: int
    box_used_xyxy: list[float]
    points_used: list[list[float]]
    point_labels: list[int]
    metrics: MaskMetrics
    gemma_action: Optional[GemmaAction]
    dice_vs_gt: Optional[float]
    elapsed_ms: float
    # Kept in-memory only (not JSON-serialised)
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "box_used_xyxy": list(self.box_used_xyxy),
            "points_used": [list(p) for p in self.points_used],
            "point_labels": list(self.point_labels),
            "metrics": self.metrics.to_dict(),
            "gemma_action": self.gemma_action.to_dict() if self.gemma_action else None,
            "dice_vs_gt": self.dice_vs_gt,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class AgentState:
    run_id: str
    image_path: str
    raw_user_text: str
    refined_prompt: str = ""
    synonyms: list[str] = field(default_factory=list)
    candidate_boxes: list[Box] = field(default_factory=list)
    iterations: list[IterationResult] = field(default_factory=list)
    stop_reason: str = ""
    # Runtime-only (never serialised)
    image_np: Optional[np.ndarray] = field(default=None, repr=False)
    image_embed: Any = field(default=None, repr=False)

    @classmethod
    def new(cls, image_path: str, user_text: str) -> "AgentState":
        run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
        return cls(run_id=run_id, image_path=image_path, raw_user_text=user_text)

    def push_iter(self, result: IterationResult) -> None:
        self.iterations.append(result)

    def final_mask(self) -> Optional[np.ndarray]:
        it = self.best_iter()
        return it.mask if it else None

    def best_iter(self) -> Optional["IterationResult"]:
        """Anchor iter 0 unless a later iteration clearly improves BOTH bbox_iou AND
        compactness by >= 5%. Protects the detection-baseline mask from being
        overwritten by a noisy Gemini-driven iteration whose proxy score happens to
        be slightly higher but whose real DICE is worse."""
        if not self.iterations:
            return None

        def score(it: "IterationResult") -> float:
            m = it.metrics
            if m.is_empty or m.is_oversized or m.n_components == 0:
                return -1.0
            s = m.bbox_agreement_iou * max(m.compactness, 0.1)
            if m.n_components > 1:
                s *= m.largest_component_pct
            return s

        it0 = self.iterations[0]
        m0 = it0.metrics
        iter0_healthy = (
            not m0.is_empty and not m0.is_oversized
            and m0.n_components >= 1 and m0.bbox_agreement_iou >= 0.85
        )

        if not iter0_healthy:
            # Fall back to pure proxy scoring when iter 0 isn't trustworthy.
            return max(self.iterations, key=lambda it: (score(it), -it.iteration))

        best = it0
        margin = 1.05  # need 5% improvement on both axes
        for it in self.iterations[1:]:
            m = it.metrics
            if m.is_empty or m.is_oversized or m.n_components == 0:
                continue
            if (m.bbox_agreement_iou >= m0.bbox_agreement_iou * margin
                    and m.compactness >= m0.compactness * margin
                    and score(it) > score(best)):
                best = it
        return best

    def fail(self, reason: str) -> "AgentState":
        self.stop_reason = reason
        return self

    def to_dict(self) -> dict:
        best = self.best_iter()
        return {
            "run_id": self.run_id,
            "image_path": self.image_path,
            "raw_user_text": self.raw_user_text,
            "refined_prompt": self.refined_prompt,
            "synonyms": list(self.synonyms),
            "candidate_boxes": [b.to_dict() for b in self.candidate_boxes],
            "iterations": [it.to_dict() for it in self.iterations],
            "stop_reason": self.stop_reason,
            "best_iter": best.iteration if best else None,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
