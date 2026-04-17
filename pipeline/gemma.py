"""Gemma wrapper: prompt refinement (text) + mask analysis (multimodal).

Uses google-genai SDK against Google AI Studio. The API surface here is
intentionally tiny: two methods, both returning pydantic-validated dicts.
"""
from __future__ import annotations

import io
import json
import time
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from . import config


class PromptRefinement(BaseModel):
    search_text: str = Field(..., description="1-3 lowercase words")
    synonyms: list[str] = Field(default_factory=list, max_length=3)


class MaskAction(BaseModel):
    action: Literal[
        "stop", "add_positive", "add_negative",
        "expand_box", "shrink_box", "replace_box",
    ]
    params: dict = Field(default_factory=dict)
    rationale: str = ""


_REFINE_SYSTEM = (
    "You are a medical imaging assistant. Your job is to convert a user's "
    "instruction into a short Grounding DINO text prompt. "
    "Return STRICT JSON: {\"search_text\": <1-3 lowercase words>, "
    "\"synonyms\": [up to 3 alternate lowercase phrases]}. "
    "No markdown, no commentary."
)

_ANALYZE_SYSTEM = (
    "You are reviewing a medical image segmentation produced by MedSAM. "
    "You see two images: (1) the original, (2) the same image with the "
    "predicted mask overlaid in translucent red. Decide whether the mask "
    "is already correct or needs one correction. Respond with STRICT JSON: "
    "{\"action\": one of [stop, add_positive, add_negative, expand_box, "
    "shrink_box, replace_box], \"params\": {...}, \"rationale\": <short>}. "
    "Coordinates are in the ORIGINAL image pixel space. "
    "Use 'stop' when the mask looks correct or cannot clearly be improved."
)


def _overlay_rgba(image_np: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Original image with a translucent red tint where mask==1."""
    img = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    red = np.zeros((*mask.shape, 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = (mask.astype(bool) * 120).astype(np.uint8)
    overlay = Image.fromarray(red, mode="RGBA").resize(img.size)
    return Image.alpha_composite(img, overlay)


def _downscale_png(img: Image.Image, side: int) -> bytes:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > side:
        ratio = side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_json_loose(text: str) -> dict:
    """Gemma sometimes wraps JSON in ```json fences. Strip them."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start : end + 1]
    return json.loads(t)


class GemmaClient:
    def __init__(self, api_key: str, model_id: str) -> None:
        from google import genai  # imported lazily so non-Colab imports work

        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id

    # ------------------------------------------------------------------
    def refine_prompt(self, user_text: str) -> PromptRefinement:
        from google.genai import types

        user_msg = (
            "Dataset: Kvasir-SEG (colonoscopy polyps). "
            f"User instruction: {user_text}"
        )
        cfg = types.GenerateContentConfig(
            system_instruction=_REFINE_SYSTEM,
            response_mime_type="application/json",
            temperature=0.0,
        )
        raw = self._retry(
            lambda: self._client.models.generate_content(
                model=self._model_id, contents=user_msg, config=cfg,
            )
        )
        data = _parse_json_loose(raw.text or "")
        try:
            return PromptRefinement(**data)
        except ValidationError:
            # Fallback: echo the user text as-is so the pipeline still progresses
            return PromptRefinement(search_text=user_text.strip().lower()[:40])

    # ------------------------------------------------------------------
    def analyze_mask(
        self,
        image_np: np.ndarray,
        mask: np.ndarray,
        metrics: dict,
        refined_prompt: str,
        iteration: int,
        max_iter: int,
    ) -> MaskAction:
        from google.genai import types

        orig_png = _downscale_png(Image.fromarray(image_np), config.GEMMA_IMAGE_SIDE)
        overlay = _overlay_rgba(image_np, mask)
        overlay_png = _downscale_png(overlay, config.GEMMA_IMAGE_SIDE)

        instructions = (
            f"Target: {refined_prompt}. "
            f"Metrics (JSON): {json.dumps(metrics)}. "
            f"Iteration {iteration}/{max_iter}. "
            "Return ONE JSON object with keys action/params/rationale."
        )
        parts = [
            types.Part.from_bytes(data=orig_png, mime_type="image/png"),
            types.Part.from_bytes(data=overlay_png, mime_type="image/png"),
            types.Part.from_text(text=instructions),
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=_ANALYZE_SYSTEM,
            response_mime_type="application/json",
            temperature=0.2,
        )
        raw = self._retry(
            lambda: self._client.models.generate_content(
                model=self._model_id, contents=parts, config=cfg,
            )
        )
        try:
            return MaskAction(**_parse_json_loose(raw.text or ""))
        except (ValidationError, json.JSONDecodeError):
            return MaskAction(action="stop", rationale="parse_failure")

    # ------------------------------------------------------------------
    def _retry(self, fn, tries: int = 2, base_delay: float = 1.0):
        last_exc: Exception | None = None
        for attempt in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — any Gemma SDK error
                last_exc = e
                time.sleep(base_delay * (2 ** attempt))
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------
def smoke_test(client: GemmaClient, image_np: np.ndarray) -> dict:
    """Round-trip text + image to verify the key and model work."""
    ref = client.refine_prompt("find the polyp")
    dummy_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    dummy_mask[
        image_np.shape[0] // 3 : 2 * image_np.shape[0] // 3,
        image_np.shape[1] // 3 : 2 * image_np.shape[1] // 3,
    ] = 1
    act = client.analyze_mask(
        image_np, dummy_mask,
        metrics={"area_pct": 0.11, "is_empty": False, "is_oversized": False},
        refined_prompt=ref.search_text, iteration=1, max_iter=5,
    )
    return {"refine": ref.model_dump(), "analyze": act.model_dump()}
