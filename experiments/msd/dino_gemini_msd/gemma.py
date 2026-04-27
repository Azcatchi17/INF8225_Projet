"""Gemini wrapper: prompt refinement (text) + mask analysis (multimodal).

Uses google-genai SDK against Google AI Studio. The API surface here is
intentionally tiny: two methods, both returning pydantic-validated dicts.
"""
from __future__ import annotations

import io
import json
import traceback
import re
import time
from collections import deque
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


class DetectedBox(BaseModel):
    box_2d: list[float] = Field(..., min_length=4, max_length=4)
    score: float = 1.0


class DetectionResult(BaseModel):
    boxes: list[DetectedBox] = Field(default_factory=list)


_REFINE_SYSTEM = (
    "You are a medical imaging assistant. Your job is to convert a user's "
    "instruction into a short Grounding DINO text prompt. "
    "Return STRICT JSON: {\"search_text\": <1-3 lowercase words>, "
    "\"synonyms\": [up to 3 alternate lowercase phrases]}. "
    "No markdown, no commentary."
)

_ANALYZE_SYSTEM = (
    "You are an expert radiologist reviewing a medical image segmentation produced by MedSAM "
    "on an abdominal CT scan. You see two images: (1) the original CT, (2) the same image with the "
    "predicted mask overlaid in translucent red. "
    "CONTEXT: Pancreatic tumors typically appear as hypodense (darker gray) masses within the "
    "pancreatic parenchyma. CRITICAL: Do NOT confuse the tumor with intestinal gas or air, "
    "which appear pitch black. "
    "Decide whether the mask is already correct or needs one correction. Respond with STRICT JSON: "
    "{\"action\": one of [stop, add_positive, add_negative, expand_box, "
    "shrink_box, replace_box], \"params\": {...}, \"rationale\": <short>}. "
    "Coordinates are in the ORIGINAL image pixel space. "
    "Use 'stop' when the mask correctly covers the tumor or if no clear improvement can be made."
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


def _extract_retry_delay(exc: Exception) -> float | None:
    text = str(exc)
    patterns = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"'retryDelay': '([0-9]+(?:\.[0-9]+)?)s'",
        r'"retryDelay":\s*"([0-9]+(?:\.[0-9]+)?)s"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


class GemmaClient:
    def __init__(self, api_key: str, model_id: str) -> None:
        from google import genai  # imported lazily so non-Colab imports work

        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id
        self._prompt_cache: dict[str, PromptRefinement] = {}
        self._request_times: deque[float] = deque()

    # ------------------------------------------------------------------
    def refine_prompt(self, user_text: str) -> PromptRefinement:
        from google.genai import types

        cache_key = user_text.strip().lower()
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        user_msg = (
            "Dataset: MSD Pancreas (abdominal CT scans). "
            f"User instruction: {user_text}"
        )
        cfg = types.GenerateContentConfig(
            system_instruction=_REFINE_SYSTEM,
            response_mime_type="application/json",
            response_schema=PromptRefinement,
            # thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
        )
        try:
            raw = self._retry(
                lambda: self._client.models.generate_content(
                    model=self._model_id, contents=user_msg, config=cfg,
                )
            )
        except Exception as exc:
            print(
                f"[gemma] refine_prompt fallback after {type(exc).__name__}: "
                "using the raw user prompt",
                flush=True,
            )
            refined = PromptRefinement(search_text=user_text.strip().lower()[:40])
            self._prompt_cache[cache_key] = refined
            return refined
        parsed = getattr(raw, "parsed", None)
        if isinstance(parsed, PromptRefinement):
            self._prompt_cache[cache_key] = parsed
            return parsed
        try:
            refined = PromptRefinement(**_parse_json_loose(raw.text or ""))
        except (ValidationError, json.JSONDecodeError):
            # Fallback: echo the user text as-is so the pipeline still progresses
            refined = PromptRefinement(search_text=user_text.strip().lower()[:40])
        self._prompt_cache[cache_key] = refined
        return refined

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
            # thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        )
        try:
            raw = self._retry(
                lambda: self._client.models.generate_content(
                    model=self._model_id, contents=parts, config=cfg,
                )
            )
        except Exception as exc:
            print(
                f"[gemma] analyze_mask fallback after {type(exc).__name__}: "
                "returning stop",
                flush=True,
            )
            return MaskAction(
                action="stop", rationale=f"service_error:{type(exc).__name__}"
            )

        # --- PROTECTION SÉCURISÉE CONTRE ATTRIBUTEERROR ---
        try:
            # On utilise getattr pour éviter le crash si .text est manquant
            response_text = getattr(raw, "text", None)
            
            # Si .text est None, on tente d'extraire manuellement le premier candidat
            if response_text is None and hasattr(raw, "candidates") and raw.candidates:
                response_text = raw.candidates[0].content.parts[0].text
            
            if not response_text:
                return MaskAction(action="stop", rationale="empty_api_response")

            return MaskAction(**_parse_json_loose(response_text))
            
        except (ValidationError, json.JSONDecodeError, AttributeError, IndexError) as e:
            print(f"[gemma] parse_failure: {str(e)}")
            return MaskAction(action="stop", rationale=f"parse_failure:{type(e).__name__}")

    # ------------------------------------------------------------------
    def detect_boxes(
        self,
        image_path: str,
        text: str,
        score_threshold: float = 0.3,
        top_k: int = 5,
    ) -> list:
        """Gemini-native object detection. Returns list[Box] in pixel coords.

        Replaces Grounding DINO in the VLM-only variant. Uses Gemini's
        documented bbox convention: box_2d = [ymin, xmin, ymax, xmax] in
        the 0..1000 normalised range."""
        from google.genai import types
        from .state import Box

        img = Image.open(image_path)
        W, H = img.size
        png = _downscale_png(img, config.GEMMA_IMAGE_SIDE)

        system = (
            "You are a medical object detector for MSD Pancreas (abdominal CT "
            "scans). Detect ALL instances of the requested target (e.g., pancreatic tumor). "
            "Return STRICT JSON: {\"boxes\": [{\"box_2d\": [ymin, xmin, "
            "ymax, xmax], \"score\": <0-1 confidence>}, ...]}. "
            "Coordinates MUST be normalised to the 0-1000 range. "
            "Return an empty list if nothing matches or if you only see gas/healthy organs."
        )
        instructions = (
            f"Target: {text}. Return ONE JSON object with key 'boxes'."
        )
        parts = [
            types.Part.from_bytes(data=png, mime_type="image/png"),
            types.Part.from_text(text=instructions),
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=DetectionResult,
            # thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
        )
        try:
            raw = self._retry(
                lambda: self._client.models.generate_content(
                    model=self._model_id, contents=parts, config=cfg,
                )
            )
        except Exception as exc:
            print(
                f"[gemma] detect_boxes fallback after {type(exc).__name__}: "
                "returning no boxes",
                flush=True,
            )
            return []
        parsed = getattr(raw, "parsed", None)
        if isinstance(parsed, DetectionResult):
            items = [b.model_dump() for b in parsed.boxes]
        else:
            try:
                data = _parse_json_loose(raw.text or "")
                items = data.get("boxes", []) if isinstance(data, dict) else []
            except (ValueError, json.JSONDecodeError):
                return []

        out: list[Box] = []
        for item in items:
            box = item.get("box_2d")
            if not (isinstance(box, list) and len(box) == 4):
                continue
            score = float(item.get("score", 1.0))
            if score < score_threshold:
                continue
            ymin, xmin, ymax, xmax = [float(v) for v in box]
            out.append(Box(
                xyxy=(xmin / 1000.0 * W, ymin / 1000.0 * H,
                      xmax / 1000.0 * W, ymax / 1000.0 * H),
                score=score,
            ))

        out.sort(key=lambda b: -b.score)
        return out[:top_k]

    # ------------------------------------------------------------------
    def _wait_for_rate_limit(self) -> None:
        limit = config.GEMMA_MAX_REQUESTS_PER_MIN
        if limit <= 0:
            return

        now = time.monotonic()
        window = 60.0
        while self._request_times and now - self._request_times[0] >= window:
            self._request_times.popleft()

        if len(self._request_times) < limit:
            return

        sleep_for = window - (now - self._request_times[0]) + config.GEMMA_RETRY_BUFFER_SEC
        if sleep_for > 0:
            time.sleep(sleep_for)

        now = time.monotonic()
        while self._request_times and now - self._request_times[0] >= window:
            self._request_times.popleft()

    # ------------------------------------------------------------------
    def _retry(self, fn):
        last_exc: Exception | None = None
        for attempt in range(config.GEMMA_MAX_RETRIES):
            self._wait_for_rate_limit()
            try:
                result = fn()
                self._request_times.append(time.monotonic())
                return result
            except Exception as e:  # noqa: BLE001 — any Gemma SDK error
                last_exc = e
                traceback.print_exc()
                if attempt == config.GEMMA_MAX_RETRIES - 1:
                    break
                retry_delay = _extract_retry_delay(e)
                if retry_delay is None:
                    retry_delay = config.GEMMA_RETRY_BASE_DELAY * (2 ** attempt)
                retry_delay = min(retry_delay, config.GEMMA_RETRY_MAX_DELAY)
                print(
                    f"[gemma] attempt {attempt + 1}/{config.GEMMA_MAX_RETRIES} "
                    f"failed ({type(e).__name__}); sleeping {retry_delay:.1f}s",
                    flush=True,
                )
                time.sleep(retry_delay + config.GEMMA_RETRY_BUFFER_SEC)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------
def smoke_test(client: GemmaClient, image_np: np.ndarray) -> dict:
    """Round-trip text + image to verify the key and model work."""
    ref = client.refine_prompt("find the tumor")
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
