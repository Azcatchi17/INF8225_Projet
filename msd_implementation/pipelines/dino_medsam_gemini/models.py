"""Module-level singletons for DINO / MedSAM / Gemini so that re-running
the 'load models' cell in the notebook is instant (the kernel keeps them
alive across cell reruns)."""
from __future__ import annotations

import os
from typing import Any

import torch

from . import config

_CACHE: dict[str, Any] = {}


def _device() -> str:
    return config.DEVICE if torch.cuda.is_available() else "cpu"


def get_dino_model():
    if "dino" in _CACHE:
        return _CACHE["dino"]

    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    model = init_detector(
        str(config.DINO_CONFIG),
        str(config.DINO_CHECKPOINT),
        device=_device(),
    )
    _CACHE["dino"] = model
    return model


def get_medsam_model():
    if "medsam" in _CACHE:
        return _CACHE["medsam"]

    from MedSAM.segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(config.MEDSAM_CHECKPOINT))
    model = model.to(_device())
    model.eval()
    _CACHE["medsam"] = model
    return model


def get_gemini_client():
    if "gemini" in _CACHE:
        return _CACHE["gemini"]

    from .gemini import GeminiClient

    api_key = _resolve_api_key()
    client = GeminiClient(api_key=api_key, model_id=config.GEMINI_MODEL_ID)
    _CACHE["gemini"] = client
    return client


def _resolve_api_key() -> str:
    key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if key:
        return key

    colab_error: Exception | None = None
    try:  # Colab Secrets
        from google.colab import userdata  # type: ignore

        key = userdata.get(config.GEMINI_API_KEY_ENV)
        if key:
            os.environ[config.GEMINI_API_KEY_ENV] = key
            return key
    except Exception as exc:
        colab_error = exc

    # In Colab notebooks, fall back to a hidden prompt so the user can
    # continue without restarting after adding a secret.
    try:
        from getpass import getpass

        key = getpass(
            f"Enter {config.GEMINI_API_KEY_ENV} "
            "(input hidden, leave empty to abort): "
        ).strip()
        if key:
            os.environ[config.GEMINI_API_KEY_ENV] = key
            return key
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        pass

    detail = ""
    if colab_error is not None:
        detail = f" Colab secret lookup failed with {type(colab_error).__name__}."
    raise RuntimeError(
        f"No {config.GEMINI_API_KEY_ENV}: set it as a Colab secret "
        f"(left sidebar), export it as an env var locally, or enter it "
        f"when prompted.{detail}"
    )


def reset_cache() -> None:
    """Drop all cached models (useful if swapping checkpoints)."""
    _CACHE.clear()
