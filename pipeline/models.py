"""Module-level singletons for DINO / MedSAM / Gemma so that re-running
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


def get_gemma_client():
    if "gemma" in _CACHE:
        return _CACHE["gemma"]

    from .gemma import GemmaClient

    api_key = _resolve_api_key()
    client = GemmaClient(api_key=api_key, model_id=config.GEMMA_MODEL_ID)
    _CACHE["gemma"] = client
    return client


def _resolve_api_key() -> str:
    key = os.environ.get(config.GEMMA_API_KEY_ENV)
    if key:
        return key
    try:  # Colab Secrets
        from google.colab import userdata  # type: ignore
        return userdata.get(config.GEMMA_API_KEY_ENV)
    except Exception:
        raise RuntimeError(
            f"No {config.GEMMA_API_KEY_ENV}: set it as a Colab secret "
            "(🔑 left pane) or export it as an env var locally."
        )


def reset_cache() -> None:
    """Drop all cached models (useful if swapping checkpoints)."""
    _CACHE.clear()
