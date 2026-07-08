"""DreamZero data-backend helpers for RayTrainer."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def get_data_backend(config: DictConfig) -> str:
    return str(OmegaConf.select(config, "data.backend", default="wmpo"))


def is_dreamzero_backend(config: DictConfig) -> bool:
    """True for DreamZero/VAMPO RL pipeline (``dreamzero`` or legacy ``vampo``)."""
    return get_data_backend(config) in ("dreamzero", "vampo")
