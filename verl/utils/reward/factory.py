"""Build reward model from verl / yaml config."""

from __future__ import annotations

import os
from typing import Any

from verl.utils.reward.base import BaseRewardModel


def _reward_cfg(cfg: dict[str, Any] | Any) -> dict[str, Any]:
    if hasattr(cfg, "get"):
        rm_cfg = cfg.get("reward", {})
    else:
        rm_cfg = dict(cfg)
    if hasattr(rm_cfg, "items"):
        return dict(rm_cfg)
    return rm_cfg


def build_reward_model(cfg: dict[str, Any] | Any) -> BaseRewardModel:
    """Instantiate VideoMAE success classifier from ``reward`` config block."""
    rm_cfg = _reward_cfg(cfg)

    backend = str(rm_cfg.get("backend", "videomae")).lower()
    if backend not in {"videomae", "video_mae"}:
        raise ValueError(
            f"Unsupported reward.backend={backend!r}; only videomae is supported"
        )
    return _build_videomae(rm_cfg)


def _build_videomae(rm_cfg: dict[str, Any]):
    from verl.utils.reward.videomae_reward import (
        DEFAULT_VIDEOMAE_BACKBONE,
        VideoMAERewardModel,
    )

    ckpt = (
        rm_cfg.get("videomae_checkpoint")
        or rm_cfg.get("checkpoint_path")
        or os.environ.get("VIDEOMAE_CKPT")
    )
    if not ckpt:
        raise ValueError(
            "reward requires videomae_checkpoint or env VIDEOMAE_CKPT"
        )
    threshold = rm_cfg.get("rm_threshold")
    if threshold is None:
        threshold = rm_cfg.get("videomae_threshold")
    hf_model_id = (
        rm_cfg.get("hf_model_id")
        or rm_cfg.get("hf_model_path")
        or os.environ.get("VIDEOMAE_BACKBONE")
        or DEFAULT_VIDEOMAE_BACKBONE
    )
    return VideoMAERewardModel(
        checkpoint_path=str(ckpt),
        threshold=float(threshold) if threshold is not None else None,
        img_size=int(rm_cfg.get("img_size", 224)),
        window_size=int(rm_cfg.get("window_size", 8)),
        min_steps=int(rm_cfg.get("min_steps", 32)),
        batch_size=int(rm_cfg.get("batch_size", 32)),
        device=rm_cfg.get("device"),
        hf_model_id=str(hf_model_id),
    )
