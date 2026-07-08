"""Distributed strategy helpers for VAMPO + verl (FSDP only)."""

from __future__ import annotations

import os
import zlib

import torch
from omegaconf import OmegaConf


def get_actor_strategy(config) -> str:
    """Resolve actor parallel strategy from Hydra config."""
    for path in (
        "actor.strategy",
        "actor_rollout_ref.actor.strategy",
        "strategy",
    ):
        val = OmegaConf.select(config, path, default=None)
        if val is None:
            continue
        strategy = str(val).lower()
        if strategy and strategy not in ("none", ""):
            return strategy
    return "fsdp"


def seed_rollout_sample(group_uid: str, sample_idx: int) -> None:
    """Deterministic per-sample RNG seed for reproducible rollouts."""
    key = f"{group_uid}:{sample_idx}".encode()
    seed = int(zlib.crc32(key) & 0x7FFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_parallel_strategy_env(config) -> None:
    from verl.utils.vamverl_env import PARALLEL_STRATEGY

    strategy = get_actor_strategy(config)
    os.environ[PARALLEL_STRATEGY] = strategy if strategy in ("fsdp", "ddp") else "fsdp"
