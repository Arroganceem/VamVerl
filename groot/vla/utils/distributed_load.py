"""Helpers for rank-aware checkpoint loading under torch.distributed."""

from __future__ import annotations

import os

import torch.distributed as dist


def parallel_strategy() -> str:
    return os.environ.get("VAMPO_PARALLEL_STRATEGY", "").lower()


def is_dist_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def should_load_checkpoint_weights() -> bool:
    """FSDP: rank0 loads checkpoint; other ranks meta-init then sync_module_states."""
    return is_dist_rank0()


def use_fsdp_meta_init() -> bool:
    """Rank>0 meta init for FSDP broadcast path."""
    if parallel_strategy() == "ddp":
        return False
    return dist.is_initialized() and dist.get_world_size() > 1 and dist.get_rank() != 0


def fsdp_rank0_load_device() -> str:
    """FSDP: rank0 loads on CPU; weights move to GPU during FSDP wrap (avoids 42GiB+ flatten OOM/hang)."""
    if should_load_checkpoint_weights() and dist.is_initialized() and dist.get_world_size() > 1:
        return "cpu"
    return "cpu"
