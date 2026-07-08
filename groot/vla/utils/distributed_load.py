"""Helpers for rank-aware checkpoint loading under torch.distributed."""

from __future__ import annotations

import os

import torch.distributed as dist


def parallel_strategy() -> str:
    return (
        os.environ.get("VAMVERL_PARALLEL_STRATEGY")
        or os.environ.get("VAMPO_PARALLEL_STRATEGY", "")
    ).lower()


def is_dist_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_sharded_checkpoint_dir() -> str | None:
    from groot.vla.utils.fsdp_sharded_checkpoint import get_sharded_checkpoint_dir as _get

    return _get()


def use_fsdp_sharded_checkpoint() -> bool:
    from groot.vla.utils.fsdp_sharded_checkpoint import use_fsdp_sharded_checkpoint as _use

    return _use()


def set_sharded_checkpoint_dir(path: str | None) -> None:
    if path:
        os.environ["VAMVERL_FSDP_SHARDED_CHECKPOINT"] = path
    else:
        os.environ.pop("VAMVERL_FSDP_SHARDED_CHECKPOINT", None)
        os.environ.pop("VAMPO_FSDP_SHARDED_CHECKPOINT", None)


def should_load_checkpoint_weights() -> bool:
    """Full safetensors load (rank0 only). Disabled when per-rank FSDP shards are configured."""
    if use_fsdp_sharded_checkpoint():
        return False
    return is_dist_rank0()


def should_load_replicated_checkpoint() -> bool:
    """Each rank loads replicated.safetensors (T5/VAE/projector) before FSDP DiT shard load."""
    return use_fsdp_sharded_checkpoint()


def should_load_local_component_weights() -> bool:
    """Load T5/VAE/DiT component files from disk (rank0 legacy path)."""
    if use_fsdp_sharded_checkpoint():
        return False
    return should_load_checkpoint_weights()


def use_fsdp_meta_init() -> bool:
    """Meta init: all ranks when sharded checkpoint; rank>0 only for legacy broadcast path."""
    if parallel_strategy() == "ddp":
        return False
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return False
    if use_fsdp_sharded_checkpoint():
        return True
    return dist.get_rank() != 0


def fsdp_rank0_load_device() -> str:
    """FSDP: rank0 loads on CPU; weights move to GPU during FSDP wrap (avoids 42GiB+ flatten OOM/hang)."""
    if should_load_checkpoint_weights() and dist.is_initialized() and dist.get_world_size() > 1:
        return "cpu"
    return "cpu"
