"""FSDP per-rank checkpoint save/load for DreamZero VLA (sequential DiT block wrap)."""

from __future__ import annotations

import gc
import json
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

CHECKPOINT_VERSION = 1
REPLICATED_FILENAME = "replicated.safetensors"
METADATA_FILENAME = "metadata.json"
RANK_DIR_PREFIX = "rank_"


def get_sharded_checkpoint_dir() -> str | None:
    path = (
        os.environ.get("VAMVERL_FSDP_SHARDED_CHECKPOINT")
        or os.environ.get("VAMPO_FSDP_SHARDED_CHECKPOINT", "")
    ).strip()
    if not path:
        return None
    meta = os.path.join(path, METADATA_FILENAME)
    return path if os.path.isfile(meta) else None


def use_fsdp_sharded_checkpoint() -> bool:
    return get_sharded_checkpoint_dir() is not None


def _rank_dir(checkpoint_dir: str, rank: int | None = None) -> str:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return os.path.join(checkpoint_dir, f"{RANK_DIR_PREFIX}{rank}")


def _block_filename(block_idx: int) -> str:
    return f"dit_block_{block_idx:02d}.pt"


# DiT linear targets wrapped by PEFT LoRA (dense ckpt uses .weight; LoRA model uses .base_layer.weight).
_LORA_DIT_KEY_PREFIXES = (
    "self_attn.q.",
    "self_attn.k.",
    "self_attn.v.",
    "self_attn.o.",
    "cross_attn.q.",
    "cross_attn.k.",
    "cross_attn.v.",
    "cross_attn.o.",
    "ffn.0.",
    "ffn.2.",
)


def _is_dense_lora_target_key(key: str) -> bool:
    if ".base_layer." in key or ".lora_" in key:
        return False
    if not (key.endswith(".weight") or key.endswith(".bias")):
        return False
    return any(key.startswith(prefix) for prefix in _LORA_DIT_KEY_PREFIXES)


def remap_dense_dit_shard_for_lora(shard_sd: dict[str, Any]) -> dict[str, Any]:
    """Map dense DiT block shard keys to PEFT base_layer keys for RL LoRA training."""
    if not any(_is_dense_lora_target_key(k) for k in shard_sd):
        return shard_sd
    remapped: dict[str, Any] = {}
    for key, value in shard_sd.items():
        if _is_dense_lora_target_key(key):
            param_name = key.rsplit(".", 1)[1]
            module_name = key[: -(len(param_name) + 1)]
            remapped[f"{module_name}.base_layer.{param_name}"] = value
        else:
            remapped[key] = value
    return remapped


def _vla_dit_uses_lora(vla_module: nn.Module) -> bool:
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    for wan_model in vla_module.modules():
        if isinstance(wan_model, CausalWanModel):
            for block in wan_model.blocks:
                inner = block.module if isinstance(block, FSDP) else block
                return any("base_layer" in name for name, _ in inner.named_parameters())
    return False


def _iter_fsdp_dit_blocks(vla_module: nn.Module):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    for wan_model in vla_module.modules():
        if isinstance(wan_model, CausalWanModel):
            for i, block in enumerate(wan_model.blocks):
                if isinstance(block, FSDP):
                    yield wan_model, i, block


def _fsdp_param_ids(vla_module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for _, _, block in _iter_fsdp_dit_blocks(vla_module):
        for p in block.parameters():
            ids.add(id(p))
        for b in block.buffers():
            ids.add(id(b))
    return ids


def collect_replicated_state_dict(vla_module: nn.Module) -> dict[str, torch.Tensor]:
    """Non-FSDP DiT weights (T5/VAE/projector/LoRA/etc.) replicated on every rank."""
    fsdp_ids = _fsdp_param_ids(vla_module)
    state: dict[str, torch.Tensor] = {}
    for name, param in vla_module.named_parameters():
        if id(param) not in fsdp_ids:
            state[name] = param.detach().cpu().contiguous()
    for name, buf in vla_module.named_buffers():
        if id(buf) not in fsdp_ids:
            state[name] = buf.detach().cpu().contiguous()
    return state


def verify_sharded_checkpoint(checkpoint_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(checkpoint_dir, METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    world_size = int(meta["world_size"])
    if dist.is_initialized() and dist.get_world_size() != world_size:
        raise RuntimeError(
            f"Sharded checkpoint world_size={world_size} != dist world_size={dist.get_world_size()}"
        )
    rep_path = os.path.join(checkpoint_dir, REPLICATED_FILENAME)
    if not os.path.isfile(rep_path):
        raise FileNotFoundError(f"Missing {rep_path}")
    n_blocks = int(meta.get("n_dit_blocks", 0))
    for rank in range(world_size):
        rank_dir = _rank_dir(checkpoint_dir, rank)
        if not os.path.isdir(rank_dir):
            raise FileNotFoundError(f"Missing rank directory {rank_dir}")
        if n_blocks > 0:
            for i in range(n_blocks):
                block_path = os.path.join(rank_dir, _block_filename(i))
                if not os.path.isfile(block_path):
                    raise FileNotFoundError(f"Missing {block_path}")
    return meta


def load_replicated_state_dict(vla_module: nn.Module, checkpoint_dir: str) -> None:
    """Each rank loads full replicated weights (T5/VAE/projector); DiT comes from per-rank shards."""
    verify_sharded_checkpoint(checkpoint_dir)
    rep_path = os.path.join(checkpoint_dir, REPLICATED_FILENAME)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[FSDP shard] rank{rank}: loading replicated weights from {rep_path}", flush=True)
    state = load_file(rep_path, device="cpu")
    missing, unexpected = vla_module.load_state_dict(state, strict=False)
    del state
    gc.collect()
    if rank == 0:
        print(
            f"[FSDP shard] replicated load: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )


def save_fsdp_dit_block_shards(vla_module: nn.Module, checkpoint_dir: str) -> int:
    """Save one FSDP shard file per DiT block per rank."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_dir = _rank_dir(checkpoint_dir, rank)
    os.makedirs(rank_dir, exist_ok=True)
    n_saved = 0
    for _, block_idx, block in _iter_fsdp_dit_blocks(vla_module):
        out_path = os.path.join(rank_dir, _block_filename(block_idx))
        with FSDP.state_dict_type(
            block,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            shard_sd = block.state_dict()
        torch.save(shard_sd, out_path)
        n_saved += 1
        del shard_sd
        gc.collect()
    if rank == 0:
        print(f"[FSDP shard] saved {n_saved} DiT block shards per rank under {checkpoint_dir}", flush=True)
    return n_saved


def load_fsdp_dit_block_shards(vla_module: nn.Module, checkpoint_dir: str) -> int:
    """Each rank loads only its local FSDP DiT block shards."""
    meta = verify_sharded_checkpoint(checkpoint_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_dir = _rank_dir(checkpoint_dir, rank)
    n_loaded = 0
    remap_lora = _vla_dit_uses_lora(vla_module)
    if remap_lora and rank == 0:
        print("[FSDP shard] remapping dense DiT shards → LoRA base_layer keys", flush=True)
    for _, block_idx, block in _iter_fsdp_dit_blocks(vla_module):
        in_path = os.path.join(rank_dir, _block_filename(block_idx))
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"rank{rank} missing block shard {in_path}")
        shard_sd = torch.load(in_path, map_location="cpu", weights_only=True)
        if remap_lora:
            shard_sd = remap_dense_dit_shard_for_lora(shard_sd)
        with FSDP.state_dict_type(
            block,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            missing, unexpected = block.load_state_dict(shard_sd, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys loading {in_path}: {unexpected[:8]}"
                    + (f" ... (+{len(unexpected) - 8})" if len(unexpected) > 8 else "")
                )
            lora_missing = [k for k in missing if ".lora_" in k]
            other_missing = [k for k in missing if ".lora_" not in k]
            if other_missing:
                raise RuntimeError(
                    f"Missing non-LoRA keys loading {in_path}: {other_missing[:8]}"
                    + (f" ... (+{len(other_missing) - 8})" if len(other_missing) > 8 else "")
                )
            if lora_missing and rank == 0 and block_idx == 0:
                print(
                    f"[FSDP shard] dense→LoRA remap: skipped {len(lora_missing)} "
                    "uninitialized lora keys (expected)",
                    flush=True,
                )
        n_loaded += 1
        del shard_sd
        gc.collect()
    print(
        f"[FSDP shard] rank{rank}: loaded {n_loaded} DiT block shards "
        f"(world_size={meta['world_size']})",
        flush=True,
    )
    return n_loaded


def save_fsdp_sharded_checkpoint(
    vla_module: nn.Module,
    checkpoint_dir: str,
    *,
    source_model_path: str,
    wrap_mode: str = "sequential_dit_blocks",
) -> None:
    """Save replicated + per-rank DiT FSDP shards. Call after FSDP wrap."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    os.makedirs(checkpoint_dir, exist_ok=True)

    replicated = collect_replicated_state_dict(vla_module)
    rep_path = os.path.join(checkpoint_dir, REPLICATED_FILENAME)
    if rank == 0:
        save_file(replicated, rep_path)
        meta = {
            "version": CHECKPOINT_VERSION,
            "world_size": world_size,
            "wrap_mode": wrap_mode,
            "source_model_path": source_model_path,
            "n_dit_blocks": sum(1 for _ in _iter_fsdp_dit_blocks(vla_module)),
        }
        with open(os.path.join(checkpoint_dir, METADATA_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[FSDP shard] rank0: saved replicated ({len(replicated)} tensors) → {rep_path}", flush=True)
    del replicated
    gc.collect()

    if dist.is_initialized():
        dist.barrier()

    save_fsdp_dit_block_shards(vla_module, checkpoint_dir)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"[FSDP shard] checkpoint ready at {checkpoint_dir}", flush=True)
