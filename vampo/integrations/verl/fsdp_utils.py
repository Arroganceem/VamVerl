"""FSDP helpers for DreamZero / VAMPO (4-GPU FULL_SHARD across Ray workers)."""

from __future__ import annotations

import functools
import gc
import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from vampo.integrations.verl.parallel_utils import get_actor_strategy

logger = logging.getLogger(__name__)

_fsdp_wrap_unit_idx = 0


def _reset_wrap_counter() -> None:
    global _fsdp_wrap_unit_idx
    _fsdp_wrap_unit_idx = 0


def _log_fsdp_mem(tag: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"[FSDP init] rank0: {tag} · cuda alloc={alloc:.1f}GiB reserved={reserved:.1f}GiB",
            flush=True,
        )


def init_fn(module: nn.Module) -> nn.Module:
    """Non-rank0 empty init for FSDP sync_module_states (avoid verl OpenVLA/timm import)."""
    global _fsdp_wrap_unit_idx
    rank = dist.get_rank()
    _fsdp_wrap_unit_idx += 1
    if rank == 0:
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        print(
            f"[FSDP init] rank0: wrap unit {_fsdp_wrap_unit_idx} "
            f"{module.__class__.__name__} ({n_params / 1e6:.1f}M params)",
            flush=True,
        )
    if rank != 0:
        module.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return module


def init_fsdp_device_mesh() -> DeviceMesh:
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size()
    return init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])


def fsdp_enabled(config) -> bool:
    strategy = get_actor_strategy(config)
    fsdp_cfg = getattr(getattr(config, "actor", config), "fsdp_config", None)
    if fsdp_cfg and fsdp_cfg.get("disable", False):
        return False
    return strategy == "fsdp" and dist.is_initialized() and dist.get_world_size() > 1


def get_dreamzero_fsdp_wrap_policy(vla_module: nn.Module, fsdp_config: dict | None):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanAttentionBlock,
        CausalWanModel,
    )

    fsdp_config = fsdp_config or {}
    if fsdp_config.get("disable", False):
        return None

    _BUILTIN_WRAP = {
        "CausalWanAttentionBlock": CausalWanAttentionBlock,
        "CausalWanModel": CausalWanModel,
    }
    layer_names = fsdp_config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", ["CausalWanAttentionBlock"]
    )
    layer_cls = set()
    for name in layer_names:
        if name in _BUILTIN_WRAP:
            layer_cls.add(_BUILTIN_WRAP[name])
        else:
            from transformers.trainer_pt_utils import get_module_class_from_name

            cls = get_module_class_from_name(vla_module, name)
            if cls is None:
                raise RuntimeError(f"FSDP wrap class not found in VLA: {name}")
            layer_cls.add(cls)

    if not layer_cls:
        return None
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


def _fix_scalar_params_for_fsdp(module: nn.Module) -> None:
    """FSDP rejects 0-d Parameters (e.g. CLIP log_scale); promote to shape (1,)."""
    for mod in module.modules():
        for pname, param in list(mod._parameters.items()):
            if param is not None and param.ndim == 0:
                mod._parameters[pname] = nn.Parameter(param.reshape(1))


def _iter_causal_wan_models(root: nn.Module):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanModel,
    )

    for module in root.modules():
        if isinstance(module, CausalWanModel):
            yield module


def _wrap_dit_blocks_sequential(
    vla_module: nn.Module,
    *,
    sharding_strategy: ShardingStrategy,
    mixed_precision: MixedPrecision,
    cpu_offload: CPUOffload | None,
    device_mesh: DeviceMesh,
) -> int:
    """Wrap each DiT block as its own FSDP unit (~800MB) instead of one 30GB flat param."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanAttentionBlock,
    )

    rank = dist.get_rank()
    wrapped = 0
    for wan_model in _iter_causal_wan_models(vla_module):
        n_blocks = len(wan_model.blocks)
        for i in range(n_blocks):
            block = wan_model.blocks[i]
            if isinstance(block, FSDP):
                continue
            if not isinstance(block, CausalWanAttentionBlock):
                continue
            t0 = time.monotonic()
            if rank == 0:
                n_params = sum(p.numel() for p in block.parameters())
                print(
                    f"[FSDP init] rank0: DiT block {i + 1}/{n_blocks} start "
                    f"({n_params / 1e6:.1f}M params)",
                    flush=True,
                )
            wan_model.blocks[i] = FSDP(
                block,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                cpu_offload=cpu_offload,
                sync_module_states=True,
                param_init_fn=init_fn,
                use_orig_params=True,
                limit_all_gathers=True,
                device_mesh=device_mesh,
            )
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            wrapped += 1
            if rank == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"[FSDP init] rank0: DiT block {i + 1}/{n_blocks} done ({elapsed:.1f}s)",
                    flush=True,
                )
                _log_fsdp_mem(f"after block {i + 1}/{n_blocks}")
    return wrapped


def prepare_vla_for_fsdp_wrap(vla_module: nn.Module) -> None:
    """Free cache before FSDP wrap. Do not move the full 14B model to CUDA on rank0 first."""
    rank = dist.get_rank()
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        on_cuda = next(vla_module.parameters(), None) is not None and next(
            vla_module.parameters()
        ).is_cuda
        print(
            f"[FSDP init] rank0: pre-wrap model on_cuda={on_cuda} "
            f"(weights sync during FSDP constructor)",
            flush=True,
        )
    _log_fsdp_mem("ready for FSDP wrap")
    print(
        f"[FSDP init] rank{rank}: ready for FSDP wrap "
        f"(rank0 loads CPU → GPU via sync_module_states)",
        flush=True,
    )


def wrap_vla_fsdp(
    vla_module: nn.Module,
    *,
    fsdp_config: dict | None,
    device_mesh: DeviceMesh,
) -> FSDP:
    fsdp_config = fsdp_config or {}
    sequential_block_wrap = bool(fsdp_config.get("sequential_block_wrap", False))
    _fix_scalar_params_for_fsdp(vla_module)

    if sequential_block_wrap:
        # Blocks pre-wrapped; outer policy only wraps CausalWanModel shells (+ backbone units).
        outer_cfg = {
            **fsdp_config,
            "wrap_policy": {"transformer_layer_cls_to_wrap": ["CausalWanModel"]},
        }
        auto_wrap_policy = get_dreamzero_fsdp_wrap_policy(vla_module, outer_cfg)
        wrap_names = ["CausalWanAttentionBlock (sequential)", "CausalWanModel (outer)"]
    else:
        auto_wrap_policy = get_dreamzero_fsdp_wrap_policy(vla_module, fsdp_config)
        wrap_names = fsdp_config.get("wrap_policy", {}).get(
            "transformer_layer_cls_to_wrap", ["CausalWanAttentionBlock"]
        )

    sharding_strategy = (
        ShardingStrategy.FULL_SHARD if auto_wrap_policy is not None else ShardingStrategy.SHARD_GRAD_OP
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    rank = dist.get_rank()
    print(
        f"[FSDP init] rank{rank}: starting FSDP wrap "
        f"(layers={wrap_names}, world_size={dist.get_world_size()}, "
        f"sequential_block_wrap={sequential_block_wrap})",
        flush=True,
    )
    _reset_wrap_counter()
    gc.collect()
    torch.cuda.empty_cache()
    _log_fsdp_mem("before wrap")

    use_cpu_offload = bool(fsdp_config.get("param_offload", False)) and bool(
        fsdp_config.get("allow_cpu_offload", False)
    )
    cpu_offload = CPUOffload(offload_params=True) if use_cpu_offload else None

    if sequential_block_wrap:
        n_blocks = _wrap_dit_blocks_sequential(
            vla_module,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            device_mesh=device_mesh,
        )
        if rank == 0:
            print(f"[FSDP init] rank0: sequential DiT blocks wrapped: {n_blocks}", flush=True)
        _log_fsdp_mem("after sequential blocks")
        # Blocks already synced per-block; outer wrap only shards shells + encoders.
        outer_sync = False
    else:
        outer_sync = True

    if rank == 0:
        print("[FSDP init] rank0: entering FSDP constructor (may take several minutes)...", flush=True)
    t_wrap = time.monotonic()
    fsdp_module = FSDP(
        vla_module,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        sync_module_states=outer_sync,
        param_init_fn=init_fn,
        use_orig_params=True,
        limit_all_gathers=True,
        device_mesh=device_mesh,
    )
    if rank == 0:
        print(
            f"[FSDP init] rank0: FSDP constructor done ({time.monotonic() - t_wrap:.1f}s)",
            flush=True,
        )
    _log_fsdp_mem("after full wrap")
    logger.info(
        "Wrapped VLA with FSDP (%s, world_size=%s, param_offload=%s, sequential=%s)",
        sharding_strategy,
        dist.get_world_size(),
        bool(cpu_offload),
        sequential_block_wrap,
    )
    print(
        f"[FSDP init] rank{dist.get_rank()}: Wrapped VLA with FSDP "
        f"({sharding_strategy}, param_offload={bool(cpu_offload)}, "
        f"sequential_block_wrap={sequential_block_wrap})",
        flush=True,
    )
    return fsdp_module


def fsdp_post_initialize(vla_module: nn.Module) -> None:
    """Call VLA.post_initialize on FSDP-wrapped or plain module."""
    if isinstance(vla_module, FSDP):
        vla_module.module.post_initialize()
    elif hasattr(vla_module, "post_initialize"):
        vla_module.post_initialize()
