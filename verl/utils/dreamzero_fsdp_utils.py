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

from verl.utils.dreamzero_parallel import get_actor_strategy

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


def init_fn_all_empty(module: nn.Module) -> nn.Module:
    """All ranks empty init; used with per-rank sharded checkpoint load (no rank0 broadcast)."""
    global _fsdp_wrap_unit_idx
    rank = dist.get_rank()
    _fsdp_wrap_unit_idx += 1
    if rank == 0:
        print(
            f"[FSDP init] rank0: wrap unit {_fsdp_wrap_unit_idx} "
            f"{module.__class__.__name__} (sharded empty init)",
            flush=True,
        )
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
    sync_module_states: bool = True,
    param_init_fn=None,
) -> int:
    """Wrap each DiT block as its own FSDP unit; rank0 peak ~800MB/block vs ~50GiB flat."""
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
                sync_module_states=sync_module_states,
                param_init_fn=param_init_fn or init_fn,
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
    from groot.vla.utils.distributed_load import use_fsdp_sharded_checkpoint

    rank = dist.get_rank()
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        on_cuda = next(vla_module.parameters(), None) is not None and next(
            vla_module.parameters()
        ).is_cuda
        if use_fsdp_sharded_checkpoint():
            hint = "replicated.safetensors loaded; DiT from per-rank shards after block wrap"
        else:
            hint = "weights sync during FSDP constructor"
        print(
            f"[FSDP init] rank0: pre-wrap model on_cuda={on_cuda} ({hint})",
            flush=True,
        )
    _log_fsdp_mem("ready for FSDP wrap")
    if use_fsdp_sharded_checkpoint():
        print(
            f"[FSDP init] rank{rank}: ready for FSDP wrap (per-rank sharded DiT load)",
            flush=True,
        )
    else:
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
) -> nn.Module:
    from groot.vla.utils.distributed_load import use_fsdp_sharded_checkpoint

    fsdp_config = fsdp_config or {}
    sequential_block_wrap = bool(fsdp_config.get("sequential_block_wrap", False))
    sharded_load = use_fsdp_sharded_checkpoint()
    sharded_dir_cfg = fsdp_config.get("sharded_checkpoint_dir")
    if sharded_dir_cfg and not sharded_load:
        from groot.vla.utils.distributed_load import set_sharded_checkpoint_dir

        set_sharded_checkpoint_dir(str(sharded_dir_cfg))
        sharded_load = use_fsdp_sharded_checkpoint()
    if sharded_load and not sequential_block_wrap:
        raise RuntimeError(
            "fsdp_config.sharded_checkpoint_dir requires sequential_block_wrap: true"
        )
    _fix_scalar_params_for_fsdp(vla_module)

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
    if sharded_load:
        mode = "sequential_dit_blocks+sharded_ckpt"
    else:
        mode = "sequential_dit_blocks" if sequential_block_wrap else "standard_auto_wrap"
    print(
        f"[FSDP init] rank{rank}: starting FSDP wrap "
        f"(layers={wrap_names}, world_size={dist.get_world_size()}, mode={mode})",
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
        block_sync = not sharded_load
        block_init_fn = init_fn_all_empty if sharded_load else init_fn
        n_blocks = _wrap_dit_blocks_sequential(
            vla_module,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            device_mesh=device_mesh,
            sync_module_states=block_sync,
            param_init_fn=block_init_fn,
        )
        if rank == 0:
            print(f"[FSDP init] rank0: sequential DiT blocks wrapped: {n_blocks}", flush=True)
        _log_fsdp_mem("after sequential blocks")
        if sharded_load:
            from groot.vla.utils.fsdp_sharded_checkpoint import (
                get_sharded_checkpoint_dir,
                load_fsdp_dit_block_shards,
            )

            ckpt_dir = get_sharded_checkpoint_dir()
            assert ckpt_dir is not None
            load_fsdp_dit_block_shards(vla_module, ckpt_dir)
            dist.barrier()
            print(
                f"[FSDP init] rank{rank}: sharded DiT load done "
                f"(blocks={n_blocks}, per-rank shards)",
                flush=True,
            )
        else:
            print(
                f"[FSDP init] rank{rank}: sequential DiT-only FSDP done "
                f"(blocks={n_blocks}, skip outer wrap; VAE/T5 stay local per rank)",
                flush=True,
            )
        return vla_module

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
        sync_module_states=True,
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
    _log_fsdp_mem("after wrap")
    logger.info(
        "Wrapped VLA with FSDP (%s, world_size=%s, param_offload=%s)",
        sharding_strategy,
        dist.get_world_size(),
        bool(cpu_offload),
    )
    print(
        f"[FSDP init] rank{dist.get_rank()}: Wrapped VLA with FSDP "
        f"({sharding_strategy}, param_offload={bool(cpu_offload)})",
        flush=True,
    )
    return fsdp_module


def fsdp_post_initialize(vla_module: nn.Module) -> None:
    """Call VLA.post_initialize on FSDP-wrapped or plain module."""
    if isinstance(vla_module, FSDP):
        vla_module.module.post_initialize()
    elif hasattr(vla_module, "post_initialize"):
        vla_module.post_initialize()


def clip_grad_norm_vla(vla_module: nn.Module, parameters, max_norm: float):
    """Clip grads for root FSDP or sequential DiT-only FSDP (nested blocks)."""
    if isinstance(vla_module, FSDP):
        return vla_module.clip_grad_norm_(max_norm=max_norm)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def offload_fsdp_grad(module: nn.Module) -> None:
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_grad(module: nn.Module, device_id) -> None:
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_param_and_grad(module: nn.Module, offload_grad: bool = False) -> None:
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        param.data = param.data.to("cpu", non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_param_and_grad(module: nn.Module, device_id, load_grad: bool = False) -> None:
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to(device_id, non_blocking=True)
        param.data = param.data.to(device_id, non_blocking=True)
        if load_grad and param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_optimizer(optimizer) -> None:
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_optimizer(optimizer, device_id) -> None:
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()
