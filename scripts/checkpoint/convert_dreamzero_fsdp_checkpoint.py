#!/usr/bin/env python3
"""One-time convert DreamZero full checkpoint → FSDP per-rank DiT shards + replicated weights.

Rank0 loads the full model once, FSDP broadcast fills all ranks, then each rank writes
its 1/world_size DiT shards. Training can then init without rank0 holding the full 23B.

Example (single node, 4 GPU):
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 \\
    scripts/checkpoint/convert_dreamzero_fsdp_checkpoint.py \\
    --model-path /home/robotem/Models/DreamZero-DROID \\
    --output /home/robotem/Models/DreamZero-DROID-fsdp4

Example (4 nodes × 1 GPU):
  bash scripts/checkpoint/convert_dreamzero_fsdp_cluster4.sh
"""

from __future__ import annotations

import argparse
import gc
import os
from datetime import timedelta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DreamZero safetensors to FSDP per-rank DiT shards"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="DreamZero-DROID directory with model.safetensors.index.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for metadata.json, replicated.safetensors, rank_*/",
    )
    args = parser.parse_args()

    os.environ.pop("VAMPO_FSDP_SHARDED_CHECKPOINT", None)

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=180))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model_path = os.path.abspath(args.model_path)
    output_path = os.path.abspath(args.output)

    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        print(
            f"[convert] world_size={world_size} model={model_path} → {output_path}",
            flush=True,
        )

    from groot.vla.model.dreamzero.base_vla import VLA

    if rank == 0:
        print("[convert] loading full VLA on rank0 (one-time, ~50GiB CPU peak)...", flush=True)
    vla = VLA.from_pretrained(model_path)
    vla.eval()
    vla.requires_grad_(False)

    if rank == 0:
        n_params = sum(p.numel() for p in vla.parameters())
        print(f"[convert] VLA loaded ({n_params / 1e9:.2f}B params)", flush=True)

    from verl.utils.dreamzero_fsdp_utils import (
        init_fsdp_device_mesh,
        prepare_vla_for_fsdp_wrap,
        wrap_vla_fsdp,
    )

    fsdp_config = {
        "sequential_block_wrap": True,
        "wrap_policy": {"transformer_layer_cls_to_wrap": ["CausalWanAttentionBlock"]},
    }

    device_mesh = init_fsdp_device_mesh()
    prepare_vla_for_fsdp_wrap(vla)
    dist.barrier()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rank == 0:
        print("[convert] FSDP sequential wrap + sync_module_states broadcast...", flush=True)
    vla = wrap_vla_fsdp(vla, fsdp_config=fsdp_config, device_mesh=device_mesh)

    from groot.vla.utils.fsdp_sharded_checkpoint import (
        save_fsdp_sharded_checkpoint,
        verify_sharded_checkpoint,
    )

    save_fsdp_sharded_checkpoint(
        vla,
        output_path,
        source_model_path=model_path,
    )
    dist.barrier()

    if rank == 0:
        meta = verify_sharded_checkpoint(output_path)
        print(f"[convert] checkpoint verified: {meta}", flush=True)
        print(f"[convert] set in yaml: sharded_checkpoint_dir: {output_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
