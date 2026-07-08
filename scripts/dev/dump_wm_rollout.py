#!/usr/bin/env python3
"""Run one in-process WM rollout on a single GPU and dump imagined video for reward diagnosis.

Example (on cluster GPU node):
  python scripts/dev/dump_wm_rollout.py --index 0
  python scripts/dev/dump_wm_rollout.py --index 0 --diagnose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_init(init_dir: Path, index: int) -> tuple[str, dict, str]:
    from verl.workers.rollout.imagination.rollout import InitStateStore

    store = InitStateStore(init_dir)
    return store.get(index)


def _init_single_gpu_dist() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump real WM imagined video from one rollout")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/home/robotem/Models/DreamZero-DROID"))
    parser.add_argument("--tokenizer-path", default=os.environ.get("TOKENIZER_PATH", "/home/robotem/Models/umt5-xxl"))
    parser.add_argument("--init-states", type=Path, default=ROOT / "data/init_states")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-wm-steps", type=int, default=8)
    parser.add_argument(
        "--primary-camera-key",
        default="observation.images.exterior_image_1_left",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "debug_reward/wm_rollout",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available; run on a GPU node.", file=sys.stderr)
        return 1

    os.environ.setdefault("VAMPO_DEBUG_REWARD", "1")
    os.environ["VAMPO_DEBUG_REWARD_DIR"] = str(args.out_dir)

    _init_single_gpu_dist()

    state_id, init_obs, prompt = _load_init(args.init_states, args.index)
    print(f"Init: {state_id}")
    print(f"Prompt: {prompt!r}")
    print(f"Device: {args.device}  max_wm_steps={args.max_wm_steps}")

    from verl.workers.rollout.imagination.policy.runner import PolicyRunner
    from verl.workers.rollout.imagination.rollout import ImaginationRollout
    from verl.workers.rollout.dreamzero_backend import DreamZeroInProcessBackend
    from verl.utils.vla.dreamzero_policy import DreamZeroPolicyModule

    policy = DreamZeroPolicyModule(
        model_path=args.model_path,
        device=args.device,
        action_horizon=8,
        action_dim=8,
        imagined_frames=8,
        rl_fine_tune_mode="full",
        tune_projector=True,
        tune_diffusion_model=False,
        primary_camera_key=args.primary_camera_key,
        tokenizer_path_override=args.tokenizer_path,
    )
    policy.eval()
    if hasattr(policy.groot, "post_initialize"):
        policy.groot.post_initialize()

    backend = DreamZeroInProcessBackend(policy, prompt=prompt)
    runner = PolicyRunner(backend)
    rollout = ImaginationRollout(
        policy=runner,
        reward_model=None,
        max_wm_steps=args.max_wm_steps,
        primary_camera_key=args.primary_camera_key,
    )

    print("Running WM rollout (this may take several minutes)...")
    with torch.inference_mode():
        traj = rollout.rollout_one(init_obs, prompt, state_id, uid=str(uuid.uuid4()))

    video = np.asarray(traj.video)
    print(f"Done. traj.video shape={video.shape}  wm_chunks={len(traj.chunks)}")

    latest = args.out_dir / "LATEST"
    dump_dir = Path(latest.read_text().strip()) if latest.is_file() else None
    if dump_dir is None or not dump_dir.is_dir():
        # fallback: newest subdir
        subs = sorted(args.out_dir.glob(f"{state_id}_*"), key=lambda p: p.stat().st_mtime)
        dump_dir = subs[-1] if subs else None
    if dump_dir is None:
        print("ERROR: dump directory not found", file=sys.stderr)
        return 1

    meta = json.loads((dump_dir / "meta.json").read_text())
    print(f"Dump → {dump_dir}")
    print(
        f"  frame_diff_full={meta['frame_diff_full']:.3f}  "
        f"last8={meta['frame_diff_last8']:.3f}  "
        f"first_vs_last={meta['frame_diff_first_vs_last']:.3f}"
    )
    print(f"  PNGs: compare_first.png compare_last.png last8_f*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
