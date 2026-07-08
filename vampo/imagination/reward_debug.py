"""Debug dumps for VLM reward inputs (imagined WM video)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vampo.rl.trajectory import Trajectory


def _frame_diff_mean(video: np.ndarray) -> float:
    if video.shape[0] < 2:
        return 0.0
    diffs = []
    for i in range(1, video.shape[0]):
        a = video[i - 1].astype(np.float32)
        b = video[i].astype(np.float32)
        diffs.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(diffs)) if diffs else 0.0


def debug_reward_enabled() -> bool:
    return os.environ.get("VAMPO_DEBUG_REWARD", "0").lower() in {"1", "true", "yes"}


def debug_reward_dir() -> Path:
    root = os.environ.get("VAMPO_DEBUG_REWARD_DIR", "./debug_reward/wm_rollout")
    return Path(root)


def dump_trajectory_video(
    traj: Trajectory,
    *,
    state_id: str,
    out_root: Path | None = None,
    last_n: int = 8,
) -> Path:
    """Save full imagined video + last *last_n* frames for VLM diagnosis."""
    video = np.asarray(traj.video)
    if video.ndim != 4:
        raise ValueError(f"Expected traj.video (T,H,W,C), got {video.shape}")

    out_root = out_root or debug_reward_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{state_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_n = min(last_n, int(video.shape[0]))
    last8 = video[-last_n:]

    np.save(out_dir / "video_full.npy", video)
    np.save(out_dir / "video_last8.npy", last8)

    for i, frame in enumerate(last8):
        arr = np.clip(frame, 0, 255).astype(np.uint8)[:, :, :3]
        Image.fromarray(arr).save(out_dir / f"last8_f{i:03d}.png")

    # First vs last for semantic change inspection
    if video.shape[0] >= 2:
        Image.fromarray(np.clip(video[0], 0, 255).astype(np.uint8)[:, :, :3]).save(
            out_dir / "compare_first.png"
        )
        Image.fromarray(np.clip(video[-1], 0, 255).astype(np.uint8)[:, :, :3]).save(
            out_dir / "compare_last.png"
        )

    meta = {
        "state_id": state_id,
        "prompt": traj.prompt,
        "uid": traj.uid,
        "complete": bool(traj.complete),
        "finish_step": int(traj.finish_step),
        "wm_steps": len(traj.chunks),
        "video_shape": list(video.shape),
        "last8_shape": list(last8.shape),
        "frame_diff_full": _frame_diff_mean(video),
        "frame_diff_last8": _frame_diff_mean(last8),
        "frame_diff_first_vs_last": float(
            np.mean(np.abs(video[0].astype(np.float32) - video[-1].astype(np.float32)))
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_root / "LATEST").write_text(str(out_dir.resolve()))
    return out_dir
