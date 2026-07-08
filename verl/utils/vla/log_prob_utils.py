"""Rollout flow log-prob validation and tensor layout helpers."""

from __future__ import annotations

import math
import os

import torch

from verl.utils.vla.flow_log_prob import compute_joint_flow_log_prob_from_paths
from verl.utils.vla.trajectory import ChunkRecord, Trajectory


def log_prob_scalar_from_chunk(
    chunk: ChunkRecord,
    *,
    action_sigma: float,
    video_sigma: float,
) -> float | None:
    """Reconstruct WM-step log π from stored flow path/ε when scalar was not kept."""
    if (
        chunk.flow_path is None
        or chunk.flow_eps is None
        or chunk.video_flow_path is None
        or chunk.video_flow_eps is None
    ):
        return None
    try:
        return compute_joint_flow_log_prob_from_paths(
            chunk.flow_path,
            chunk.flow_eps,
            chunk.video_flow_path,
            chunk.video_flow_eps,
            action_sigma=action_sigma,
            video_sigma=video_sigma,
        )
    except Exception:
        return None


def debug_log_prob_enabled() -> bool:
    return os.environ.get("VAMPO_DEBUG_LOG_PROB", "0").lower() in {"1", "true", "yes"}


def _format_lp_values(values: list[float]) -> list[str]:
    out: list[str] = []
    for v in values:
        if not math.isfinite(v):
            out.append("nan")
        elif abs(v) >= 1000 or (0 < abs(v) < 1e-3):
            out.append(f"{v:.4e}")
        else:
            out.append(f"{v:.4f}")
    return out


def build_rollout_log_prob_tensors(
    trajectories: list[Trajectory],
    max_wm: int,
    action_flat: int,
    *,
    device: torch.device | str = "cpu",
    action_sigma: float | None = None,
    video_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build per-WM log probs, token-expanded old_log_probs, and trajectory scalars."""
    if not trajectories or max_wm <= 0 or action_flat <= 0:
        return None

    batch_size = len(trajectories)
    per_step = torch.zeros(batch_size, max_wm, dtype=torch.float32)
    missing: list[str] = []
    recovered: list[str] = []

    for i, traj in enumerate(trajectories):
        for t in range(max_wm):
            if t >= len(traj.chunks):
                missing.append(f"traj{i}:short_chunks")
                break
            chunk = traj.chunks[t]
            lp = chunk.flow_log_prob
            if lp is None and action_sigma is not None and video_sigma is not None:
                lp = log_prob_scalar_from_chunk(
                    chunk,
                    action_sigma=float(action_sigma),
                    video_sigma=float(video_sigma),
                )
                if lp is not None:
                    recovered.append(f"traj{i}:step{t}")
            if lp is None:
                missing.append(f"traj{i}:step{t}:none")
                break
            if not math.isfinite(float(lp)):
                missing.append(f"traj{i}:step{t}:nonfinite")
                break
            per_step[i, t] = float(lp)
        if missing:
            break

    if missing:
        print(
            "VAMPO rollout log_prob missing/invalid: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else ""),
            flush=True,
        )
        return None
    if recovered:
        print(
            "VAMPO rollout log_prob recovered from flow path/ε: "
            + ", ".join(recovered[:8])
            + (" ..." if len(recovered) > 8 else ""),
            flush=True,
        )

    device = torch.device(device)
    scalar = per_step.sum(dim=-1)
    old_log_probs = (
        per_step.unsqueeze(-1)
        .expand(-1, -1, action_flat)
        .reshape(batch_size, max_wm * action_flat)
        .to(device)
    )
    return (
        per_step.to(device),
        old_log_probs,
        scalar.to(device),
    )


def log_probs_degenerate(
    old_log_probs: torch.Tensor | None,
    rollout_log_prob_scalar: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> bool:
    if rollout_log_prob_scalar is not None:
        vals = rollout_log_prob_scalar.detach().float().cpu()
        if vals.numel() == 0:
            return True
        if not torch.isfinite(vals).all():
            return True
        if float(vals.abs().max()) <= eps:
            return True
        return False

    if old_log_probs is None:
        return True
    vals = old_log_probs.detach().float().cpu()
    if vals.numel() == 0 or not torch.isfinite(vals).all():
        return True
    return float(vals.abs().max()) <= eps


def apply_recomputed_log_prob_fields(output) -> None:
    """Populate rollout_log_probs / scalar from token-expanded old_log_probs (verl layout)."""
    old_log_probs = output.batch.get("old_log_probs")
    responses = output.batch.get("responses")
    if old_log_probs is None or responses is None:
        return
    traj_len, action_flat = int(responses.shape[1]), int(responses.shape[2])
    per_step, scalar = rebuild_log_prob_fields_from_old(old_log_probs, traj_len, action_flat)
    output.batch["rollout_log_probs"] = per_step.to(old_log_probs.device)
    output.batch["rollout_log_prob_scalar"] = scalar.to(old_log_probs.device)


def rebuild_log_prob_fields_from_old(
    old_log_probs: torch.Tensor,
    traj_len: int,
    action_flat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover per-WM log probs and trajectory scalars from token-expanded old_log_probs."""
    flat = old_log_probs.reshape(old_log_probs.size(0), traj_len, action_flat)
    per_step = flat[:, :, 0].contiguous()
    scalar = per_step.sum(dim=-1)
    return per_step, scalar


def log_rollout_log_prob_summary(
    per_step: torch.Tensor | None,
    scalar: torch.Tensor | None,
    *,
    phase: str = "rollout",
) -> None:
    if scalar is None:
        print(f"VAMPO [{phase}] rollout_log_prob: missing", flush=True)
        return
    scalars = scalar.detach().float().cpu().tolist()
    per_traj = []
    if per_step is not None:
        for row in per_step.detach().float().cpu().tolist():
            per_traj.append(_format_lp_values([float(x) for x in row]))
    print(
        f"VAMPO [{phase}] rollout_log_prob scalar={_format_lp_values(scalars)}"
        + (f" per_wm={per_traj}" if per_traj else ""),
        flush=True,
    )
