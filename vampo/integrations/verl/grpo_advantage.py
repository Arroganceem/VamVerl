"""VAMPO GRPO advantages with tie-break when VideoMAE scores are identical within a group."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from vampo.integrations.verl.log_prob_utils import _format_lp_values
from verl import DataProto


def _group_std(values: list[float] | np.ndarray, epsilon: float = 1e-6) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 1.0
    return float(np.std(arr))


def _normalize_group_values(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Per-group (mean, std) normalize with NaN-safe and zero-std fallback."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        arr = np.arange(arr.size, dtype=np.float64)
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    if std <= epsilon:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mean) / (std + epsilon)


def _trajectory_log_prob_scalars(
    rollout_log_prob_scalar: torch.Tensor | None,
    old_log_probs: torch.Tensor | None,
    eos_mask: torch.Tensor | None,
) -> np.ndarray | None:
    """Prefer WM-step trace sums; fall back to masked token mean on old_log_probs."""
    if rollout_log_prob_scalar is not None:
        vals = rollout_log_prob_scalar.detach().float().cpu().numpy().astype(np.float64)
        if np.all(np.isfinite(vals)) and np.any(np.abs(vals) > 0):
            return vals

    if old_log_probs is None or eos_mask is None:
        return None

    mask = eos_mask.float()
    lp_sum = (old_log_probs.float() * mask).sum(dim=-1)
    lp_denom = mask.sum(dim=-1).clamp_min(1.0)
    vals = (lp_sum / lp_denom).detach().float().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(vals)):
        return None
    if not np.any(np.abs(vals) > 0):
        return None
    return vals


def compute_vampo_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    *,
    old_log_probs: torch.Tensor | None = None,
    rollout_log_prob_scalar: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO outcome advantage; tie-break with rollout log-prob when reward std is ~0."""
    response_length = token_level_rewards.shape[-1]
    outcome = token_level_rewards.sum(dim=-1).detach().float().cpu().numpy()
    group_ids = np.asarray([str(x) for x in index])

    id2indices: dict[str, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        id2indices[gid].append(i)

    tiebreak = outcome.copy().astype(np.float64)
    lp_scalar = _trajectory_log_prob_scalars(
        rollout_log_prob_scalar, old_log_probs, eos_mask
    )
    if lp_scalar is not None:
        for gid, idxs in id2indices.items():
            if len(idxs) <= 1:
                continue
            group_rewards = [float(outcome[i]) for i in idxs]
            if _group_std(group_rewards, epsilon) > epsilon:
                continue
            lp_vals = [float(lp_scalar[i]) for i in idxs]
            for i in idxs:
                tiebreak[i] = float(lp_scalar[i])
            print(
                f"VAMPO GRPO tie-break uid={gid[:8]} n={len(idxs)} "
                f"reward={group_rewards[0]:.4f} → log_prob={_format_lp_values(lp_vals)}",
                flush=True,
            )
            if _group_std(lp_vals, epsilon) <= epsilon:
                for j, i in enumerate(idxs):
                    tiebreak[i] = float(j)
                print(
                    f"VAMPO GRPO tie-break uid={gid[:8]} → rank by sample index "
                    f"(log_prob tied or non-finite)",
                    flush=True,
                )
    elif old_log_probs is not None:
        print(
            "VAMPO GRPO tie-break: reward tied but rollout log_prob unavailable/degenerate; "
            "using sample-index rank fallback",
            flush=True,
        )
        for gid, idxs in id2indices.items():
            if len(idxs) <= 1:
                continue
            group_rewards = [float(outcome[i]) for i in idxs]
            if _group_std(group_rewards, epsilon) <= epsilon:
                for j, i in enumerate(idxs):
                    tiebreak[i] = float(j)

    adv_scalar = np.zeros(len(tiebreak), dtype=np.float64)
    for gid, idxs in id2indices.items():
        group_vals = tiebreak[idxs]
        adv_scalar[idxs] = _normalize_group_values(group_vals, epsilon)

    adv_scalar_t = torch.tensor(adv_scalar, device=token_level_rewards.device, dtype=torch.float32)
    advantages = adv_scalar_t.unsqueeze(-1).expand(-1, response_length) * eos_mask.float()
    if lp_scalar is not None:
        print(
            f"VAMPO GRPO rollout_log_prob_scalar={_format_lp_values(lp_scalar.tolist())} "
            f"adv={_format_lp_values(adv_scalar.tolist())} mean_adv={float(adv_scalar.mean()):.4f}",
            flush=True,
        )
    return advantages, advantages.clone()


def compute_vampo_grpo_from_batch(data: DataProto, config) -> tuple[torch.Tensor, torch.Tensor]:
    """Driver-side GRPO for VAMPO batches (matches ``compute_advantage`` mask layout)."""
    token_level_rewards = data.batch["token_level_rewards"]
    responses = data.batch["responses"]
    response_length = responses.size(1) * responses.size(2)
    finish_step = data.batch["finish_step"] * config.actor_rollout_ref.model.action_token_len
    steps = torch.arange(response_length, device=responses.device)
    steps_expanded = steps.unsqueeze(0).expand(responses.size(0), -1)
    response_mask = steps_expanded < finish_step.unsqueeze(1)
    return compute_vampo_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=response_mask,
        index=data.non_tensor_batch["uid"],
        old_log_probs=data.batch.get("old_log_probs"),
        rollout_log_prob_scalar=data.batch.get("rollout_log_prob_scalar"),
    )
