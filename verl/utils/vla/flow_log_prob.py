"""Flow-matching action chain log-prob utilities for PPO."""

from __future__ import annotations

import math

import numpy as np
import torch


def _gp_log_prob_sum(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Sum of diagonal Gaussian log-probs (matches DreamZero wan_flow_matching)."""
    var = float(sigma) ** 2
    diff = target.float() - mean.float()
    return (-0.5 * ((diff * diff) / var + math.log(2 * math.pi * var))).sum()


def _standard_normal_log_prob_sum(sample: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (sample.float() ** 2 + math.log(2 * math.pi))).sum()


def compute_joint_flow_log_prob_from_paths(
    action_path: np.ndarray,
    action_eps: np.ndarray,
    video_path: np.ndarray,
    video_eps: np.ndarray,
    *,
    action_sigma: float,
    video_sigma: float,
) -> float:
    """Exact log π from stored flow path/ε (no DiT re-forward).

    Trace layout per WM step: path[k+1] = μ_k + σ·ε_k.
    """
    ap = torch.as_tensor(np.ascontiguousarray(action_path), dtype=torch.float64)
    ae = torch.as_tensor(np.ascontiguousarray(action_eps), dtype=torch.float64)
    vp = torch.as_tensor(np.ascontiguousarray(video_path), dtype=torch.float64)
    ve = torch.as_tensor(np.ascontiguousarray(video_eps), dtype=torch.float64)

    if ap.ndim < 2 or vp.ndim < 2 or ae.shape[0] != ve.shape[0]:
        raise ValueError(
            f"Invalid trace shapes action_path={ap.shape} action_eps={ae.shape} "
            f"video_path={vp.shape} video_eps={ve.shape}"
        )

    total = torch.zeros((), dtype=torch.float64)
    total = total + _standard_normal_log_prob_sum(ap[0])
    total = total + _standard_normal_log_prob_sum(vp[0])

    for i in range(ae.shape[0]):
        a_target = ap[i + 1]
        a_mean = a_target - float(action_sigma) * ae[i]
        total = total + _gp_log_prob_sum(a_target, a_mean, action_sigma)

        v_target = vp[i + 1]
        v_mean = v_target - float(video_sigma) * ve[i]
        total = total + _gp_log_prob_sum(v_target, v_mean, video_sigma)

    val = float(total.item())
    return val if math.isfinite(val) else float("nan")
