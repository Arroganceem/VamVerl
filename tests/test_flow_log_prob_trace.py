"""Tests for trace path/ε log prob reconstruction (FSDP-safe fallback)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vampo.integrations.verl.flow_log_prob import compute_joint_flow_log_prob_from_paths
from vampo.integrations.verl.log_prob_utils import log_prob_scalar_from_chunk
from vampo.rl.trajectory import ChunkRecord


def _synthetic_trace(*, steps: int = 3, action_sigma: float = 0.05, video_sigma: float = 0.05):
    rng = np.random.default_rng(0)
    action_path = [rng.normal(size=(4, 7)).astype(np.float32)]
    action_eps = []
    video_path = [rng.normal(size=(2, 8, 16, 16)).astype(np.float32)]
    video_eps = []
    for _ in range(steps):
        ae = rng.normal(size=(4, 7)).astype(np.float32)
        ve = rng.normal(size=(2, 8, 16, 16)).astype(np.float32)
        action_eps.append(ae)
        video_eps.append(ve)
        action_path.append(action_path[-1] + action_sigma * ae)
        video_path.append(video_path[-1] + video_sigma * ve)
    return (
        np.stack(action_path, axis=0),
        np.stack(action_eps, axis=0),
        np.stack(video_path, axis=0),
        np.stack(video_eps, axis=0),
    )


def test_compute_joint_flow_log_prob_from_paths_finite():
    ap, ae, vp, ve = _synthetic_trace()
    val = compute_joint_flow_log_prob_from_paths(
        ap, ae, vp, ve, action_sigma=0.05, video_sigma=0.05
    )
    assert math.isfinite(val)
    assert abs(val) > 1e-6


def test_log_prob_scalar_from_chunk_recovers_when_scalar_missing():
    ap, ae, vp, ve = _synthetic_trace(steps=2)
    chunk = ChunkRecord(
        obs={"x": np.zeros(3)},
        action=np.zeros((4, 7)),
        video_frames=np.zeros((4, 8, 8, 3), dtype=np.uint8),
        flow_path=ap,
        flow_eps=ae,
        video_flow_path=vp,
        video_flow_eps=ve,
        flow_log_prob=None,
    )
    val = log_prob_scalar_from_chunk(chunk, action_sigma=0.05, video_sigma=0.05)
    assert val is not None
    assert math.isfinite(val)
    assert abs(val) > 1e-6


def test_trace_log_prob_self_consistent():
    """path[k+1]=μ+σ·ε ⇒ reconstructed μ gives zero diff in GP term."""
    sigma = 0.05
    eps = np.array([[1.0], [-2.0]], dtype=np.float32)
    path0 = np.zeros((1, 1), dtype=np.float32)
    path1 = path0 + sigma * eps[0:1]
    path2 = path1 + sigma * eps[1:2]
    ap = np.stack([path0, path1, path2], axis=0)
    ae = eps
    vp = ap.copy()
    ve = ae.copy()
    val = compute_joint_flow_log_prob_from_paths(
        ap, ae, vp, ve, action_sigma=sigma, video_sigma=sigma
    )
    assert math.isfinite(val)


def test_trace_log_prob_nan_paths_returns_nan():
    sigma = 0.05
    eps = np.array([[1.0], [-2.0]], dtype=np.float32)
    path0 = np.zeros((1, 1), dtype=np.float32)
    path1 = path0 + sigma * eps[0:1]
    path2 = path1 + sigma * eps[1:2]
    ap = np.stack([path0, path1, path2], axis=0)
    ap_nan = ap.copy()
    ap_nan[1, 0, 0] = np.nan
    ae = eps
    vp = ap.copy()
    ve = ae.copy()
    val = compute_joint_flow_log_prob_from_paths(
        ap_nan, ae, vp, ve, action_sigma=sigma, video_sigma=sigma
    )
    assert math.isnan(val)
