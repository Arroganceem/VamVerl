"""CPU-only unit tests for RL reward / GRPO pipeline (no VLA / no training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vampo.integrations.verl.grpo_advantage import compute_vampo_grpo_outcome_advantage
from vampo.integrations.verl.log_prob_utils import (
    build_rollout_log_prob_tensors,
    log_probs_degenerate,
)
from vampo.integrations.verl.staged_checks import check_rl_pipeline_mock
from vampo.reward.videomae_reward import load_videomae_threshold
from vampo.reward.wmpo_sliding import (
    build_sliding_clips,
    frame_finish_to_wm_step,
    scan_clips_for_success,
)


def test_build_rollout_log_prob_tensors_shape():
    from vampo.rl.trajectory import ChunkRecord, Trajectory

    traj = Trajectory(
        init_state_id="s0",
        prompt="p",
        uid="u0",
        chunks=[
            ChunkRecord(
                obs={"x": np.zeros(4)},
                action=np.zeros((2, 3)),
                video_frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
                flow_log_prob=-1.5,
            ),
            ChunkRecord(
                obs={"x": np.zeros(4)},
                action=np.zeros((2, 3)),
                video_frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
                flow_log_prob=-2.0,
            ),
        ],
    )
    built = build_rollout_log_prob_tensors([traj], max_wm=2, action_flat=6)
    assert built is not None
    per_step, old_log_probs, scalar = built
    assert per_step.shape == (1, 2)
    assert old_log_probs.shape == (1, 12)
    assert scalar.item() == pytest.approx(-3.5)
    assert not log_probs_degenerate(old_log_probs, scalar)


def test_grpo_tie_break_by_log_prob():
    batch = 4
    response_len = 8
    token_level_rewards = torch.zeros(batch, response_len)
    token_level_rewards[2, -1] = 1.0
    eos_mask = torch.ones(batch, response_len)
    index = np.array(["g0", "g0", "g1", "g1"], dtype=object)
    rollout_log_prob_scalar = torch.tensor([-1.0, -3.0, -0.5, -0.5])

    adv, _ = compute_vampo_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=eos_mask,
        index=index,
        rollout_log_prob_scalar=rollout_log_prob_scalar,
    )
    adv_sum = adv.sum(dim=-1).numpy()
    assert np.all(np.isfinite(adv_sum))
    assert adv_sum[0] > adv_sum[1]
    assert adv_sum[2] > adv_sum[3]


def test_wmpo_sliding_scan():
    video = np.zeros((40, 16, 16, 3), dtype=np.uint8)
    clips = build_sliding_clips(video, window_size=8, stride=1, min_steps=32)

    def probs_fn(_clip_imgs):
        rows = []
        for _ in _clip_imgs:
            rows.append([0.1, 0.9])
        return np.asarray(rows, dtype=np.float64)

    early = scan_clips_for_success(clips, probs_fn=probs_fn, threshold=0.5, batch_size=4)
    assert early.complete is True
    assert early.finish_step < 39


def test_frame_finish_to_wm_step():
    assert frame_finish_to_wm_step(8, frames_per_chunk=4, max_wm_steps=8) == 2
    assert frame_finish_to_wm_step(100, frames_per_chunk=4, max_wm_steps=4) == 4


def test_load_videomae_threshold_sidecar(tmp_path: Path):
    ckpt = tmp_path / "videomae_droid.pth"
    ckpt.write_bytes(b"fake")
    sidecar = ckpt.with_suffix(".json")
    sidecar.write_text(json.dumps({"threshold": 0.374}))
    assert load_videomae_threshold(str(ckpt)) == pytest.approx(0.374)


def test_rl_pipeline_mock_check():
    result = check_rl_pipeline_mock()
    assert result.status == "ok", result.message
