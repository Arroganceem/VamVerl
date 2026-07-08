"""Convert VAMPO trajectories to verl DataProto batches."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from verl.workers.rollout.imagination.policy.obs_utils import flatten_obs
from verl.utils.vla.trajectory import Trajectory
from verl import DataProto


def trajectories_to_dataproto(
    trajectories: list[Trajectory],
    action_horizon: int,
    action_dim: int,
    feat_dim: int = 512,
    device: torch.device | str = "cpu",
) -> DataProto:
    """Build verl-compatible TensorDict from imagined trajectories (no old_log_probs; HybridEngine recomputes)."""
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    device = torch.device(device)
    batch_size = len(trajectories)
    max_wm = max(len(t.chunks) for t in trajectories)
    max_wm = max(max_wm, 1)
    action_flat = action_horizon * action_dim

    obs_features = torch.zeros(batch_size, max_wm, feat_dim, dtype=torch.float32)
    responses = torch.zeros(batch_size, max_wm, action_flat, dtype=torch.float32)
    input_ids = torch.zeros(batch_size, max_wm, 1, dtype=torch.long)
    attention_mask = torch.ones(batch_size, max_wm, 1, dtype=torch.long)

    complete = torch.zeros(batch_size, dtype=torch.bool)
    finish_step = torch.zeros(batch_size, dtype=torch.int64)
    state_ids = []

    for i, traj in enumerate(trajectories):
        state_ids.append(traj.init_state_id)
        complete[i] = bool(traj.complete)
        finish_step[i] = int(traj.finish_step)

        for step, chunk in enumerate(traj.chunks):
            feat = flatten_obs(chunk.obs)
            if feat.numel() > feat_dim:
                feat = feat[:feat_dim]
            elif feat.numel() < feat_dim:
                feat = torch.cat([feat, torch.zeros(feat_dim - feat.numel())])
            obs_features[i, step] = feat

            act = torch.as_tensor(chunk.action, dtype=torch.float32).reshape(-1)
            if act.numel() > action_flat:
                act = act[:action_flat]
            elif act.numel() < action_flat:
                act = torch.cat([act, torch.zeros(action_flat - act.numel())])
            responses[i, step] = act

    batch = TensorDict(
        {
            "responses": responses,
            "obs_features": obs_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.zeros(batch_size, max_wm, 1, dtype=torch.float32),
            "complete": complete,
            "finish_step": finish_step,
            "state_id": torch.arange(batch_size, dtype=torch.int64),
        },
        batch_size=batch_size,
    )
    batch = batch.to(device)

    proto = DataProto(
        batch=batch,
        non_tensor_batch={
            "state_id_str": np.array(state_ids, dtype=object),
            "obs_chunks": np.array(
                [[dict(chunk.obs) for chunk in traj.chunks] for traj in trajectories],
                dtype=object,
            ),
            "prompts": np.array([traj.prompt for traj in trajectories], dtype=object),
            "flow_traces": np.array(
                [
                    [
                        {
                            "action_path": c.flow_path,
                            "action_eps": c.flow_eps,
                            "video_path": c.video_flow_path,
                            "video_eps": c.video_flow_eps,
                            # legacy keys
                            "path": c.flow_path,
                            "eps": c.flow_eps,
                        }
                        for c in traj.chunks
                    ]
                    for traj in trajectories
                ],
                dtype=object,
            ),
        },
    )
    return proto
