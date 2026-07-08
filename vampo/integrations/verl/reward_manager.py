"""WMPO RobRewardManager — sparse binary reward at finish_step × action_token_len."""

from __future__ import annotations

import torch

from verl import DataProto


class VAMPORewardManager:
    """Maps rollout ``complete`` (0/1) to sparse token reward (WMPO-compatible)."""

    def __init__(self, num_examine: int, config) -> None:
        self.num_examine = num_examine
        self.config = config

    def verify(self, data: DataProto):
        completes = data.batch["complete"].tolist()
        batch_size = data.batch["responses"].size(0)
        assert len(completes) == batch_size
        score = [float(item) for item in completes]
        format_ok = [1.0 for _ in completes]

        device = data.batch["responses"].device
        data.batch["acc"] = torch.tensor(score, dtype=torch.float32, device=device)
        data.batch["format_correctness"] = torch.tensor(format_ok, dtype=torch.float32, device=device)

        reward_metrics = {"all": data.batch["acc"].mean().item()}
        format_metrics = {"all": data.batch["format_correctness"].mean().item()}
        reward_format_metrics = {"all": data.batch["acc"].mean().item()}
        return score, reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        reward_tensor_dict: dict = {}
        reward_metrics: dict = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        verifier_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))

        action_token_len = self.config.actor_rollout_ref.model.action_token_len
        valid_response_length = data.batch["finish_step"] * action_token_len

        if "acc" in data.batch:
            verifier_score = data.batch["acc"].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, _, _ = self.verify(data)
            reward_metrics.update(verifier_metrics)

        for i in range(verifier_reward.shape[0]):
            idx = int(valid_response_length[i].item()) - 1
            if idx >= 0:
                verifier_reward[i, idx] += verifier_score[i]

        reward_tensor_dict["gt_scores"] = verifier_reward

        coef = float(self.config.verifier.reward_coef)
        if coef != 0:
            reward_metrics["verifier"] = reward_tensor_dict["gt_scores"].sum(dim=1).mean().item()
            reward_tensor = reward_tensor + coef * reward_tensor_dict["gt_scores"]

        reward_tensor_dict["all"] = reward_tensor
        reward_metrics["reward_all"] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        return reward_tensor_dict, reward_metrics
