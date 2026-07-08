"""Reward · VideoMAE success classifier (WMPO-compatible)."""

from verl.utils.reward.base import BaseRewardModel, SuccessResult
from verl.utils.reward.factory import build_reward_model
from verl.utils.reward.videomae_reward import VideoMAERewardModel

__all__ = [
    "BaseRewardModel",
    "SuccessResult",
    "VideoMAERewardModel",
    "build_reward_model",
]

# Training entry: python -m verl.utils.reward.train_videomae --config verl/utils/reward/configs/videomae_droid.yaml
