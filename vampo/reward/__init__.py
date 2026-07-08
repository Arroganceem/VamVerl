"""Reward · VideoMAE success classifier (WMPO-compatible)."""

from vampo.reward.base import BaseRewardModel, SuccessResult
from vampo.reward.factory import build_reward_model
from vampo.reward.videomae_reward import VideoMAERewardModel

__all__ = [
    "BaseRewardModel",
    "SuccessResult",
    "VideoMAERewardModel",
    "build_reward_model",
]

# Training entry: python -m vampo.reward.train_videomae --config configs/videomae_droid.yaml
