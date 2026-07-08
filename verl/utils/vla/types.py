"""Shared datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepOutput:
    """One policy step: action chunk + imagined pixel frames for reward evaluation."""

    action: np.ndarray  # (horizon, action_dim)
    video_frames: np.ndarray  # (T, H, W, C) uint8
    info: dict = field(default_factory=dict)
    flow_path: np.ndarray | None = None  # [K+1, H, D] normalized action latents
    flow_eps: np.ndarray | None = None  # [K, H, D]
    video_flow_path: np.ndarray | None = None
    video_flow_eps: np.ndarray | None = None
