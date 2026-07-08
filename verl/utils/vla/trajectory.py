"""Trajectory types for GRPO rollout."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChunkRecord:
    obs: dict
    action: np.ndarray
    video_frames: np.ndarray
    flow_path: np.ndarray | None = None
    flow_eps: np.ndarray | None = None
    video_flow_path: np.ndarray | None = None
    video_flow_eps: np.ndarray | None = None
    flow_log_prob: float | None = None
    flow_cond: dict | None = None


@dataclass
class Trajectory:
    init_state_id: str
    prompt: str
    uid: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    complete: bool = False
    finish_step: int = 0  # WM step count for verl mask (converted from frame index at rollout)

    @property
    def video(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((1, 64, 64, 3), dtype=np.uint8)
        return np.concatenate([c.video_frames for c in self.chunks], axis=0)
