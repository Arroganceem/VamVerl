"""In-process PolicyBackend backed by DreamZeroPolicyModule."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from verl.utils.vla.types import StepOutput
from verl.workers.rollout.imagination.policy.base import PolicyBackend
from verl.utils.vla.dreamzero_policy import DreamZeroPolicyModule


class DreamZeroInProcessBackend(PolicyBackend):
    def __init__(self, module: DreamZeroPolicyModule, prompt: str = ""):
        self.module = module
        self.default_prompt = prompt

    def reset_episode(self) -> None:
        self.module.reset_episode()

    def infer(self, obs: dict, prompt: str) -> StepOutput:
        prompt = prompt or self.default_prompt
        action, video, flow_lp, trace = self.module.sample_step(obs, prompt)
        ap, ae, vp, ve = trace.to_numpy()
        lp_val = float(flow_lp.detach().float().cpu().item()) if flow_lp.numel() == 1 else None
        return StepOutput(
            action=action,
            video_frames=video,
            flow_path=ap,
            flow_eps=ae,
            video_flow_path=vp,
            video_flow_eps=ve,
            flow_cond=trace.flow_cond,
            info={"backend": "vla_inprocess", "flow_log_prob": lp_val},
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self.module.trainable_parameters_list()

    @contextmanager
    def rollout_mode(self) -> Iterator[None]:
        self.module.eval()
        for p in self.module.parameters():
            if not p.requires_grad:
                p.requires_grad_(False)
        yield

    @contextmanager
    def train_mode(self) -> Iterator[None]:
        self.module.train()
        yield
        self.module.eval()

    def save(self, path: str) -> None:
        self.module.save_rl_checkpoint(path)
