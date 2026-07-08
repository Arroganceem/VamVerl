"""In-process PolicyBackend backed by VLAPolicyModule."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from vampo.core.types import StepOutput
from vampo.imagination.policy.base import PolicyBackend
from vampo.integrations.verl.vla_policy import VLAPolicyModule


class InProcessVLABackend(PolicyBackend):
    def __init__(self, module: VLAPolicyModule, prompt: str = ""):
        self.module = module
        self.default_prompt = prompt

    def reset_episode(self) -> None:
        self.module.reset_episode()

    def infer(self, obs: dict, prompt: str) -> StepOutput:
        prompt = prompt or self.default_prompt
        action, video, _log_prob, trace = self.module.sample_step(obs, prompt)
        ap, ae, vp, ve = trace.to_numpy()
        return StepOutput(
            action=action,
            video_frames=video,
            flow_path=ap,
            flow_eps=ae,
            video_flow_path=vp,
            video_flow_eps=ve,
            info={"backend": "vla_inprocess"},
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
