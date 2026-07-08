"""Unified policy runner for rollout and GRPO update."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from vampo.core.types import StepOutput
from vampo.imagination.policy.base import PolicyBackend


class PolicyRunner:
    def __init__(self, backend: PolicyBackend):
        self.backend = backend

    def reset_episode(self) -> None:
        self.backend.reset_episode()

    def infer(self, obs: dict, prompt: str) -> StepOutput:
        return self.backend.infer(obs, prompt)

    @property
    def trainable_params(self) -> list[torch.nn.Parameter]:
        return self.backend.trainable_parameters()

    @contextmanager
    def rollout_mode(self) -> Iterator[None]:
        with self.backend.rollout_mode():
            yield

    @contextmanager
    def train_mode(self) -> Iterator[None]:
        with self.backend.train_mode():
            yield

    def save_checkpoint(self, path: str) -> None:
        self.backend.save(path)
