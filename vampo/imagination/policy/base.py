"""Policy backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

import torch

from vampo.core.types import StepOutput


class PolicyBackend(ABC):
    """Pluggable policy backend (in-process VLA for verl training)."""

    @abstractmethod
    def reset_episode(self) -> None: ...

    @abstractmethod
    def infer(self, obs: dict, prompt: str) -> StepOutput: ...

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def is_trainable(self) -> bool:
        return len(self.trainable_parameters()) > 0

    @contextmanager
    def rollout_mode(self) -> Iterator[None]:
        yield

    @contextmanager
    def train_mode(self) -> Iterator[None]:
        yield

    def state_dict(self) -> dict:
        return {}

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
