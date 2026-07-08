"""Shared reward model types (WMPO: complete + finish_step only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SuccessResult:
    complete: bool
    finish_step: int  # frame index (earliest success window end - 1)


class BaseRewardModel(Protocol):
    def predict_success(
        self,
        video: np.ndarray,
        prompt: str = "",
        batch_size: int = 32,
    ) -> SuccessResult: ...
