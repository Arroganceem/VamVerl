"""verl StateDataset-compatible wrapper around InitStateStore."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vampo.imagination.rollout import InitStateStore


class VAMPOInitStateDataset(Dataset):
    """Expose init states for verl rollout (state_id + init_index per sample)."""

    def __init__(self, init_states_dir: str | Path, pad_to: int = 0):
        self.store = InitStateStore(init_states_dir)
        if len(self.store) == 0:
            raise RuntimeError(
                f"No init states under {init_states_dir}; "
                "run: python -m vampo.data.init_states_bootstrap"
            )
        self.pad_to = pad_to
        self._indices = list(range(len(self.store)))

        if self.pad_to > len(self._indices):
            extra = self.pad_to - len(self._indices)
            rng = random.Random(1)
            self._indices.extend(rng.choices(self._indices, k=extra))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        init_index = self._indices[idx]
        state_id, _, _ = self.store.get(init_index)
        return {
            "state_id": torch.tensor(init_index, dtype=torch.int64),
            "init_index": torch.tensor(init_index, dtype=torch.int64),
            "state_id_str": state_id,
        }
