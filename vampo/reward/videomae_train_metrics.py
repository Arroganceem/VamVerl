"""Training metrics helpers (aligned with UniVTM univtm/training/trainer_base.py)."""

from __future__ import annotations

import torch
import torch.distributed as dist


class MetricEMA:
    """Exponential moving average for noisy per-step metrics."""

    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self._state: dict[str, float] = {}

    def update(self, metrics: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, val in metrics.items():
            prev = self._state.get(key, val)
            smoothed = self.beta * prev + (1.0 - self.beta) * val
            self._state[key] = smoothed
            out[f"{key}_ema"] = smoothed
        return out


def ddp_mean_scalar(val: float, device: torch.device, world_size: int) -> float:
    t = torch.tensor([val], device=device, dtype=torch.float64)
    if dist.is_initialized() and world_size > 1:
        if hasattr(dist.ReduceOp, "AVG"):
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
    return float(t.item())


def ddp_mean_tensor(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if dist.is_initialized() and world_size > 1:
        out = t.detach().clone()
        if hasattr(dist.ReduceOp, "AVG"):
            dist.all_reduce(out, op=dist.ReduceOp.AVG)
        else:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            out /= world_size
        return out
    return t.detach()


def format_step_metrics(row: dict[str, float], *, step_sec: float) -> str:
    """Primary display uses smoothed loss (UniVTM trainer_base pattern)."""
    display_loss = row.get("loss_ema", row["loss"])
    parts = [f"loss={display_loss:.4f}", f"sec={step_sec:.1f}s"]
    if "loss_raw" in row:
        parts.append(f"raw={row['loss_raw']:.4f}")
    if "batch_pos" in row:
        parts.append(f"pos={row['batch_pos']:.2f}")
    if "loss_ema" in row:
        parts.append(f"ema={row['loss_ema']:.4f}")
    return " ".join(parts)


class EpochMetricAccumulator:
    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count = 0

    def add(self, row: dict[str, float]) -> None:
        for key, val in row.items():
            if key.endswith("_ema"):
                continue
            self._sum[key] = self._sum.get(key, 0.0) + float(val)
        self._count += 1

    def mean(self) -> dict[str, float]:
        if self._count <= 0:
            return {}
        return {k: v / self._count for k, v in self._sum.items()}

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._sum.clear()
        self._count = 0
