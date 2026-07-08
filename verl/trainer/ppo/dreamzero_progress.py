"""Driver/worker progress logs for VAMPO verl training (stdout, Ray-visible)."""

from __future__ import annotations

from collections import defaultdict
import math

import torch

from verl import DataProto


def _batch_size(data: DataProto) -> int:
    return int(data.batch.batch_size[0]) if hasattr(data.batch, "batch_size") else len(data.batch["responses"])


def _state_labels(data: DataProto, batch_size: int) -> list[str]:
    raw = data.non_tensor_batch.get("state_id_str")
    if raw is not None and len(raw) == batch_size:
        return [str(x) for x in raw]
    if "state_id" in data.batch:
        return [str(int(x)) for x in data.batch["state_id"].tolist()]
    return [str(i) for i in range(batch_size)]


def _uid_labels(data: DataProto, batch_size: int) -> list[str]:
    raw = data.non_tensor_batch.get("uid")
    if raw is not None and len(raw) == batch_size:
        return [str(x) for x in raw]
    return [str(i) for i in range(batch_size)]


def log_batch_rewards(
    data: DataProto,
    *,
    global_step: int | None = None,
    n_samples: int = 1,
    phase: str = "reward",
) -> None:
    """Print per-trajectory complete flags and batch aggregates (WMPO-style)."""
    batch = data.batch
    if "acc" in batch:
        scores = batch["acc"].detach().float().cpu().tolist()
    elif "complete" in batch:
        scores = [float(x) for x in batch["complete"].detach().cpu().tolist()]
    else:
        return

    batch_size = len(scores)
    completes = (
        batch["complete"].detach().cpu().tolist()
        if "complete" in batch
        else [None] * batch_size
    )
    finish = (
        batch["finish_step"].detach().cpu().tolist()
        if "finish_step" in batch
        else [None] * batch_size
    )
    state_ids = _state_labels(data, batch_size)
    uids = _uid_labels(data, batch_size)

    header = f"VAMPO [{phase}]"
    if global_step is not None:
        header += f" global_step={global_step}"
    print(header, flush=True)

    for i in range(batch_size):
        grp = ""
        if n_samples > 1:
            grp = f" grp={i // n_samples} sample={i % n_samples}"
        complete_s = completes[i] if completes[i] is not None else "?"
        finish_s = finish[i] if finish[i] is not None else "?"
        sid = state_ids[i]
        print(
            f"  [{i}] state={sid}{grp} "
            f"complete={complete_s} finish_wm={finish_s}",
            flush=True,
        )

    n_complete = sum(1 for c in completes if c)
    print(
        f"  >> success_rate={n_complete}/{batch_size} "
        f"n_samples={n_samples} batch={batch_size}",
        flush=True,
    )


def log_grpo_summary(
    data: DataProto,
    *,
    global_step: int | None = None,
    n_samples: int = 1,
    action_token_len: int = 64,
) -> None:
    """Print sparse token rewards and GRPO advantage stats per uid group."""
    batch = data.batch
    if "advantages" not in batch:
        return

    if "finish_step" in batch:
        outcome = batch["token_level_rewards"].sum(dim=-1) if "token_level_rewards" in batch else None
    else:
        outcome = None

    adv = batch["advantages"]
    # Per-trajectory scalar advantage (masked mean)
    if "finish_step" in batch:
        response_length = adv.shape[-1]
        steps = torch.arange(response_length, device=adv.device)
        mask = steps.unsqueeze(0) < (batch["finish_step"] * action_token_len).unsqueeze(1)
        adv_scalar = (adv * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    else:
        adv_scalar = adv.sum(dim=-1)

    adv_list = adv_scalar.detach().float().cpu().tolist()
    uids = _uid_labels(data, len(adv_list))

    if outcome is not None:
        outcome_list = outcome.detach().float().cpu().tolist()
    else:
        outcome_list = [0.0] * len(adv_list)

    header = "VAMPO [grpo]"
    if global_step is not None:
        header += f" global_step={global_step}"
    print(header, flush=True)

    by_uid: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for i, uid in enumerate(uids):
        by_uid[uid].append((i, float(outcome_list[i]), float(adv_list[i])))

    for uid, rows in by_uid.items():
        parts = [f"out={out:.2f}/adv={adv:.2f}" for _, out, adv in rows]
        print(f"  uid={uid[:8]} n={len(rows)}: " + " | ".join(parts), flush=True)

    mean_adv = sum(adv_list) / max(len(adv_list), 1)
    adv_fmt = f"{mean_adv:.4f}" if math.isfinite(mean_adv) else "nan"
    print(
        f"  >> mean_outcome={sum(outcome_list)/max(len(outcome_list),1):.4f} "
        f"mean_adv={adv_fmt} groups={len(by_uid)}",
        flush=True,
    )


def log_actor_update(
    metrics: dict,
    *,
    global_step: int | None = None,
) -> None:
    """Print PPO actor metrics returned from worker update_actor."""
    header = "VAMPO [update_actor]"
    if global_step is not None:
        header += f" global_step={global_step}"
    print(header, flush=True)
    keys = (
        "actor/pg_loss",
        "actor/entropy_loss",
        "actor/grad_norm",
        "actor/entropy",
        "timing/update_actor",
    )
    for key in keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key}={val:.6f}", flush=True)
            else:
                print(f"  {key}={val}", flush=True)
    extras = [k for k in sorted(metrics) if k.startswith("actor/") and k not in keys]
    for key in extras[:8]:
        print(f"  {key}={metrics[key]}", flush=True)


def log_step_banner(
    *,
    global_step: int,
    epoch: int,
    phase: str,
) -> None:
    print(f"VAMPO === global_step={global_step} epoch={epoch} {phase} ===", flush=True)
