"""Train VideoMAE success/failure classifier on DROID WebDataset shards."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from verl.utils.reward.videomae_dataset import (
    PrecomputedClipDataset,
    collate_clips,
    resolve_shard_globs,
)
from verl.utils.reward.videomae_load import (
    configure_trainable,
    count_trainable,
    load_videomae_classifier,
    optimizer_param_groups,
)
from verl.utils.reward.videomae_reward import DEFAULT_VIDEOMAE_BACKBONE
from verl.utils.reward.videomae_train_metrics import (
    EpochMetricAccumulator,
    MetricEMA,
    ddp_mean_scalar,
    ddp_mean_tensor,
    format_step_metrics,
)

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(os.environ.get("VAMVERL_ROOT", Path(__file__).resolve().parents[2]))


def _resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def _get_dist_env() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank, torch.device("cuda", local_rank)


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _training_cooldown_settings(train_cfg: dict[str, Any]) -> tuple[int, float]:
    """Return (every_steps, sleep_sec). Env VAMPO_COOLDOWN_* overrides yaml; 0 disables."""
    every = int(
        os.environ.get(
            "VAMPO_COOLDOWN_EVERY_STEPS",
            train_cfg.get(
                "cooldown_every_steps",
                train_cfg.get("pause_every_steps", 0) or 0,
            )
            or 0,
        )
    )
    sec = float(
        os.environ.get(
            "VAMPO_COOLDOWN_SEC",
            train_cfg.get(
                "cooldown_sec",
                train_cfg.get("pause_seconds", 0) or 0,
            )
            or 0,
        )
    )
    return max(every, 0), max(sec, 0.0)


def _maybe_training_cooldown(
    global_step: int,
    *,
    every_steps: int,
    cooldown_sec: float,
    rank: int,
) -> None:
    if every_steps <= 0 or cooldown_sec <= 0:
        return
    if global_step <= 0 or global_step % every_steps != 0:
        return
    if _is_rank0(rank):
        logger.info(
            "cooldown @ step %d: sleep %.0fs (all ranks barrier)",
            global_step,
            cooldown_sec,
        )
    dist.barrier()
    time.sleep(cooldown_sec)
    dist.barrier()


@torch.no_grad()
def evaluate_ddp(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    thresh_min: float,
    thresh_max: float,
    thresh_steps: int,
) -> tuple[dict[str, Any], dict[str, float]] | None:
    model.eval()
    logits_local: list[list[float]] = []
    trues_local: list[int] = []

    for vids, ys, _ in loader:
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids).logits
        logits_local.extend(logits.float().cpu().tolist())
        trues_local.extend(ys.cpu().tolist())

    logits_gather: list[Any] = [None] * world_size
    trues_gather: list[Any] = [None] * world_size
    dist.all_gather_object(logits_gather, logits_local)
    dist.all_gather_object(trues_gather, trues_local)
    if rank != 0:
        return None

    logits = [x for part in logits_gather for x in part]
    trues = [x for part in trues_gather for x in part]
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].numpy()

    thresholds = np.linspace(thresh_min, thresh_max, thresh_steps)
    all_metrics: dict[str, Any] = {}
    best = {"f1": -1.0, "thresh": float(thresholds[0])}

    for th in thresholds:
        preds = (probs >= th).astype(np.int32).tolist()
        f1 = f1_score(trues, preds, zero_division=0)
        all_metrics[f"thresh_{th:.2f}"] = OrderedDict(
            acc=accuracy_score(trues, preds),
            precision=precision_score(trues, preds, zero_division=0),
            recall=recall_score(trues, preds, zero_division=0),
            f1=f1,
        )
        if f1 > best["f1"]:
            best = {"f1": float(f1), "thresh": float(th)}

    return all_metrics, best


def _save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    *,
    threshold: float,
    step: int,
    f1: float,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"videomae_{tag}_step{step}_f1{f1:.4f}_th{threshold:.2f}"
    ckpt_path = out_dir / f"{stem}.pth"
    payload = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "threshold": float(threshold),
        "step": int(step),
        "f1": float(f1),
    }
    torch.save(payload, ckpt_path)
    sidecar = ckpt_path.with_suffix(".json")
    sidecar.write_text(json.dumps({"threshold": threshold, "f1": f1, "step": step}, indent=2))
    return ckpt_path


def _export_rl_checkpoint(
    out_dir: Path,
    model: nn.Module,
    *,
    threshold: float,
    step: int,
    f1: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "threshold": float(threshold),
        "step": int(step),
        "f1": float(f1),
    }
    pth = out_dir / "videomae_droid.pth"
    torch.save(payload, pth)
    (out_dir / "videomae_droid.json").write_text(
        json.dumps({"threshold": threshold, "f1": f1, "step": step}, indent=2)
    )
    logger.info("RL checkpoint → %s (threshold=%.3f)", pth, threshold)


def _build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    clip_dir = _resolve_data_path(
        data_cfg.get("clip_dir")
        or os.environ.get("VIDEOMAE_CLIP_DIR")
        or "data/videomae_droid_clips"
    )
    clip_manifest = clip_dir / "manifest.json"
    if not clip_manifest.is_file():
        raise RuntimeError(
            f"Missing clip dataset at {clip_dir}. Run: bash scripts/data/prep_component2_data_local.sh"
        )

    cm = json.loads(clip_manifest.read_text())
    train_globs = resolve_shard_globs(cm["train_glob"])
    val_globs = resolve_shard_globs(cm["val_glob"])
    if not train_globs or not val_globs:
        raise RuntimeError(f"Missing clip shards under {clip_dir}")

    tr_ds = PrecomputedClipDataset(
        train_globs,
        img_size=int(model_cfg.get("img_size", 224)),
        mode="train",
        backbone=model_cfg.get("name"),
        shuffle_buf=int(data_cfg.get("shuffle_buf", 512)),
        use_resample=bool(data_cfg.get("use_resample", True)),
    )
    va_ds = PrecomputedClipDataset(
        val_globs,
        img_size=int(model_cfg.get("img_size", 224)),
        mode="val",
        backbone=model_cfg.get("name"),
        shuffle_buf=0,
        use_resample=False,
    )
    runtime: dict[str, Any] = {
        "clip_dir": str(clip_dir),
        "train_clip_shards": len(train_globs),
        "val_clip_shards": len(val_globs),
        "train_clips": int(cm.get("train_clips", 0)),
        "val_clips": int(cm.get("val_clips", 0)),
    }

    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_clips,
        persistent_workers=bool(train_cfg.get("num_workers", 4) > 0),
        drop_last=True,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_clips,
        persistent_workers=bool(train_cfg.get("num_workers", 4) > 0),
        drop_last=False,
    )
    runtime["per_gpu_batch"] = int(train_cfg["batch_size"])
    return tr_ld, va_ld, runtime


def _class_weights(cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    pw = train_cfg.get("pos_weight", "auto")
    if pw is None or pw is False:
        return None
    if pw != "auto":
        return torch.tensor([1.0, float(pw)], device=device)

    split_dir = _resolve_data_path(
        data_cfg.get("split_dir")
        or os.environ.get("DROID_SPLIT_DIR", "data/splits")
    )
    ready_path = split_dir / "videomae_dataset_ready.json"
    if not ready_path.is_file():
        return None
    tr = json.loads(ready_path.read_text()).get("train", {})
    pos = int(tr.get("success_clips", 0))
    neg = int(tr.get("failure_clips", 0))
    if pos <= 0:
        return None
    ratio = neg / pos
    logger.info("Class weights auto: neg/pos=%.1f (pos=%d neg=%d)", ratio, pos, neg)
    return torch.tensor([1.0, ratio], device=device)


def _phase_for_epoch(epoch: int, train_cfg: dict[str, Any]) -> tuple[bool, int]:
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    if epoch <= freeze_epochs:
        return True, 0
    return False, int(train_cfg.get("unfreeze_last_n_layers", 0))


def train(config_path: str | Path) -> None:
    cfg = _load_config(config_path)
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    out_dir = _resolve_data_path(cfg.get("output", {}).get("dir", "./checkpoints"))

    world_size, rank, local_rank, device = _get_dist_env()
    if _is_rank0(rank):
        logger.info("DDP world_size=%d config=%s", world_size, config_path)

    tr_ld, va_ld, runtime = _build_loaders(cfg)
    if _is_rank0(rank):
        logger.info("Data runtime: %s", runtime)

    model, backbone_path = load_videomae_classifier(
        cfg["model"].get("name"),
        window=int(cfg["model"].get("window", 8)),
        device=device,
    )
    configure_trainable(
        model,
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
    )
    model = DDP(model, device_ids=[local_rank])
    class_weights = _class_weights(cfg, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if _is_rank0(rank) and class_weights is not None:
        logger.info("CrossEntropyLoss weights=%s", class_weights.tolist())

    wandb_run = None
    wandb_cfg = cfg.get("wandb", {})
    if bool(wandb_cfg.get("enable", False)) and _is_rank0(rank):
        import wandb

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "vamverl")),
            name=str(wandb_cfg.get("run_name", "videomae_droid")),
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
            notes=wandb_cfg.get("notes"),
            config={**cfg, "runtime": {**runtime, "backbone": backbone_path}},
        )

    max_epochs = int(train_cfg.get("max_epochs", 1))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1000))
    max_steps = int(train_cfg.get("max_steps", max_epochs * steps_per_epoch))
    eval_every = int(train_cfg.get("eval_every", 500))
    log_every = int(train_cfg.get("log_every", 10))
    loss_ema_beta = float(train_cfg.get("loss_ema_beta", 0.99))
    grad_clip = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    cooldown_every_steps, cooldown_sec = _training_cooldown_settings(train_cfg)
    cooldown_enabled = cooldown_every_steps > 0 and cooldown_sec > 0.0
    if _is_rank0(rank) and cooldown_enabled:
        logger.info(
            "Training cooldown: every %d steps sleep %.0fs",
            cooldown_every_steps,
            cooldown_sec,
        )

    global_step = 0
    best_f1 = -1.0
    best_thresh = float(eval_cfg.get("thresh_min", 0.3))
    metric_ema = MetricEMA(loss_ema_beta) if loss_ema_beta > 0.0 else None
    optimizer: torch.optim.Optimizer | None = None
    current_phase = (-1, -1)

    for epoch in range(1, max_epochs + 1):
        freeze_backbone, unfreeze_n = _phase_for_epoch(epoch, train_cfg)
        phase_key = (int(freeze_backbone), int(unfreeze_n))
        if phase_key != current_phase:
            configure_trainable(
                model.module,
                freeze_backbone=freeze_backbone,
                unfreeze_last_n_layers=unfreeze_n,
            )
            head_lr = float(
                train_cfg.get("head_lr_finetune" if unfreeze_n else "head_lr", 1e-3)
            )
            backbone_lr = float(train_cfg.get("lr", 1e-5))
            optimizer = torch.optim.AdamW(
                optimizer_param_groups(
                    model.module,
                    head_lr=head_lr,
                    backbone_lr=backbone_lr,
                    weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                )
            )
            trainable, total = count_trainable(model.module)
            if _is_rank0(rank):
                logger.info(
                    "Epoch %d phase freeze=%s unfreeze_last=%d trainable=%d/%d lrs head=%.1e backbone=%.1e",
                    epoch,
                    freeze_backbone,
                    unfreeze_n,
                    trainable,
                    total,
                    head_lr,
                    backbone_lr,
                )
            current_phase = phase_key

        assert optimizer is not None
        tr_iter = iter(tr_ld)
        epoch_metrics = EpochMetricAccumulator()
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{max_epochs}",
            disable=not _is_rank0(rank),
        )
        epoch_start = time.time()
        for step_in_epoch in pbar:
            if global_step >= max_steps:
                break
            try:
                vids, ys, _ = next(tr_iter)
            except StopIteration:
                tr_iter = iter(tr_ld)
                vids, ys, _ = next(tr_iter)

            step_t0 = time.perf_counter()
            model.train()
            vids = vids.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            logits = model(pixel_values=vids).logits
            loss = criterion(logits, ys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            _maybe_training_cooldown(
                global_step,
                every_steps=cooldown_every_steps,
                cooldown_sec=cooldown_sec,
                rank=rank,
            )

            loss_ddp = float(ddp_mean_tensor(loss, world_size).item())
            pos_rate = ddp_mean_scalar(float(ys.float().mean().item()), device, world_size)
            row: dict[str, float] = {
                "loss": loss_ddp,
                "loss_raw": float(loss.item()),
                "batch_pos": pos_rate,
            }
            if metric_ema is not None:
                row.update(metric_ema.update({"loss": loss_ddp}))
            epoch_metrics.add(row)
            step_sec = time.perf_counter() - step_t0

            if _is_rank0(rank):
                status = format_step_metrics(row, step_sec=step_sec)
                pbar.set_postfix_str(status)
                if global_step % log_every == 0:
                    elapsed = max(time.time() - epoch_start, 1e-6)
                    sps = step_in_epoch / elapsed
                    logger.info(
                        "epoch=%d/%d step=%d/%d global=%d %s sps=%.2f",
                        epoch,
                        max_epochs,
                        step_in_epoch + 1,
                        steps_per_epoch,
                        global_step,
                        status,
                        sps,
                    )
                    if wandb_run is not None:
                        wandb_payload = {
                            "train/loss": row["loss"],
                            "train/loss_raw": row["loss_raw"],
                            "train/batch_pos": pos_rate,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/step_sec": step_sec,
                        }
                        if "loss_ema" in row:
                            wandb_payload["train/loss_ema"] = row["loss_ema"]
                        wandb_run.log(wandb_payload, step=global_step)

            if global_step % eval_every == 0:
                out = evaluate_ddp(
                    model,
                    va_ld,
                    device,
                    rank=rank,
                    world_size=world_size,
                    thresh_min=float(eval_cfg.get("thresh_min", 0.3)),
                    thresh_max=float(eval_cfg.get("thresh_max", 1.0)),
                    thresh_steps=int(eval_cfg.get("thresh_steps", 20)),
                )
                dist.barrier()
                if _is_rank0(rank) and out is not None:
                    all_metrics, best = out
                    logger.info(
                        "Val @ step %d best_f1=%.4f thresh=%.2f",
                        global_step,
                        best["f1"],
                        best["thresh"],
                    )
                    _save_checkpoint(
                        out_dir,
                        model,
                        threshold=best["thresh"],
                        step=global_step,
                        f1=best["f1"],
                        tag="step",
                    )
                    if best["f1"] > best_f1:
                        best_f1 = best["f1"]
                        best_thresh = best["thresh"]
                        _export_rl_checkpoint(
                            out_dir,
                            model,
                            threshold=best_thresh,
                            step=global_step,
                            f1=best_f1,
                        )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/best_f1": best["f1"],
                                "val/best_thresh": best["thresh"],
                            },
                            step=global_step,
                        )
                dist.barrier()

            if global_step >= max_steps:
                break

        if _is_rank0(rank) and epoch_metrics.count > 0:
            ep = epoch_metrics.mean()
            logger.info(
                "epoch %d/%d done steps=%d avg_loss=%.4f global=%d",
                epoch,
                max_epochs,
                epoch_metrics.count,
                ep.get("loss", 0.0),
                global_step,
            )

        if global_step >= max_steps:
            break

    if _is_rank0(rank):
        logger.info(
            "Training done steps=%d best_f1=%.4f threshold=%.3f",
            global_step,
            best_f1,
            best_thresh,
        )
        if wandb_run is not None:
            wandb_run.finish()
    dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Train VideoMAE DROID success classifier")
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "VIDEOMAE_TRAIN_CONFIG",
            str(Path(__file__).resolve().parent / "configs/videomae_droid.yaml"),
        ),
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
