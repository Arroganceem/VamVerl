"""Load VideoMAE classifier backbone and configure trainable parameters."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from verl.utils.reward.videomae_reward import resolve_videomae_backbone

try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except ImportError:
    from transformers import VideoMAEFeatureExtractor  # type: ignore


def load_videomae_classifier(
    backbone: str | None,
    *,
    window: int = 8,
    num_labels: int = 2,
    device: torch.device | str | None = None,
) -> tuple[VideoMAEForVideoClassification, Any]:
    """Build a 2-class VideoMAE classifier from a local backbone directory."""
    backbone_path, local_only = resolve_videomae_backbone(backbone)
    load_kw = {"local_files_only": local_only}
    cfg = VideoMAEConfig.from_pretrained(
        backbone_path, num_frames=window, num_labels=num_labels, **load_kw
    )
    model = VideoMAEForVideoClassification.from_pretrained(
        backbone_path,
        config=cfg,
        ignore_mismatched_sizes=True,
        **load_kw,
    )
    if device is not None:
        model = model.to(device)
    return model, backbone_path


def feature_extractor(backbone: str | None, *, img_size: int = 224):
    backbone_path, local_only = resolve_videomae_backbone(backbone)
    return VideoMAEFeatureExtractor.from_pretrained(
        backbone_path, size=img_size, local_files_only=local_only
    )


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def configure_trainable(
    model: VideoMAEForVideoClassification,
    *,
    freeze_backbone: bool,
    unfreeze_last_n_layers: int = 0,
) -> None:
    """Freeze/unfreeze encoder blocks for phased fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if freeze_backbone or unfreeze_last_n_layers <= 0:
        return

    encoder = model.videomae.encoder
    layers = encoder.layer
    n = min(int(unfreeze_last_n_layers), len(layers))
    for layer in layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True


def optimizer_param_groups(
    model: VideoMAEForVideoClassification,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append(
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
        )
    groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups
