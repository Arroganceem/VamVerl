"""VideoMAE success classifier — WMPO robwm_rollout.predict_success compatible."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except ImportError:
    from transformers import VideoMAEFeatureExtractor  # type: ignore

from vampo.reward.base import SuccessResult
from vampo.reward.wmpo_sliding import (
    batched_success_probs,
    build_sliding_clips,
    scan_clips_for_success,
)

logger = logging.getLogger(__name__)

DEFAULT_VIDEOMAE_BACKBONE = "/home/robotem/Models/videomae-base"


def resolve_videomae_backbone(hf_model_id: str | None = None) -> tuple[str, bool]:
    raw = hf_model_id or os.environ.get("VIDEOMAE_BACKBONE") or DEFAULT_VIDEOMAE_BACKBONE
    path = Path(raw).expanduser()
    if path.is_dir() or path.is_file():
        return str(path.resolve()), True
    raise FileNotFoundError(
        f"VideoMAE backbone not found locally: {raw!r}. "
        f"Place weights under {DEFAULT_VIDEOMAE_BACKBONE!r} or set VIDEOMAE_BACKBONE."
    )


def load_videomae_threshold(checkpoint_path: str, default: float = 0.82) -> float:
    ckpt = Path(checkpoint_path)
    for sidecar in (ckpt.with_suffix(".json"), ckpt.with_name(ckpt.stem + ".threshold.json")):
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            if "threshold" in data:
                return float(data["threshold"])
            if "thresh" in data:
                return float(data["thresh"])
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "threshold" in state:
            return float(state["threshold"])
    except Exception:
        pass
    return default


class VideoMAERewardModel:
    """Frozen VideoMAE sliding-window verifier (same semantics as WMPO rollout)."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        threshold: float | None = None,
        img_size: int = 224,
        window_size: int = 8,
        min_steps: int = 32,
        batch_size: int = 32,
        device: str | None = None,
        hf_model_id: str | None = None,
    ):
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"VideoMAE checkpoint not found: {checkpoint_path}")

        self.threshold = float(threshold if threshold is not None else load_videomae_threshold(checkpoint_path))
        self.window_size = int(window_size)
        self.min_steps = int(min_steps)
        self.batch_size = max(1, int(batch_size))
        self.img_size = int(img_size)
        self._checkpoint_path = str(ckpt)
        self._backbone_path, self._local_only = resolve_videomae_backbone(hf_model_id)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = None
        self.model = None

        logger.info(
            "VideoMAE reward (lazy): ckpt=%s threshold=%.3f window=%d min_steps=%d device=%s",
            checkpoint_path,
            self.threshold,
            self.window_size,
            self.min_steps,
            self.device,
        )

    def _load_if_needed(self) -> None:
        if self.model is not None:
            self.ensure_on_device(self.device)
            return
        load_kw = {"local_files_only": self._local_only}
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(
            self._backbone_path, size=self.img_size, **load_kw
        )
        cfg = VideoMAEConfig.from_pretrained(
            self._backbone_path, num_frames=self.window_size, num_labels=2, **load_kw
        )
        self.model = VideoMAEForVideoClassification.from_pretrained(
            self._backbone_path,
            config=cfg,
            ignore_mismatched_sizes=True,
            **load_kw,
        ).to(self.device)
        state = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        logger.info("VideoMAE reward loaded on %s", self.device)

    def offload(self) -> None:
        """Drop weights between trajectories (GB10 unified memory — release, not CPU pin)."""
        self.model = None
        self.feature_extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_on_device(self, device: str | torch.device | None = None) -> None:
        target = torch.device(device or self.device)
        self.device = target
        if self.model is None:
            return
        if next(self.model.parameters()).device != target:
            self.model.to(target)

    def _probs_fn(self, clip_imgs: list[list[np.ndarray]]) -> np.ndarray:
        assert self.model is not None and self.feature_extractor is not None
        return batched_success_probs(
            self.model, self.feature_extractor, clip_imgs, self.device
        )

    @torch.no_grad()
    def predict_success(
        self,
        video: np.ndarray,
        prompt: str = "",
        batch_size: int = 32,
    ) -> SuccessResult:
        del prompt
        self._load_if_needed()
        clips = build_sliding_clips(
            video,
            window_size=self.window_size,
            min_steps=self.min_steps,
        )
        outcome = scan_clips_for_success(
            clips,
            probs_fn=self._probs_fn,
            threshold=self.threshold,
            batch_size=max(1, int(batch_size or self.batch_size)),
        )
        return SuccessResult(
            complete=outcome.complete,
            finish_step=outcome.finish_step,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        videos: list[np.ndarray],
        prompt: str = "",
        batch_size: int = 32,
    ) -> list[SuccessResult]:
        del prompt
        return [
            self.predict_success(v, batch_size=batch_size)
            for v in videos
        ]
