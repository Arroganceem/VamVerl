"""WMPO-style VideoMAE sliding-window success scan (robwm_rollout.predict_success)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SlidingSuccess:
    """Binary outcome + frame index of earliest success window (WMPO layout)."""

    complete: bool
    finish_step: int


def build_sliding_clips(
    video: np.ndarray,
    *,
    window_size: int = 8,
    stride: int = 1,
    min_steps: int = 32,
) -> list[tuple[np.ndarray, int, int]]:
    """Chronological 8-frame clips; only windows ending after ``min_steps``."""
    total_frames = int(video.shape[0])
    if total_frames <= 0:
        return []
    min_end = max(window_size, min_steps)
    min_end = min(min_end, total_frames)
    clips: list[tuple[np.ndarray, int, int]] = []
    for end in range(total_frames, min_end - 1, -stride):
        clip = video[end - window_size : end]
        clips.append((clip, end - window_size, end))
    return clips[::-1]


def scan_clips_for_success(
    clips: list[tuple[np.ndarray, int, int]],
    *,
    probs_fn,
    threshold: float,
    batch_size: int = 32,
) -> SlidingSuccess:
    """Return earliest window with P(success) >= threshold (time-forward scan)."""
    if not clips:
        return SlidingSuccess(complete=False, finish_step=0)

    total_frames = clips[-1][2]
    finish_step = total_frames - 1
    complete = False

    for i in range(0, len(clips), batch_size):
        batch = clips[i : i + batch_size]
        ranges = [(c[1], c[2]) for c in batch]
        clip_imgs = [[frame for frame in c[0]] for c in batch]
        probs = probs_fn(clip_imgs)
        for (start, end), prob_row in zip(ranges, probs):
            success_prob = float(prob_row[1])
            if success_prob >= threshold and end - 1 < finish_step:
                finish_step = end - 1
                complete = True
                return SlidingSuccess(complete=True, finish_step=int(finish_step))

    return SlidingSuccess(complete=complete, finish_step=int(finish_step))


def frame_finish_to_wm_step(
    finish_frame: int,
    *,
    frames_per_chunk: int,
    max_wm_steps: int,
) -> int:
    """Map VideoMAE frame index to WM-step count for verl ``finish_step`` mask."""
    wm = max(1, int(np.ceil(finish_frame / max(frames_per_chunk, 1))))
    return min(wm, max_wm_steps)


def batched_success_probs(
    model,
    feature_extractor,
    clip_imgs: list[list[np.ndarray]],
    device: torch.device,
) -> np.ndarray:
    """Run VideoMAE on a batch of clips; return (B, 2) softmax probabilities."""
    inputs = feature_extractor(clip_imgs, return_tensors="pt")["pixel_values"].to(device)
    logits = model(pixel_values=inputs).logits
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()
