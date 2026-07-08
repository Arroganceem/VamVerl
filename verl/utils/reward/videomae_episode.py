"""Load DROID episodes for VideoMAE clip export."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from verl.utils.data.lerobot_io import (
    load_episode_parquet,
    load_episodes,
    read_mp4_frames_cropped,
    resolve_video_path,
)

logger = logging.getLogger(__name__)

DEFAULT_CAMERA = "exterior_image_1_left"
DEFAULT_MAX_FRAMES = 256
DEFAULT_CROP = "tail"


def resize_clip_uint8(clip: np.ndarray, size: int = 224) -> np.ndarray:
    """clip: (T,H,W,3) uint8 -> (T,size,size,3) uint8"""
    if clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3), got {clip.shape}")
    out = np.empty((clip.shape[0], size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(clip):
        out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


def _episode_success(episode: dict, frame_count: int) -> bool:
    for key in ("success", "is_success", "task_success"):
        if key in episode:
            return bool(episode[key])
    return frame_count > 0


def load_episode_meta_map(dataset_root: Path) -> dict[int, dict]:
    episodes = load_episodes(dataset_root)
    return {int(ep["episode_index"]): ep for ep in episodes}


def episode_success(
    episode_index: int,
    frame_count: int,
    episode_meta: dict[int, dict] | None = None,
) -> bool:
    ep = (episode_meta or {}).get(int(episode_index), {})
    return _episode_success(ep, frame_count)


def export_episode(
    dataset_root: Path,
    episode_index: int,
    *,
    camera: str = DEFAULT_CAMERA,
    max_frames: int = DEFAULT_MAX_FRAMES,
    crop: str = DEFAULT_CROP,
    episode_meta: dict[int, dict] | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Read one DROID episode as uint8 video + meta for success-window training."""
    video_path = resolve_video_path(dataset_root, episode_index, camera)
    if not video_path.is_file():
        logger.debug("Skip ep=%d missing video %s", episode_index, video_path)
        return None
    try:
        video, start_offset, total_frames = read_mp4_frames_cropped(
            video_path, max_frames, crop=crop
        )
    except Exception as exc:
        logger.warning("Skip ep=%d read failed: %s", episode_index, exc)
        return None
    if video.shape[0] < 8:
        logger.warning("Skip ep=%d too short T=%d", episode_index, video.shape[0])
        return None

    finish_abs = total_frames - 1
    try:
        df = load_episode_parquet(dataset_root, episode_index)
        finish_abs = min(int(len(df) - 1), total_frames - 1)
    except FileNotFoundError:
        pass

    if finish_abs < start_offset:
        logger.warning(
            "Skip ep=%d finish=%d before crop start=%d (crop=%s)",
            episode_index,
            finish_abs,
            start_offset,
            crop,
        )
        return None

    finish_step = finish_abs - start_offset
    finish_step = min(finish_step, int(video.shape[0] - 1))

    meta = {
        "episode_index": int(episode_index),
        "finish_step": int(finish_step),
        "complete": episode_success(episode_index, total_frames, episode_meta),
        "success": episode_success(episode_index, total_frames, episode_meta),
        "camera": camera,
        "num_frames": int(video.shape[0]),
        "max_frames": int(max_frames),
        "crop": crop,
        "start_offset": int(start_offset),
        "total_frames": int(total_frames),
    }
    return video.astype(np.uint8), meta
