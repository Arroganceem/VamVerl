"""Shared observation helpers for policy backends."""

from __future__ import annotations

import numpy as np
import torch

OXE_DROID_VIDEO_KEYS = (
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
)

_LEROBOT_IMAGE_PREFIX = "observation.images."

# DreamZero causal WM: 1 frame on first step, then 4 (matches serve_dreamzero_wan22).
FRAMES_PER_WM_CHUNK = 4


def is_video_obs_key(key: str) -> bool:
    return key.startswith(_LEROBOT_IMAGE_PREFIX) or key.startswith("video.")


def init_frame_buffers(obs: dict) -> dict[str, list[np.ndarray]]:
    """Seed per-camera frame lists from init obs (typically T=1)."""
    buffers: dict[str, list[np.ndarray]] = {}
    for key, value in obs.items():
        if not is_video_obs_key(key):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 4:
            video = _normalize_video_array(value)
            buffers[key] = [video[t] for t in range(video.shape[0])]
    return buffers


def build_obs_with_video_history(
    obs: dict,
    buffers: dict[str, list[np.ndarray]],
    *,
    is_first_wm_step: bool,
    prompt: str = "",
) -> dict:
    """Build obs with T=1 (first WM step) or T=4 (later steps) video stacks."""
    num_frames = 1 if is_first_wm_step else FRAMES_PER_WM_CHUNK
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    for key, buf in buffers.items():
        if not buf:
            continue
        frames_to_use = buf[-num_frames:] if len(buf) >= num_frames else list(buf)
        while len(frames_to_use) < num_frames:
            frames_to_use.insert(0, buf[0])
        out[key] = np.stack(frames_to_use, axis=0).astype(np.uint8)
    text = _prompt_text(out, prompt)
    if text:
        out["annotation.language.action_text"] = text
        out.setdefault("annotation.language.language_instruction", text)
    return out


def append_imagined_frame(
    buffers: dict[str, list[np.ndarray]],
    video_frames: np.ndarray,
) -> None:
    """Append latest imagined frame to all camera buffers (closed-loop feedback)."""
    frames = np.asarray(video_frames)
    if frames.ndim == 4:
        imagined = frames[-1]
    elif frames.ndim == 3:
        imagined = frames
    else:
        raise ValueError(f"Expected imagined video (T,H,W,C) or (H,W,C), got {frames.shape}")
    for key, buf in buffers.items():
        if not buf:
            continue
        target_h, target_w = int(buf[0].shape[0]), int(buf[0].shape[1])
        frame = imagined
        if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
            frame = resize_video_frames(frame[np.newaxis], target_h, target_w)[0]
        buf.append(frame.astype(np.uint8, copy=False))


def flatten_obs(obs: dict, feat_dim: int = 512) -> torch.Tensor:
    parts = []
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, np.ndarray):
            parts.append(torch.from_numpy(v.astype(np.float32).reshape(-1)))
        elif isinstance(v, str):
            parts.append(torch.tensor([float(len(v))]))
    if not parts:
        x = torch.zeros(feat_dim)
    else:
        x = torch.cat(parts)
        if x.numel() > feat_dim:
            x = x[:feat_dim]
        elif x.numel() < feat_dim:
            x = torch.cat([x, torch.zeros(feat_dim - x.numel())])
    return x


def resize_video_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize (T,H,W,C) uint8 video to (*, height, width, C)."""
    video = np.asarray(frames)
    if video.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C), got {video.shape}")
    if video.shape[1] == height and video.shape[2] == width:
        return video.astype(np.uint8, copy=False)
    try:
        import cv2

        resized = [
            cv2.resize(video[t], (width, height), interpolation=cv2.INTER_LINEAR)
            for t in range(video.shape[0])
        ]
    except ImportError:
        from PIL import Image

        resized = [
            np.array(Image.fromarray(video[t]).resize((width, height), Image.BILINEAR))
            for t in range(video.shape[0])
        ]
    return np.stack(resized, axis=0).astype(np.uint8)


def base_frame_from_obs(obs: dict, default_shape: tuple[int, int, int] = (64, 64, 3)) -> np.ndarray:
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            frame = v[-1] if v.ndim == 4 else v
            if frame.shape[-1] in (1, 3):
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                return frame.astype(np.uint8)
    return np.zeros(default_shape, dtype=np.uint8)


def _normalize_video_array(arr: np.ndarray) -> np.ndarray:
    """Ensure uint8 video with shape (T, H, W, C)."""
    video = np.asarray(arr)
    if video.dtype != np.uint8:
        video = video.astype(np.uint8)
    while video.ndim > 4 and video.shape[0] == 1:
        video = video[0]
    if video.ndim == 3 and video.shape[-1] in (1, 3):
        if video.shape[0] == 1 and video.shape[1] > 8:
            raise ValueError(
                f"Corrupted init_state video shape {video.shape} (height=1). "
                "Rebuild init_states: INIT_STATES_FORCE=1 bash scripts/build_init_states_from_droid.sh"
            )
        return video[np.newaxis, ...]
    if video.ndim == 4:
        if video.shape[-3] == 1 and video.shape[-2] > 8:
            raise ValueError(
                f"Corrupted init_state video shape {video.shape} (height=1). "
                "Rebuild init_states: INIT_STATES_FORCE=1 bash scripts/build_init_states_from_droid.sh"
            )
        return video
    if video.ndim == 5 and video.shape[0] == 1:
        return video[0]
    raise ValueError(f"Expected video (H,W,C) or (T,H,W,C), got shape {video.shape}")


def _prompt_text(obs: dict, prompt: str) -> str:
    if prompt:
        return prompt
    for key in (
        "annotation.language.action_text",
        "annotation.language.language_instruction",
        "prompt",
    ):
        if key not in obs:
            continue
        value = obs[key]
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else str(value[0])
        return str(value)
    return ""


def convert_rl_obs_to_vla_obs(
    obs: dict,
    prompt: str = "",
    *,
    required_video_keys: tuple[str, ...] = OXE_DROID_VIDEO_KEYS,
) -> dict:
    """Map LeRobot / RL rollout obs keys to GrootSimPolicy eval_transform keys."""
    out: dict = {}
    for key, value in obs.items():
        if key.startswith(_LEROBOT_IMAGE_PREFIX):
            model_key = "video." + key[len(_LEROBOT_IMAGE_PREFIX) :]
            out[model_key] = _normalize_video_array(value)
        elif key.startswith("video.") and isinstance(value, np.ndarray):
            out[key] = _normalize_video_array(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    available = [key for key in required_video_keys if key in out]
    if available:
        fallback = out[available[0]]
        for key in required_video_keys:
            if key not in out:
                out[key] = fallback.copy()

    text = _prompt_text(out, prompt)
    if text:
        out.setdefault("annotation.language.action_text", text)
        out.setdefault("annotation.language.language_instruction", text)
    return out
