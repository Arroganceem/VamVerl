"""Canonical VamVerl environment variables (verl + DreamZero cluster training)."""

from __future__ import annotations

import os

# Primary names (propagate to Ray workers / runtime_env)
NCCL_TIMEOUT_MIN = "VAMVERL_NCCL_TIMEOUT_MIN"
RANK0_NODE_IP = "VAMVERL_RANK0_NODE_IP"
FSDP_SHARDED_CHECKPOINT = "VAMVERL_FSDP_SHARDED_CHECKPOINT"
PARALLEL_STRATEGY = "VAMVERL_PARALLEL_STRATEGY"
COOLDOWN_EVERY_STEPS = "VAMVERL_COOLDOWN_EVERY_STEPS"
COOLDOWN_SEC = "VAMVERL_COOLDOWN_SEC"
DEBUG_LOG_PROB = "VAMVERL_DEBUG_LOG_PROB"
DEBUG_REWARD = "VAMVERL_DEBUG_REWARD"
DEBUG_REWARD_DIR = "VAMVERL_DEBUG_REWARD_DIR"
FLOW_RL_SIGMA = "VAMVERL_FLOW_RL_SIGMA"
FLOW_RL_VIDEO_SIGMA = "VAMVERL_FLOW_RL_VIDEO_SIGMA"

_LEGACY_ALIASES: dict[str, str] = {
    NCCL_TIMEOUT_MIN: "VAMPO_NCCL_TIMEOUT_MIN",
    RANK0_NODE_IP: "VAMPO_RANK0_NODE_IP",
    FSDP_SHARDED_CHECKPOINT: "VAMPO_FSDP_SHARDED_CHECKPOINT",
    PARALLEL_STRATEGY: "VAMPO_PARALLEL_STRATEGY",
    COOLDOWN_EVERY_STEPS: "VAMPO_COOLDOWN_EVERY_STEPS",
    COOLDOWN_SEC: "VAMPO_COOLDOWN_SEC",
    DEBUG_LOG_PROB: "VAMPO_DEBUG_LOG_PROB",
    DEBUG_REWARD: "VAMPO_DEBUG_REWARD",
    DEBUG_REWARD_DIR: "VAMPO_DEBUG_REWARD_DIR",
    FLOW_RL_SIGMA: "VAMPO_FLOW_RL_SIGMA",
    FLOW_RL_VIDEO_SIGMA: "VAMPO_FLOW_RL_VIDEO_SIGMA",
}

RAY_RUNTIME_ENV_KEYS = (
    "PYTHONPATH",
    "VAMVERL_ROOT",
    "MODEL_PATH",
    "WAN21_DIR",
    "WAN22_DIR",
    "TOKENIZER_PATH",
    "HF_HUB_OFFLINE",
    "DROID_DATA_ROOT",
    "INIT_STATES_DIR",
    "DROID_SPLIT_DIR",
    "VIDEOMAE_CKPT",
    "VIDEOMAE_BACKBONE",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_MODE",
    NCCL_TIMEOUT_MIN,
    RANK0_NODE_IP,
    FSDP_SHARDED_CHECKPOINT,
    "RAY_HEAD_IP",
    "HEAD_IP",
)


def get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    legacy = _LEGACY_ALIASES.get(name)
    if legacy:
        return os.environ.get(legacy, default)
    return default


def setdefault(name: str, value: str) -> None:
    if os.environ.get(name):
        return
    legacy = _LEGACY_ALIASES.get(name)
    if legacy and os.environ.get(legacy):
        os.environ[name] = os.environ[legacy]
    else:
        os.environ.setdefault(name, value)


def pop(name: str) -> str | None:
    val = os.environ.pop(name, None)
    if val is not None:
        return val
    legacy = _LEGACY_ALIASES.get(name)
    if legacy:
        return os.environ.pop(legacy, None)
    return None
