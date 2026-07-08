#!/usr/bin/env bash
# RL 训练前检查（数据 / VLA / init_states / VideoMAE reward）
#
# checkpoint 未就绪时:
#   bash scripts/preflight/preflight_rl.sh --skip-reward
#
# 准备开训:
#   bash scripts/preflight/preflight_rl.sh --strict-reward
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export VAMVERL_ROOT="$ROOT"
export MODEL_PATH="${MODEL_PATH:-/home/robotem/Models/DreamZero-DROID}"
export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}"
export INIT_STATES_DIR="${INIT_STATES_DIR:-$ROOT/data/init_states}"
export DROID_SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
export VIDEOMAE_BACKBONE="${VIDEOMAE_BACKBONE:-/home/robotem/Models/videomae-base}"
export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$ROOT/checkpoints/videomae_droid.pth}"
CONFIG="${CONFIG:-$ROOT/configs/vampo_ppo_trainer.yaml}"

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi
ENV_PYTHON="${CONDA_PREFIX:-}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || ENV_PYTHON="python"

"${ENV_PYTHON}" "$ROOT/scripts/dev/ensure_libero_config.py"

ARGS=(--config "$CONFIG")
if [[ "${1:-}" == "--skip-reward" ]]; then
  ARGS+=(--skip-reward)
  shift
elif [[ "${1:-}" == "--strict-reward" ]]; then
  ARGS+=(--strict-reward)
  shift
fi
if [[ "${1:-}" == "--json" ]]; then
  ARGS+=(--json)
  shift
fi

exec "${ENV_PYTHON}" "$ROOT/verl/trainer/preflight/dreamzero_preflight.py" "${ARGS[@]}" "$@"
