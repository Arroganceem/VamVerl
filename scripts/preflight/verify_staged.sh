#!/usr/bin/env bash
# 分阶段验证 VamVerl 代码完整性（无需完整 RL 训练 / 14B 加载）
#
# GitHub / CI 默认（Stage 0–3，CPU，无 VideoMAE 权重）:
#   bash scripts/preflight/verify_staged.sh
#
# 含本地数据检查（Stage 1，需 DROID + init_states）:
#   bash scripts/preflight/verify_staged.sh --through 1
#
# 含 VideoMAE CPU 冒烟（Stage 4，需 ckpt + backbone，仍不加载 DreamZero）:
#   bash scripts/preflight/verify_staged.sh --through 4
#
# 开训前全量（Stage 5 = preflight --strict-reward）:
#   bash scripts/preflight/verify_staged.sh --through 5
#
# 仅跑 RL mock 链路:
#   bash scripts/preflight/verify_staged.sh --only rl-mock
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
export VIDEOMAE_VERIFY_DEVICE="${VIDEOMAE_VERIFY_DEVICE:-cpu}"
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

ARGS=(--config "$CONFIG")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --through)
      ARGS+=(--through "$2")
      shift 2
      ;;
    --only)
      ARGS+=(--only "$2")
      shift 2
      ;;
    --json)
      ARGS+=(--json)
      shift
      ;;
    --videomae-device)
      ARGS+=(--videomae-device "$2")
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

exec "${ENV_PYTHON}" "$ROOT/verl/trainer/preflight/verify_staged.py" "${ARGS[@]}"
