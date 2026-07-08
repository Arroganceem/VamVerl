#!/usr/bin/env bash
# Component2 开训前检查（41 上运行）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
CLIP_DIR="${VIDEOMAE_CLIP_DIR:-$ROOT/data/videomae_droid_clips}"
BACKBONE="${VIDEOMAE_BACKBONE:-/home/robotem/Models/videomae-base}"
CONFIG="${CONFIG:-$ROOT/configs/videomae_droid.yaml}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export VAMVERL_ROOT="${ROOT}"

fail=0
_ok() { echo "  OK  $*"; }
_warn() { echo "  WARN $*"; }
_bad() { echo "  FAIL $*"; fail=1; }

echo ">>> Component2 训练前检查"
echo ">>> ROOT=${ROOT}"

# prep 产物
if [[ -f "${SPLIT_DIR}/videomae_dataset_ready.json" ]]; then
  _ok "dataset_ready: ${SPLIT_DIR}/videomae_dataset_ready.json"
else
  _bad "缺少 ${SPLIT_DIR}/videomae_dataset_ready.json（prep 未完成）"
fi

if [[ -f "${CLIP_DIR}/manifest.json" ]]; then
  _ok "clip manifest: ${CLIP_DIR}/manifest.json"
else
  _bad "缺少 ${CLIP_DIR}/manifest.json（prep 未完成）"
fi

train_shards=$(find "${CLIP_DIR}/train" -name '*.tar' 2>/dev/null | wc -l)
val_shards=$(find "${CLIP_DIR}/val" -name '*.tar' 2>/dev/null | wc -l)
if [[ "${train_shards}" -gt 0 ]]; then
  _ok "train tar shards: ${train_shards}"
else
  _bad "train tar 为空"
fi
if [[ "${val_shards}" -gt 0 ]]; then
  _ok "val tar shards: ${val_shards}"
else
  _bad "val tar 为空（prep 可能仍在写 train，等 prep 完成）"
fi

# backbone
if [[ -f "${BACKBONE}/config.json" ]]; then
  _ok "VideoMAE backbone: ${BACKBONE}"
else
  _bad "缺少 backbone: ${BACKBONE}"
fi

# GPU
if python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null; then
  _ok "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
  _warn "本机 CUDA 不可用（41 rank0 需要 GPU）"
fi

# loader 干跑
python - <<PY || { _bad "DataLoader 构建失败"; }
import json
from pathlib import Path
from vampo.reward.train_videomae import _build_loaders, _load_config, _class_weights

cfg = _load_config("${CONFIG}")
tr_ld, va_ld, runtime = _build_loaders(cfg)
print("  OK  DataLoader:", runtime)
w = _class_weights(cfg, __import__('torch').device('cpu'))
print("  OK  class_weights:", w.tolist() if w is not None else None)
r = json.loads(Path("${SPLIT_DIR}/videomae_dataset_ready.json").read_text())
print(f"  OK  train clips success={r['train']['success_clips']} failure={r['train']['failure_clips']}")
PY

# 集群 worker SSH
for ip in 31 21 11; do
  if ssh -o BatchMode=yes -o ConnectTimeout=5 -o "BindAddress=${SRC_BIND:-192.168.88.41}" \
    "${REMOTE_USER:-robotem}@192.168.88.${ip}" "test -d ${ROOT}" 2>/dev/null; then
    _ok "worker 192.168.88.${ip} VamVerl 目录存在"
  else
    _warn "worker 192.168.88.${ip} 无 ${ROOT} → bash scripts/mount_nfs_cluster4.sh clips"
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=5 -o "BindAddress=${SRC_BIND:-192.168.88.41}" \
    "${REMOTE_USER:-robotem}@192.168.88.${ip}" "test -f ${CLIP_DIR}/manifest.json" 2>/dev/null; then
    _ok "worker 192.168.88.${ip} clip manifest 可读"
  else
    _warn "worker 192.168.88.${ip} 读不到 clip → bash scripts/mount_nfs_cluster4.sh clips"
  fi
done

echo ""
if [[ "${fail}" -eq 0 ]]; then
  echo ">>> 检查通过，可以开训:"
  echo "    bash scripts/train_component2_reward_cluster4.sh"
else
  echo ">>> 检查未通过，请先完成 prep / 挂载 / 同步"
  exit 1
fi
