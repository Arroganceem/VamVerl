#!/usr/bin/env bash
# 四机 VamVerl 代码同步 · 默认仅 rsync + import 校验（不 pip install）
#
# NFS 已挂载时自动跳过 rsync（代码与 41 共享）。
#
# 用法:
#   bash scripts/sync_vamverl_cluster.sh              # 同步 + 校验（默认不 install）
#   bash scripts/sync_vamverl_cluster.sh --install      # 同步后在各节点 pip install -e
#   bash scripts/sync_vamverl_cluster.sh status         # 查看各节点 NFS/校验状态
#   WORKER_IPS="21" bash scripts/sync_vamverl_cluster.sh
set -euo pipefail

VAMVERL_ROOT="${VAMVERL_ROOT:-${HOME}/WAM/VamVerl}"
ENV_NAME="${ENV_NAME:-vamverl}"
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
WORKER_HOSTS=(gx10-400b gx10-4df7 gx10-39b7)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30 -o "BindAddress=${SRC_BIND}")
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
REMOTE_ENV="${HOME}/anaconda3/envs/${ENV_NAME}"
VERIFY_PY="import numpy, numba, albumentations, tianshou, ftfy, vampo, verl; assert numpy.__version__ < '2.5', numpy.__version__; from vampo.integrations.verl.parallel_utils import get_actor_strategy; print('vampo verl ok', 'numpy', numpy.__version__, 'strategy', get_actor_strategy(__import__('omegaconf').DictConfig({'actor': {'strategy': 'fsdp'}})))"
NUMPY_PIN="${NUMPY_PIN:-numpy>=2,<2.5}"
# 默认不 install；SYNC_INSTALL=1 或 --install 才跑 pip
DO_INSTALL="${SYNC_INSTALL:-0}"

if [[ -n "${WORKER_IPS:-}" ]]; then
  WORKERS=()
  for ip in ${WORKER_IPS}; do
    [[ "${ip}" == *.* ]] && WORKERS+=("${ip##*.}") || WORKERS+=("${ip}")
  done
else
  WORKERS=(31 21 11)
fi

RSYNC_EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude 'outputs'
  --exclude 'wandb'
  --exclude 'logs'
  --exclude '.pytest_cache'
  --exclude 'data'
  --exclude 'data/**'
)

[[ -d "${VAMVERL_ROOT}" ]] || { echo "ERROR: ${VAMVERL_ROOT} 不存在"; exit 1; }

_ssh_ok() {
  local ip="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" "true" 2>/dev/null
}

_nfs_vamverl() {
  local ip="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" \
    "mountpoint -q '${VAMVERL_ROOT}' 2>/dev/null && mount | grep -F ' ${VAMVERL_ROOT} ' | grep -qE ' type nfs'" \
    2>/dev/null
}

_verify_remote() {
  local ip="$1"
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" bash -s <<REMOTE
set -euo pipefail
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
PYTHONPATH="${VAMVERL_ROOT}:\${PYTHONPATH:-}" "${REMOTE_ENV}/bin/python" -c "${VERIFY_PY}"
REMOTE
}

cmd_status() {
  local ip host failed=0
  echo ">>> VamVerl 同步状态（NFS 挂载则无需 rsync）"
  for i in "${!WORKERS[@]}"; do
    ip="${WORKERS[$i]}"
    host="${WORKER_HOSTS[$i]:-worker}"
    echo -n ">>> 192.168.88.${ip} (${host}): "
    if ! _ssh_ok "${ip}"; then
      echo "SSH 不可达"
      failed=1
      continue
    fi
    if _nfs_vamverl "${ip}"; then
      echo -n "NFS "
    else
      echo -n "本地副本 "
    fi
    if _verify_remote "${ip}" >/dev/null 2>&1; then
      echo "verify OK"
    else
      echo "verify FAIL → bash $0 --install"
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || exit 1
}

_sync_one() {
  local ip="$1"
  local host="$2"
  echo ""
  echo ">>> 192.168.88.${ip} (${host})"
  if ! _ssh_ok "${ip}"; then
    echo "ERROR: SSH 192.168.88.${ip} 不可达" >&2
    return 1
  fi
  if _nfs_vamverl "${ip}"; then
    echo "  OK  VamVerl 已 NFS 挂载 41 → 跳过 rsync"
  else
    echo ">>> rsync 代码 → 192.168.88.${ip}（不含 data/）"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" "mkdir -p ${VAMVERL_ROOT}" || return 1
    rsync -az --info=progress2 \
      -e "ssh ${SSH_OPTS[*]}" \
      "${RSYNC_EXCLUDES[@]}" \
      "${VAMVERL_ROOT}/" "${REMOTE_USER}@192.168.88.${ip}:${VAMVERL_ROOT}/" || return 1
  fi

  if [[ "${DO_INSTALL}" == "1" ]]; then
    echo ">>> pip install -e @ 192.168.88.${ip}"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" bash -s <<REMOTE
set -euo pipefail
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
"${REMOTE_ENV}/bin/python" -m pip install '${NUMPY_PIN}' -q
"${REMOTE_ENV}/bin/python" -m pip install \
  'albumentations>=1.4' 'einops>=0.8' tyro 'tianshou>=0.5,<3' ftfy \
  'imageio>=2.34' imageio-ffmpeg 'diffusers>=0.30' 'peft>=0.5' decord2 'av>=15' -q
"${REMOTE_ENV}/bin/python" -m pip install -e "${VAMVERL_ROOT}" --no-deps -q
REMOTE
  else
    echo ">>> import 校验 @ 192.168.88.${ip}（跳过 pip install，需时用 --install）"
  fi
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" bash -s <<REMOTE
set -euo pipefail
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
PYTHONPATH="${VAMVERL_ROOT}:\${PYTHONPATH:-}" "${REMOTE_ENV}/bin/python" -c "${VERIFY_PY}"
REMOTE
}

cmd_sync() {
  local failed=0 i ip host
  echo ">>> 同步策略: 代码 rsync；data/ 不传输（先 bash scripts/mount_nfs_cluster4.sh vamverl）"
  if [[ "${DO_INSTALL}" == "1" ]]; then
    echo ">>> pip install: 开启（--install / SYNC_INSTALL=1）"
  else
    echo ">>> pip install: 跳过（依赖/pyproject 变更时用 --install）"
  fi
  du -sh "${VAMVERL_ROOT}" 2>/dev/null | awk '{print "  本机仓库:", $1, "（rsync 排除 data/）"}'
  for i in "${!WORKERS[@]}"; do
    ip="${WORKERS[$i]}"
    host="${WORKER_HOSTS[$i]:-worker}"
    if ! _sync_one "${ip}" "${host}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo ">>> 部分节点失败: WORKER_IPS=\"21 11\" bash $0" >&2
    exit 1
  fi
  echo ">>> 四机 VamVerl 源码已同步"
}

case "${1:-sync}" in
  sync|""|start)
    shift || true
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --install)
          DO_INSTALL=1
          shift
          ;;
        *)
          echo "未知参数: $1" >&2
          exit 2
          ;;
      esac
    done
    cmd_sync
    ;;
  status)
    cmd_status
    ;;
  -h|--help|help)
    sed -n '2,10p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "用法: $0 [sync|status] [--install]" >&2
    exit 1
    ;;
esac
