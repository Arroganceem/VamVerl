#!/usr/bin/env bash
# Component 2 · 四机 VideoMAE 训练 · 仅负责训练启停
#
# 前置: bash scripts/mount_nfs_cluster4.sh clips
#
# 用法:
#   bash scripts/train_component2_reward_cluster4.sh          # 开训
#   bash scripts/train_component2_reward_cluster4.sh stop   # 四机停训
#   bash scripts/train_component2_reward_cluster4.sh status
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
MASTER_ADDR="${MASTER_ADDR:-192.168.88.41}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES=4
NGPU=1
NODE_RANK=0
CONFIG="${CONFIG:-$ROOT/configs/videomae_droid.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
WORKERS=(31:1 21:2 11:3)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o "BindAddress=${SRC_BIND}")
COOLDOWN_EVERY_STEPS="${COOLDOWN_EVERY_STEPS:-30}"
COOLDOWN_SEC="${COOLDOWN_SEC:-50}"
export VAMPO_COOLDOWN_EVERY_STEPS="${COOLDOWN_EVERY_STEPS}"
export VAMPO_COOLDOWN_SEC="${COOLDOWN_SEC}"

# 匹配 torch launcher + 训练模块
_TRAIN_PATTERNS=(
  'vampo\.reward\.train_videomae'
  'torch\.distributed\.run.*train_videomae'
)

_kill_local() {
  local pat
  for pat in "${_TRAIN_PATTERNS[@]}"; do
    pkill -TERM -f "${pat}" 2>/dev/null || true
  done
  sleep 2
  for pat in "${_TRAIN_PATTERNS[@]}"; do
    pkill -KILL -f "${pat}" 2>/dev/null || true
  done
  # 本机 SSH 拉起的 worker 会话
  pkill -TERM -f "ssh.*192\.168\.88\.(31|21|11).*bash -s" 2>/dev/null || true
}

_kill_remote_script() {
  cat <<'REMOTE'
set +e
PATTERNS=(
  'vampo.reward.train_videomae'
  'torch.distributed.run.*train_videomae'
)
for pat in "${PATTERNS[@]}"; do
  pkill -TERM -f "${pat}" 2>/dev/null || true
done
sleep 2
for pat in "${PATTERNS[@]}"; do
  pkill -KILL -f "${pat}" 2>/dev/null || true
done
REMOTE
}

_count_on_host() {
  local where="$1"
  local cmd="pgrep -fc 'vampo.reward.train_videomae|torch.distributed.run.*train_videomae' 2>/dev/null || echo 0"
  local raw
  if [[ "${where}" == "local" ]]; then
    raw=$(bash -c "${cmd}" 2>/dev/null || echo 0)
  else
    raw=$(ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${where}" "${cmd}" 2>/dev/null || echo -1)
  fi
  if [[ "${raw}" == "-1" ]]; then
    echo -1
    return
  fi
  printf '%s' "${raw}" | head -1 | tr -cd '0-9'
}

_stop_node() {
  local label="$1"
  local host="$2"
  if [[ "${host}" == "local" ]]; then
    _kill_local
    echo "  OK  ${label}"
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${host}" "bash -s" < <(_kill_remote_script); then
    echo "  FAIL ${label}: SSH 不可达，训练进程可能仍在 GPU 上运行" >&2
    return 1
  fi
  echo "  OK  ${label}"
}

cmd_stop() {
  local fail=0 spec ip rank
  echo ">>> 停止四机 VideoMAE 训练"
  _stop_node "rank0 @ ${MASTER_ADDR}" local || fail=1
  for spec in "${WORKERS[@]}"; do
    ip="${spec%%:*}"
    rank="${spec##*:}"
    _stop_node "rank${rank} @ 192.168.88.${ip}" "${ip}" || fail=1
  done
  sleep 1
  cmd_status
  [[ "${fail}" -eq 0 ]] || {
    echo ">>> 部分节点未停干净；.21 若 SSH 超时需 console 重启后重试 stop" >&2
    exit 1
  }
}

cmd_status() {
  local n0 n1 n2 n3 total
  n0=$(_count_on_host local); [[ -n "${n0}" ]] || n0=0
  n1=$(_count_on_host 31); [[ -n "${n1}" ]] || n1=0
  n2=$(_count_on_host 21); [[ -n "${n2}" ]] || n2=0
  n3=$(_count_on_host 11); [[ -n "${n3}" ]] || n3=0
  echo ">>> VideoMAE 训练进程: .41=${n0} .31=${n1} .21=${n2} .11=${n3}"
  if [[ "${n2}" == "-1" ]]; then echo "    .21: SSH 不可达" >&2; fi
  if [[ "${n1}" == "-1" ]]; then echo "    .31: SSH 不可达" >&2; fi
  if [[ "${n3}" == "-1" ]]; then echo "    .11: SSH 不可达" >&2; fi
  total=$((n0 + (n1 >= 0 ? n1 : 0) + (n2 >= 0 ? n2 : 0) + (n3 >= 0 ? n3 : 0)))
  echo ">>> total=${total} (期望 0 或 4)"
}

_remote_cmd() {
  local rank="$1"
  cat <<EOF
set -euo pipefail
source '${CONDA_SH}'
conda activate '${ENV_NAME}'
cd '${ROOT}'
export PYTHONPATH='${ROOT}:\${PYTHONPATH:-}'
export VAMVERL_ROOT='${ROOT}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT='${MASTER_PORT}'
export DROID_DATA_ROOT='${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}'
export DROID_SPLIT_DIR='${DROID_SPLIT_DIR:-$ROOT/data/splits}'
export VIDEOMAE_CLIP_DIR='${VIDEOMAE_CLIP_DIR:-$ROOT/data/videomae_droid_clips}'
export VIDEOMAE_BACKBONE='${VIDEOMAE_BACKBONE:-/home/robotem/Models/videomae-base}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE:-1}'
export CONFIG='${CONFIG}'
export NNODES='${NNODES}'
export NGPU='${NGPU}'
export NODE_RANK='${rank}'
export VAMPO_COOLDOWN_EVERY_STEPS='${COOLDOWN_EVERY_STEPS}'
export VAMPO_COOLDOWN_SEC='${COOLDOWN_SEC}'
exec python -m torch.distributed.run \\
  --nnodes='${NNODES}' \\
  --nproc_per_node='${NGPU}' \\
  --node_rank='${rank}' \\
  --master_addr='${MASTER_ADDR}' \\
  --master_port='${MASTER_PORT}' \\
  -m vampo.reward.train_videomae --config '${CONFIG}'
EOF
}

cmd_launch() {
  local stamp log_prefix spec ip rank log
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_prefix="${LOG_DIR}/train_videomae_cluster4_${stamp}"

  if ! bash "${ROOT}/scripts/preflight_component2_train.sh"; then
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  cd "$ROOT"

  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  export VAMVERL_ROOT="$ROOT"
  export DROID_SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
  export VIDEOMAE_CLIP_DIR="${VIDEOMAE_CLIP_DIR:-$ROOT/data/videomae_droid_clips}"
  export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}"
  export VIDEOMAE_BACKBONE="${VIDEOMAE_BACKBONE:-/home/robotem/Models/videomae-base}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

  echo ">>> Component2 四机 VideoMAE · master=${MASTER_ADDR}:${MASTER_PORT}"
  echo ">>> config=${CONFIG}"
  echo ">>> cooldown: every ${COOLDOWN_EVERY_STEPS} steps, sleep ${COOLDOWN_SEC}s"
  echo ">>> logs: ${log_prefix}_rank*.log"
  echo ">>> 停训: bash $0 stop"

  for spec in "${WORKERS[@]}"; do
    ip="${spec%%:*}"
    rank="${spec##*:}"
    log="${log_prefix}_rank${rank}.log"
    echo ">>> 启动 worker rank=${rank} @ 192.168.88.${ip} → ${log}"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" "bash -s" >"${log}" 2>&1 < <(_remote_cmd "${rank}") &
    sleep 2
  done

  echo ">>> 启动 rank0 @ ${MASTER_ADDR}（前台；Ctrl+C 后请 bash $0 stop）"
  echo ">>> tail -f ${log_prefix}_rank*.log"

  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}"
  fi
  local env_python="${CONDA_PREFIX:-}/bin/python"
  [[ -x "${env_python}" ]] || env_python="python"

  exec "${env_python}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NGPU}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m vampo.reward.train_videomae --config "${CONFIG}"
}

case "${1:-launch}" in
  launch|start|"")
    cmd_launch
    ;;
  stop)
    cmd_stop
    ;;
  status)
    cmd_status
    ;;
  -h|--help|help)
    sed -n '2,9p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "用法: $0 [launch|stop|status]" >&2
    exit 1
    ;;
esac
