#!/usr/bin/env bash
# 四机 DreamZero → FSDP per-rank checkpoint（一次性，NFS 共享 output）
#
# rank0 = .31（与训练 FSDP rank0 一致，负责一次性全量 load）
#
# 用法（在 .41 head 上）:
#   bash scripts/checkpoint/convert_dreamzero_fsdp_cluster4.sh
#   MODEL_PATH=... OUTPUT=... bash scripts/checkpoint/convert_dreamzero_fsdp_cluster4.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30 -o "BindAddress=${SRC_BIND}")

MODEL_PATH="${MODEL_PATH:-/home/robotem/Models/DreamZero-DROID}"
OUTPUT="${OUTPUT:-/home/robotem/Models/DreamZero-DROID-fsdp4}"
MASTER_ADDR="${MASTER_ADDR:-192.168.88.31}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES=4
NGPU=1
# rank0=.31, rank1=.41, rank2=.21, rank3=.11
WORKERS=(41:1 21:2 11:3)
RANK0_IP=31
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

_remote_cmd() {
  local node_rank="$1"
  cat <<EOF
set -euo pipefail
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:\${PYTHONPATH:-}"
export VAMVERL_ROOT="${ROOT}"
unset VAMPO_FSDP_SHARDED_CHECKPOINT
PY="\${CONDA_PREFIX:-}/bin/python"
[[ -x "\${PY}" ]] || PY=python
exec "\${PY}" -m torch.distributed.run \\
  --nnodes=${NNODES} \\
  --nproc_per_node=${NGPU} \\
  --node_rank=${node_rank} \\
  --master_addr=${MASTER_ADDR} \\
  --master_port=${MASTER_PORT} \\
  scripts/checkpoint/convert_dreamzero_fsdp_checkpoint.py \\
  --model-path "${MODEL_PATH}" \\
  --output "${OUTPUT}"
EOF
}

# 本机执行，避免 ssh 到自己（Permission denied）
_run_on_host() {
  local ip="$1"
  local node_rank="$2"
  local log="$3"
  local local_ip="${SRC_BIND##*.}"
  if [[ "${ip}" == "${local_ip}" ]]; then
    _remote_cmd "${node_rank}" | bash -s >"${log}" 2>&1
  else
    _remote_cmd "${node_rank}" | ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" "bash -s" >"${log}" 2>&1
  fi
}

main() {
  local stamp log_prefix spec ip rank log

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_prefix="${LOG_DIR}/convert_fsdp4_${stamp}"
  mkdir -p "${LOG_DIR}"

  echo "[convert] MODEL_PATH=${MODEL_PATH}"
  echo "[convert] OUTPUT=${OUTPUT}"
  echo "[convert] master=${MASTER_ADDR}:${MASTER_PORT} rank0=.${RANK0_IP}"
  echo "[convert] logs: ${log_prefix}_rank*.log"

  for spec in "${WORKERS[@]}"; do
    ip="${spec%%:*}"
    rank="${spec##*:}"
    log="${log_prefix}_rank${rank}.log"
    echo "[convert] start rank=${rank} @ 192.168.88.${ip} → ${log}"
    _run_on_host "${ip}" "${rank}" "${log}" &
    sleep 2
  done

  log="${log_prefix}_rank0.log"
  echo "[convert] start rank=0 @ 192.168.88.${RANK0_IP}（前台）"
  echo "[convert] tail -f ${log_prefix}_rank*.log"

  _remote_cmd 0 | ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${RANK0_IP}" "bash -s" 2>&1 | tee "${log}"

  echo "[convert] done. Training yaml: sharded_checkpoint_dir: ${OUTPUT}"
}

main "$@"
