#!/usr/bin/env bash
# 四机 Ray 集群 · 仅负责 Ray 启停，不含 NFS 同步/训练
#
# 前置（41 上依次）:
#   bash scripts/mount_nfs_cluster4.sh all
#   bash scripts/sync_vamverl_cluster.sh
#
# 用法（41 上）:
#   bash scripts/ray/start_ray_head.sh              # 起四机 Ray
#   bash scripts/ray/start_ray_head.sh stop       # 停四机 Ray
#   bash scripts/ray/start_ray_head.sh status
#   bash scripts/ray/start_ray_head.sh block       # 仅 head 前台
set -euo pipefail

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
HEAD_IP="${HEAD_IP:-192.168.88.41}"
HEAD_HOST="${HEAD_HOST:-spark-0a0b}"
WORKERS=(31 21 11)
WORKER_HOSTS=(gx10-400b gx10-4df7 gx10-39b7)
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
VAMVERL_ROOT="${VAMVERL_ROOT:-${HOME}/WAM/VamVerl}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_OBJECT_STORE_BYTES="${RAY_OBJECT_STORE_BYTES:-8589934592}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30 -o "BindAddress=${SRC_BIND}")
# stop 用更短超时，避免 .21 hung 时整脚本卡死
SSH_STOP_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=4 -o ServerAliveCountMax=2 -o "BindAddress=${SRC_BIND}")
STOP_SSH_TIMEOUT="${STOP_SSH_TIMEOUT:-25}"
LOG_DIR="${HOME}/cluster-4spark-setup"
HEAD_LOG="${LOG_DIR}/vampo-ray-head-nohup.log"
WORKER_SCRIPT="${VAMVERL_ROOT}/scripts/ray/start_ray_worker.sh"

require_head_41() {
  local hn
  hn="$(hostname)"
  [[ "${hn}" == "${HEAD_HOST}" ]] || hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${HEAD_IP}" || {
    echo "ERROR: 本脚本须在 41 (${HEAD_HOST}, ${HEAD_IP}) 执行，当前: ${hn}"
    exit 1
  }
}

setup_local_env() {
  [[ -f "${CONDA_SH}" ]] || { echo "ERROR: 无 conda: ${CONDA_SH}"; exit 1; }
  [[ -f "${HOME}/.spark-cluster-env" ]] || {
    echo "ERROR: 缺少 ~/.spark-cluster-env"
    exit 1
  }
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
  # shellcheck source=/dev/null
  source "${HOME}/.spark-cluster-env"
  # shellcheck source=/dev/null
  [[ -f "${HOME}/cluster-4spark-setup/spark-cluster-source-nccl.sh" ]] && \
    source "${HOME}/cluster-4spark-setup/spark-cluster-source-nccl.sh"

  export VLLM_HOST_IP="${VLLM_HOST_IP:-${SPARK_CLUSTER_IP}}"
  export MASTER_ADDR="${MASTER_ADDR:-${SPARK_CLUSTER_IP}}"
  export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
  export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
  export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-20}"
  export RAY_health_check_period_ms="${RAY_health_check_period_ms:-5000}"
  export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-60000}"

  RAY_BIN="${CONDA_PREFIX}/bin/ray"
  ENV_PYTHON="${CONDA_PREFIX}/bin/python"
  [[ -x "${ENV_PYTHON}" ]] || { echo "ERROR: 无 ${ENV_PYTHON}"; exit 1; }
  if [[ -f "${RAY_BIN}" ]] && head -1 "${RAY_BIN}" | grep -qv "${CONDA_PREFIX}/bin/python"; then
    sed -i "1s|.*|#!${ENV_PYTHON}|" "${RAY_BIN}"
    chmod +x "${RAY_BIN}"
  fi
  [[ -n "${VLLM_HOST_IP}" ]] || {
    echo "ERROR: SPARK_CLUSTER_IP 未设置"
    exit 1
  }
}

ray_local() {
  "${ENV_PYTHON}" "${RAY_BIN}" "$@"
}

_ray_stop_remote() {
  local ip="$1"
  timeout "${STOP_SSH_TIMEOUT}" ssh "${SSH_STOP_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" bash -s 2>/dev/null <<'REMOTE'
set +e
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
[[ -f "${CONDA_SH}" ]] && source "${CONDA_SH}" && conda activate "${ENV_NAME}" 2>/dev/null
PY="${HOME}/anaconda3/envs/${ENV_NAME}/bin/python"
RAY="${HOME}/anaconda3/envs/${ENV_NAME}/bin/ray"
if [[ -x "${PY}" && -x "${RAY}" ]]; then
  "${PY}" "${RAY}" stop --force >/dev/null 2>&1
fi
pkill -TERM -f 'raylet/raylet' 2>/dev/null
pkill -TERM -f '/ray/_private/workers/default_worker' 2>/dev/null
sleep 1
pkill -KILL -f 'raylet/raylet' 2>/dev/null
pkill -KILL -f '/ray/_private/workers/default_worker' 2>/dev/null
exit 0
REMOTE
}

_stop_worker_node() {
  local ip="$1" host="$2"
  local rc
  _ray_stop_remote "${ip}"
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    echo "stopped"
    return 0
  fi
  if [[ "${rc}" -eq 124 ]]; then
    echo "FAIL SSH/ray stop 超时 (${STOP_SSH_TIMEOUT}s)"
  elif [[ "${rc}" -eq 255 ]]; then
    echo "FAIL SSH 不可达（banner exchange 超时，节点可能 hung/OOM）"
  else
    echo "FAIL SSH 退出码 ${rc}"
  fi
  return 1
}

_stop_head_local() {
  ray_local stop --force >/dev/null 2>&1 || true
  pkill -TERM -f 'raylet/raylet' 2>/dev/null || true
  pkill -TERM -f 'gcs_server' 2>/dev/null || true
  sleep 1
  pkill -KILL -f 'raylet/raylet' 2>/dev/null || true
  pkill -KILL -f 'gcs_server' 2>/dev/null || true
}

stop_all() {
  setup_local_env
  local fail=0 i ip host tmpdir
  tmpdir="$(mktemp -d "${LOG_DIR}/ray-stop.XXXXXX")"
  echo ">>> 停止四机 Ray（worker 并行 SSH 超时 ${STOP_SSH_TIMEOUT}s）"
  for i in "${!WORKERS[@]}"; do
    ip="${WORKERS[$i]}"
    host="${WORKER_HOSTS[$i]}"
    (
      if _stop_worker_node "${ip}" "${host}" >"${tmpdir}/${ip}.out" 2>&1; then
        echo ok >"${tmpdir}/${ip}.rc"
      else
        echo fail >"${tmpdir}/${ip}.rc"
      fi
    ) &
  done
  wait
  for i in "${!WORKERS[@]}"; do
    ip="${WORKERS[$i]}"
    host="${WORKER_HOSTS[$i]}"
    echo -n ">>> .${ip} (${host}): "
    if [[ -f "${tmpdir}/${ip}.out" ]]; then
      cat "${tmpdir}/${ip}.out"
    else
      echo "FAIL 无结果"
    fi
    if [[ ! -f "${tmpdir}/${ip}.rc" ]] || [[ "$(cat "${tmpdir}/${ip}.rc")" != ok ]]; then
      fail=1
    fi
  done
  rm -rf "${tmpdir}"
  echo -n ">>> .41 (head): "
  _stop_head_local
  echo "stopped"
  if [[ "${fail}" -ne 0 ]]; then
    echo ">>> 部分 worker 未停（.21 需 console/物理重启后再 bash $0 stop）" >&2
    return 1
  fi
}

wait_for_head() {
  local i
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  for i in $(seq 1 30); do
    if ray_local status >/dev/null 2>&1; then
      echo ">>> Head 就绪 (${i}s)"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Head 未在 30s 内就绪，查看 ${HEAD_LOG}" >&2
  tail -20 "${HEAD_LOG}" 2>/dev/null || true
  exit 1
}

cmd_status() {
  require_head_41
  setup_local_env
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  if ray_local status 2>/dev/null; then
    echo ""
    echo "期望: 4 节点 / 4 GPU"
  else
    echo ">>> Ray 未运行（bash $0 启动）"
    exit 1
  fi
}

cmd_block_head_only() {
  require_head_41
  setup_local_env
  echo "=== Ray Head 前台 (${ENV_NAME}) — 仅 41 ==="
  echo "  请在 31/21/11 各开终端: bash ${WORKER_SCRIPT}"
  echo ""
  ray_local stop --force 2>/dev/null || true
  exec "${ENV_PYTHON}" "${RAY_BIN}" start --head --block \
    --node-ip-address="${VLLM_HOST_IP}" \
    --port="${RAY_PORT}" \
    --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
    --dashboard-host=0.0.0.0 \
    --num-gpus=1 \
    --disable-usage-stats
}

cmd_cluster() {
  require_head_41
  setup_local_env
  mkdir -p "${LOG_DIR}"

  [[ -x "${WORKER_SCRIPT}" ]] || {
    echo "ERROR: 缺少 ${WORKER_SCRIPT}（先 bash scripts/mount_nfs_cluster4.sh vamverl）" >&2
    exit 1
  }

  echo "=== VAMPO 四机 Ray 启动 ==="
  echo "  Head:  ${VLLM_HOST_IP}:${RAY_PORT}"
  echo "  Workers: ${WORKERS[*]}"
  echo "  前置: mount_nfs_cluster4.sh + sync_vamverl_cluster.sh"
  echo ""

  stop_all || true
  sleep 2

  echo ">>> 启动 Head（后台）→ ${HEAD_LOG}"
  nohup "${ENV_PYTHON}" "${RAY_BIN}" start --head \
    --node-ip-address="${VLLM_HOST_IP}" \
    --port="${RAY_PORT}" \
    --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
    --dashboard-host=0.0.0.0 \
    --num-gpus=1 \
    --disable-usage-stats \
    > "${HEAD_LOG}" 2>&1 &
  wait_for_head

  local i ip host fail=0
  for i in "${!WORKERS[@]}"; do
    ip="${WORKERS[$i]}"
    host="${WORKER_HOSTS[$i]}"
    echo ">>> SSH 启动 Worker 192.168.88.${ip} (${host})"
    if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" \
      "HEAD_IP=${HEAD_IP} RAY_BLOCK=0 bash ${WORKER_SCRIPT}"; then
      echo "ERROR: worker ${ip} 启动失败" >&2
      fail=1
    fi
  done
  [[ "${fail}" -eq 0 ]] || exit 1
  sleep 12

  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  ray_local status | head -25
  echo ""
  echo ">>> 四机 Ray 就绪"
  echo "  Dashboard: http://${HEAD_IP}:8265"
  echo "  开训: bash ${VAMVERL_ROOT}/scripts/train_component3_rl_cluster4.sh"
  echo "  日志: ${HEAD_LOG}  及各机 ~/cluster-4spark-setup/vampo-ray-worker-nohup.log"
}

usage() {
  cat <<EOF
用法（41 上）:
  bash scripts/ray/start_ray_head.sh           起四机 Ray
  bash scripts/ray/start_ray_head.sh stop      停四机 Ray
  bash scripts/ray/start_ray_head.sh status
  bash scripts/ray/start_ray_head.sh block     仅 head 前台

前置: bash scripts/mount_nfs_cluster4.sh all && bash scripts/sync_vamverl_cluster.sh
EOF
}

ACTION="${1:-cluster}"
case "${ACTION}" in
  cluster|start|"")
    cmd_cluster
    ;;
  block)
    cmd_block_head_only
    ;;
  status)
    cmd_status
    ;;
  stop)
    require_head_41
    if stop_all; then
      echo ">>> 已停止四机 Ray"
    else
      echo ">>> Ray stop 部分失败（见上）" >&2
      exit 1
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "未知: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
