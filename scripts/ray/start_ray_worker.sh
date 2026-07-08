#!/usr/bin/env bash
# Ray Worker（31 / 21 / 11）· 由 41 SSH 调用，或在该机手工执行
#
# 通常不需要手工跑：在 41 执行 start_ray_head.sh 会自动 SSH 到本机。
# 手工调试（SSH 登录 worker 后）:
#   bash ~/WAM/VamVerl/scripts/ray/start_ray_worker.sh
# 后台 daemon（41 SSH 调用）:
#   RAY_BLOCK=0 bash scripts/ray/start_ray_worker.sh
set -euo pipefail

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
HEAD_IP="${HEAD_IP:-192.168.88.41}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_OBJECT_STORE_BYTES="${RAY_OBJECT_STORE_BYTES:-8589934592}"
RAY_BLOCK="${RAY_BLOCK:-1}"
LOG_DIR="${HOME}/cluster-4spark-setup"
WORKER_LOG="${LOG_DIR}/vampo-ray-worker-nohup.log"

[[ -f "${CONDA_SH}" ]] || { echo "ERROR: 无 conda: ${CONDA_SH}"; exit 1; }
[[ -f "${HOME}/.spark-cluster-env" ]] || {
  echo "ERROR: 缺少 ~/.spark-cluster-env"
  exit 1
}

cd "${HOME}"
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
# shellcheck source=/dev/null
source "${HOME}/.spark-cluster-env"
# shellcheck source=/dev/null
[[ -f "${HOME}/cluster-4spark-setup/spark-cluster-source-nccl.sh" ]] && \
  source "${HOME}/cluster-4spark-setup/spark-cluster-source-nccl.sh"

export VLLM_HOST_IP="${VLLM_HOST_IP:-${SPARK_CLUSTER_IP}}"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"
export RAY_ADDRESS="${RAY_ADDRESS:-${HEAD_IP}:${RAY_PORT}}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-20}"
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-5000}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-60000}"

[[ -n "${VLLM_HOST_IP}" ]] || {
  echo "ERROR: SPARK_CLUSTER_IP 未设置"
  exit 1
}
if [[ "${VLLM_HOST_IP}" == "${HEAD_IP}" ]]; then
  echo "ERROR: 这是 worker 脚本，不能在 head (${HEAD_IP}) 上运行"
  echo "  在 41 请用: bash scripts/ray/start_ray_head.sh"
  exit 1
fi

RAY_BIN="${CONDA_PREFIX}/bin/ray"
ENV_PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || { echo "ERROR: 无 ${ENV_PYTHON}"; exit 1; }
if [[ -f "${RAY_BIN}" ]] && head -1 "${RAY_BIN}" | grep -qv "${CONDA_PREFIX}/bin/python"; then
  sed -i "1s|.*|#!${ENV_PYTHON}|" "${RAY_BIN}"
  chmod +x "${RAY_BIN}"
fi

echo "=== Ray Worker (${ENV_NAME}) — $(hostname) ==="
echo "  本机光口: ${VLLM_HOST_IP}"
echo "  Head: ${HEAD_IP}:${RAY_PORT}"
echo "  模式: $([[ "${RAY_BLOCK}" == "1" ]] && echo '前台 --block' || echo '后台 nohup')"
echo ""

"${ENV_PYTHON}" "${RAY_BIN}" stop --force 2>/dev/null || true
mkdir -p "${LOG_DIR}"

if [[ "${RAY_BLOCK}" == "1" ]]; then
  exec "${ENV_PYTHON}" "${RAY_BIN}" start --block \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --node-ip-address="${VLLM_HOST_IP}" \
    --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
    --num-gpus=1 \
    --disable-usage-stats
fi

nohup "${ENV_PYTHON}" "${RAY_BIN}" start \
  --address="${HEAD_IP}:${RAY_PORT}" \
  --node-ip-address="${VLLM_HOST_IP}" \
  --object-store-memory="${RAY_OBJECT_STORE_BYTES}" \
  --num-gpus=1 \
  --disable-usage-stats \
  > "${WORKER_LOG}" 2>&1 &

echo "Worker 已后台加入 ${HEAD_IP}:${RAY_PORT}，日志: ${WORKER_LOG}"
