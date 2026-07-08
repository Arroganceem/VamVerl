#!/usr/bin/env bash
# Component 3 · 四机 verl GRPO + PPO · 仅负责训练启停
#
# 前置:
#   bash scripts/mount_nfs_cluster4.sh all
#   bash scripts/sync_vamverl_cluster.sh
#   bash scripts/ray/start_ray_head.sh
#
# 用法:
#   bash scripts/train_component3_rl_cluster4.sh              # 开训
#   bash scripts/train_component3_rl_cluster4.sh stop         # 停四机训练进程（不关 Ray）
#   bash scripts/train_component3_rl_cluster4.sh status
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
HEAD_IP="${HEAD_IP:-192.168.88.41}"
WORKERS=(31 21 11)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o "BindAddress=${SRC_BIND}")

# 仅训练相关进程；刻意不含 raylet / gcs / dashboard（不关 Ray 集群）
_TRAIN_PATTERNS=(
  'vampo\.integrations\.verl\.main_vampo_ppo'
  'main_vampo_ppo'
  'train_component3_rl_cluster4'
  'VAMPOActorRolloutRefWorker'
  'ray::VAMPOActorRolloutRefWorker'
  'ray::main_task'
)
_TRAIN_PGREP='main_vampo_ppo|VAMPOActorRolloutRefWorker|ray::VAMPOActorRolloutRefWorker|ray::main_task'

_setup_python() {
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}"
  fi
  ENV_PYTHON="${CONDA_PREFIX:-}/bin/python"
  [[ -x "${ENV_PYTHON}" ]] || ENV_PYTHON="python"
}

_pkill_train() {
  local pat
  for pat in "${_TRAIN_PATTERNS[@]}"; do
    pkill -TERM -f "${pat}" 2>/dev/null || true
  done
  sleep 2
  for pat in "${_TRAIN_PATTERNS[@]}"; do
    pkill -KILL -f "${pat}" 2>/dev/null || true
  done
}

_kill_local() {
  _pkill_train
}

_ray_kill_training_actors() {
  _setup_python
  RAY_ADDRESS="${RAY_ADDRESS:-auto}" "${ENV_PYTHON}" - <<'PY' || true
import os
import sys

try:
    import ray
except ImportError:
    sys.exit(0)

addr = os.environ.get("RAY_ADDRESS", "auto")
try:
    ray.init(address=addr, ignore_reinit_error=True)
except Exception as e:
    print(f"[stop] Ray API 跳过: {e}")
    sys.exit(0)

markers = ("VAMPOActorRolloutRefWorker", "main_task")
killed = 0
actors = []
try:
    from ray.util.state import list_actors
    actors = list_actors(detail=True, filters=[("state", "=", "ALIVE")]) or []
except Exception:
    pass

for a in actors:
    cls = a.get("class_name") or ""
    name = a.get("name") or ""
    if not any(m in cls or m in name for m in markers):
        continue
    try:
        actor = ray.get_actor(name, namespace=a.get("ray_namespace") or None)
        ray.kill(actor, no_restart=True)
        killed += 1
        print(f"[stop] ray.kill {cls or name}")
    except Exception as ex:
        print(f"[stop] ray.kill skip {cls or name}: {ex}")

ray.shutdown()
print(f"[stop] Ray API 已终止 {killed} 个训练 actor（Ray 集群保持运行）")
PY
}

_kill_remote_script() {
  cat <<'REMOTE'
set +e
PATTERNS=(
  'vampo.integrations.verl.main_vampo_ppo'
  'main_vampo_ppo'
  'train_component3_rl_cluster4'
  'VAMPOActorRolloutRefWorker'
  'ray::VAMPOActorRolloutRefWorker'
  'ray::main_task'
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

_stop_one_node() {
  local label="$1"
  local host="$2"
  if [[ "${host}" == "local" ]]; then
    _kill_local
    echo "  OK  ${label}"
    return 0
  fi
  if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${host}" "bash -s" < <(_kill_remote_script); then
    echo "  FAIL ${label}: SSH 不可达" >&2
    return 1
  fi
  echo "  OK  ${label}"
}

_count_on_host() {
  local where="$1"
  local cmd="pgrep -fc '${_TRAIN_PGREP}' 2>/dev/null || echo 0"
  local raw
  if [[ "${where}" == "local" ]]; then
    raw=$(bash -c "${cmd}" 2>/dev/null || echo 0)
  else
    raw=$(ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${where}" "${cmd}" 2>/dev/null || echo -1)
  fi
  [[ "${raw}" == "-1" ]] && { echo -1; return; }
  printf '%s' "${raw}" | head -1 | tr -cd '0-9'
}

cmd_stop() {
  local fail=0 ip
  echo ">>> 停止 Component3 RL 训练（四机，不关 Ray）"
  echo ">>> 1/2 Ray API 终止训练 actor..."
  _ray_kill_training_actors
  echo ">>> 2/2 pkill 四机训练进程..."
  _stop_one_node "driver/worker @ ${HEAD_IP}" local || fail=1
  for ip in "${WORKERS[@]}"; do
    _stop_one_node "worker @ 192.168.88.${ip}" "${ip}" || fail=1
  done
  sleep 1
  cmd_status
  _setup_python
  if RAY_ADDRESS="${RAY_ADDRESS:-auto}" "${ENV_PYTHON}" -c "import ray; ray.init(address='auto'); ray.shutdown()" 2>/dev/null; then
    echo ">>> Ray 集群仍在运行（停 Ray: bash scripts/ray/start_ray_head.sh stop）"
  fi
  [[ "${fail}" -eq 0 ]] || {
    echo ">>> 部分节点 SSH 失败，.21 若超时需 console 重启后重试 stop" >&2
    exit 1
  }
}

cmd_status() {
  local n0 n1 n2 n3
  n0=$(_count_on_host local); [[ -n "${n0}" ]] || n0=0
  n1=$(_count_on_host 31); [[ -n "${n1}" ]] || n1=0
  n2=$(_count_on_host 21); [[ -n "${n2}" ]] || n2=0
  n3=$(_count_on_host 11); [[ -n "${n3}" ]] || n3=0
  echo ">>> RL 训练进程: .41=${n0} .31=${n1} .21=${n2} .11=${n3} (期望全 0)"
  [[ "${n2}" == "-1" ]] && echo "    .21: SSH 不可达" >&2
  [[ "${n1}" == "-1" ]] && echo "    .31: SSH 不可达" >&2
  [[ "${n3}" == "-1" ]] && echo "    .11: SSH 不可达" >&2
}

cmd_launch() {
  _setup_python

  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  export VAMVERL_ROOT="$ROOT"
  export MODEL_PATH="${MODEL_PATH:-/home/robotem/Models/DreamZero-DROID}"
  export WAN21_DIR="${WAN21_DIR:-/home/robotem/Models/Wan2.1-I2V-14B-480P}"
  export WAN22_DIR="${WAN22_DIR:-/home/robotem/Models/Wan2.2-TI2V-5B}"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/robotem/Models/umt5-xxl}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}"
  export INIT_STATES_DIR="${INIT_STATES_DIR:-$ROOT/data/init_states}"
  export DROID_SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
  export MAX_INIT_STATES="${MAX_INIT_STATES:-2888}"
  export VIDEOMAE_BACKBONE="${VIDEOMAE_BACKBONE:-/home/robotem/Models/videomae-base}"
  export VIDEOMAE_CKPT="${VIDEOMAE_CKPT:-$ROOT/checkpoints/videomae_droid.pth}"
  export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
  export HEAD_HOST="${HEAD_HOST:-spark-0a0b}"
  export HEAD_IP="${HEAD_IP:-${RAY_HEAD_IP:-192.168.88.41}}"
  export RAY_HEAD_IP="${RAY_HEAD_IP:-${HEAD_IP}}"
  export VAMPO_RANK0_NODE_IP="${VAMPO_RANK0_NODE_IP:-192.168.88.31}"
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
  export WANDB_PROJECT="${WANDB_PROJECT:-vamverl}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export VAMPO_NCCL_TIMEOUT_MIN="${VAMPO_NCCL_TIMEOUT_MIN:-120}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  cd "$ROOT"

  echo ">>> Component3 RL · 停训: bash $0 stop"

  "${ENV_PYTHON}" "$ROOT/scripts/ensure_libero_config.py"

  if [[ "${RAY_ADDRESS}" == "auto" ]]; then
    "${ENV_PYTHON}" - <<'PY' || { echo "请先: bash scripts/ray/start_ray_head.sh" >&2; exit 1; }
import sys
try:
    import ray
    ray.init(address="auto")
    gpus = int(ray.cluster_resources().get("GPU", 0))
    nodes = len(ray.nodes())
    ray.shutdown()
    print(f"[cluster] nodes={nodes} gpus={gpus}")
    if gpus < 4:
        sys.exit(1)
except Exception as e:
    print(f"[cluster] Ray 未就绪: {e}", file=sys.stderr)
    sys.exit(1)
PY
  fi

  if [[ "${PREFLIGHT_SKIP_REWARD:-0}" == "1" ]]; then
    bash scripts/preflight_rl.sh --skip-reward
  else
    bash scripts/preflight_rl.sh --strict-reward
  fi

  # shellcheck source=/dev/null
  source "$ROOT/scripts/bootstrap_init_states.sh"
  ensure_init_states

  "${ENV_PYTHON}" - <<'PY'
print("[preflight] VAMPO verl FSDP distributed path")
PY

  exec "${ENV_PYTHON}" -m vampo.integrations.verl.main_vampo_ppo \
    --config-name "${CONFIG_NAME:-vampo_ppo_trainer_cluster4}" "$@"
}

ACTION="${1:-launch}"
shift || true
case "${ACTION}" in
  launch|start|"")
    cmd_launch "$@"
    ;;
  stop)
    cmd_stop
    ;;
  status)
    cmd_status
    ;;
  -h|--help|help)
    sed -n '2,12p' "$0" | sed 's/^# \?//'
    ;;
  *)
    cmd_launch "${ACTION}" "$@"
    ;;
esac
