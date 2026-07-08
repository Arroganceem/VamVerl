#!/usr/bin/env bash
# 四机 NFS 挂载（41 → 31/21/11）· 仅负责挂载，不含 Ray/同步/训练
#
# 41 首次: sudo bash scripts/cluster/setup_nfs_exports_41.sh
#
# 用法（worker 上 robotem 通常已 sudo 免密，无需 SUDO_PASS）:
#   bash scripts/cluster/mount_nfs_cluster4.sh [all|models|vamverl|data|clips|fstab|status]
#   PERSIST_FSTAB=1 bash scripts/cluster/mount_nfs_cluster4.sh vamverl
#   WORKER_IPS="21" bash scripts/cluster/mount_nfs_cluster4.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NFS_HOST="${NFS_HOST:-192.168.88.41}"
SRC_BIND="${SRC_BIND:-192.168.88.41}"
REMOTE_USER="${REMOTE_USER:-robotem}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30 -o "BindAddress=${SRC_BIND}")
NFS_OPTS="${NFS_OPTS:-vers=3,_netdev,noatime,rsize=1048576,wsize=1048576}"

if [[ -n "${WORKER_IPS:-}" ]]; then
  WORKERS=()
  for ip in ${WORKER_IPS}; do
    [[ "${ip}" == *.* ]] && WORKERS+=("${ip##*.}") || WORKERS+=("${ip}")
  done
else
  WORKERS=(31 21 11)
fi

ACTION="${1:-all}"

_probe_remote() {
  local ip="$1" what="$2" rc
  set +e
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" \
    "NFS_HOST='${NFS_HOST}' WHAT='${what}'" bash -s <<'REMOTE'
set -euo pipefail
_nfs_ok() {
  local dst="$1" probe="$2"
  mountpoint -q "$dst" 2>/dev/null \
    && mount | grep -F " ${dst} " | grep -qE ' type nfs' \
    && [[ -e "${probe}" ]]
}
case "${WHAT}" in
  all)
    _nfs_ok /home/robotem/Models /home/robotem/Models/videomae-base/config.json && \
    _nfs_ok /home/robotem/WAM/VamVerl /home/robotem/WAM/VamVerl/data/videomae_droid_clips/manifest.json && \
    _nfs_ok /home/robotem/DATA /home/robotem/DATA/droid_lerobot/videos/chunk-000/observation.images.exterior_image_1_left/episode_000001.mp4
    ;;
  models) _nfs_ok /home/robotem/Models /home/robotem/Models/videomae-base/config.json ;;
  vamverl|clips|fstab) _nfs_ok /home/robotem/WAM/VamVerl /home/robotem/WAM/VamVerl/data/videomae_droid_clips/manifest.json ;;
  data) _nfs_ok /home/robotem/DATA /home/robotem/DATA/droid_lerobot/videos/chunk-000/observation.images.exterior_image_1_left/episode_000001.mp4 ;;
  *) echo "unknown"; exit 2 ;;
esac
REMOTE
  rc=$?
  set -e
  [[ "${rc}" -eq 0 ]] && return 0
  [[ "${rc}" -eq 255 ]] && return 2
  return 1
}

cmd_status() {
  local what="${1:-all}" ip rc fail=0
  echo ">>> NFS 状态 target=${what}"
  for ip in "${WORKERS[@]}"; do
    echo -n ">>> 192.168.88.${ip}: "
    if _probe_remote "${ip}" "${what}"; then
      echo "OK"
    else
      rc=$?
      if [[ "${rc}" -eq 2 ]]; then
        echo "FAIL SSH 不可达"
      else
        echo "FAIL 未挂载"
      fi
      fail=1
    fi
  done
  [[ "${fail}" -eq 0 ]] || exit 1
}

_mount_remote() {
  local ip="$1"
  local what="$2"
  local pass_env=""
  if [[ -n "${SUDO_PASS:-}" ]]; then
    pass_env="SUDO_PASS=$(printf '%q' "${SUDO_PASS}")"
  fi
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@192.168.88.${ip}" \
    "${pass_env} NFS_HOST='${NFS_HOST}' NFS_OPTS='${NFS_OPTS}' WHAT='${what}'" bash -s <<'REMOTE'
set -euo pipefail

_sudo() {
  if [[ -n "${SUDO_PASS:-}" ]]; then
    echo "${SUDO_PASS}" | sudo -S "$@"
  else
    sudo -n "$@"
  fi
}

_nfs_mounted() {
  local dst="$1"
  mountpoint -q "$dst" 2>/dev/null && mount | grep -F " ${dst} " | grep -qE ' type nfs'
}

_unmount_sshfs() {
  local dst="$1"
  if mountpoint -q "$dst" 2>/dev/null && mount | grep -F " ${dst} " | grep -q sshfs; then
    echo "  卸载旧 SSHFS: ${dst}"
    fusermount -u "${dst}" 2>/dev/null || _sudo umount "${dst}" 2>/dev/null || true
  fi
}

_mount_one() {
  local name="$1" src="$2" dst="$3" probe="$4"
  if [[ -e "${probe}" ]] && _nfs_mounted "${dst}"; then
    echo "  OK  ${name}: NFS 已挂载"
    return 0
  fi
  mkdir -p "${dst}"
  _unmount_sshfs "${dst}"
  if _nfs_mounted "${dst}"; then
    _sudo umount "${dst}" 2>/dev/null || true
  fi
  echo "  挂载 NFS ${src} → ${dst}"
  if ! _sudo mount -t nfs -o "${NFS_OPTS}" "${src}" "${dst}"; then
    echo "  FAIL ${name}: NFS 挂载失败（41 上是否已 sudo bash scripts/cluster/setup_nfs_exports_41.sh ?）" >&2
    return 1
  fi
  if [[ ! -e "${probe}" ]]; then
    echo "  FAIL ${name}: 挂载成功但 probe 不存在: ${probe}" >&2
    return 1
  fi
  echo "  OK  ${name}: NFS 挂载成功"
}

_ensure_vamverl_fstab() {
  local src="${NFS_HOST}:/home/robotem/WAM/VamVerl"
  local dst="/home/robotem/WAM/VamVerl"
  local line="${src} ${dst} nfs ${NFS_OPTS} 0 0"
  if grep -qF "${dst}" /etc/fstab 2>/dev/null; then
    echo "  OK  fstab: VamVerl 条目已存在"
  else
    echo "${line}" | _sudo tee -a /etc/fstab >/dev/null
    echo "  OK  fstab: 已追加 VamVerl 自动挂载"
  fi
}

case "${WHAT}" in
  all|models)
    _mount_one Models "${NFS_HOST}:/home/robotem/Models" /home/robotem/Models \
      /home/robotem/Models/videomae-base/config.json || exit 1
    ;;
esac

case "${WHAT}" in
  all|vamverl|clips|fstab)
    _mount_one VamVerl "${NFS_HOST}:/home/robotem/WAM/VamVerl" /home/robotem/WAM/VamVerl \
      /home/robotem/WAM/VamVerl/data/videomae_droid_clips/manifest.json || exit 1
    if [[ "${WHAT}" == "fstab" || "${PERSIST_FSTAB:-0}" == "1" ]]; then
      _ensure_vamverl_fstab
    fi
    ;;
esac

case "${WHAT}" in
  all|data)
    _mount_one DATA "${NFS_HOST}:/home/robotem/DATA" /home/robotem/DATA \
      /home/robotem/DATA/droid_lerobot/videos/chunk-000/observation.images.exterior_image_1_left/episode_000001.mp4 || exit 1
    ;;
esac
REMOTE
}

cmd_mount() {
  local target="$1"
  local _fail=0 ip
  echo ">>> NFS 挂载 (41 → worker) target=${target}"
  for ip in "${WORKERS[@]}"; do
    echo ">>> 192.168.88.${ip}"
    if ! _mount_remote "${ip}" "${target}"; then
      _fail=1
    fi
  done
  if [[ "${_fail}" -ne 0 ]]; then
    echo "" >&2
    echo ">>> 部分节点失败。41 上先执行:" >&2
    echo "    sudo bash ${ROOT}/scripts/cluster/setup_nfs_exports_41.sh" >&2
    echo ">>> worker 需 sudo 免密 mount（本集群默认已配置）；极少数情况可 SUDO_PASS=..." >&2
    exit 1
  fi
  echo ">>> 四机 NFS 挂载就绪"
}

case "${ACTION}" in
  status)
    cmd_status "${2:-all}"
    ;;
  all|models|vamverl|data|clips|fstab)
    cmd_mount "${ACTION}"
    ;;
  -h|--help|help)
    sed -n '2,10p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "用法: $0 [all|models|vamverl|data|clips|fstab|status]" >&2
    exit 1
    ;;
esac
