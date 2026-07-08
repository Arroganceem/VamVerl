#!/usr/bin/env bash
# 在 41 上运行一次（需 sudo），导出四机共享目录（全部 NFS）
# 用法: sudo bash scripts/setup_nfs_exports_41.sh
set -euo pipefail

EXPORTS_FILE="/etc/exports"
SUBNET="${NFS_SUBNET:-192.168.88.0/24}"
WORKERS="${NFS_WORKERS:-192.168.88.31 192.168.88.21 192.168.88.11}"

# 四机共享目录（与原先 SSHFS 一致 + 已有 UniVTM）
SHARED_PATHS=(
  /home/robotem/Models
  /home/robotem/DATA
  /home/robotem/WAM/VamVerl
  /home/robotem/WAM/UniVTM/data
)

if [[ "$(id -u)" -ne 0 ]]; then
  echo ">>> 请在 41 上用 sudo 运行:" >&2
  echo "    sudo bash scripts/setup_nfs_exports_41.sh" >&2
  exit 1
fi

_worker_opts() {
  local out=""
  for ip in ${WORKERS}; do
    out+="${ip}(rw,sync,no_subtree_check,no_root_squash) "
  done
  printf '%s' "${out% }"
}

_subnet_opts() {
  printf '%s(rw,sync,no_subtree_check,no_root_squash)' "${SUBNET}"
}

_append_export() {
  local path="$1"
  local clients="$2"
  local line="${path} ${clients}"
  if grep -qE "^${path//\//\\/}[[:space:]]" "${EXPORTS_FILE}"; then
    echo "  已有 export: ${path}（保留原条目，请手动核对客户端列表）"
  else
    echo "${line}" >> "${EXPORTS_FILE}"
    echo "  新增 export: ${path}"
  fi
}

echo ">>> 更新 ${EXPORTS_FILE}（四机共享目录 → NFS）"
for path in "${SHARED_PATHS[@]}"; do
  if [[ ! -d "${path}" ]]; then
    echo "  WARN 路径不存在，跳过: ${path}" >&2
    continue
  fi
  if [[ "${path}" == "/home/robotem/Models" ]] || [[ "${path}" == "/home/robotem/DATA" ]]; then
    _append_export "${path}" "$(_subnet_opts)"
  else
    _append_export "${path}" "$(_worker_opts)"
  fi
done

exportfs -ra
echo ""
echo ">>> 当前 export 列表:"
exportfs -v | grep -E 'Models|/DATA|VamVerl|UniVTM' || exportfs -v
echo ""
echo ">>> 41 NFS export 就绪"
echo ">>> worker: bash scripts/mount_nfs_cluster4.sh all"
