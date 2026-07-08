#!/usr/bin/env bash
# Component2 完整数据准备：读 MP4 → 导出 8 帧 uint8 图片 clip 到 tar + 标签 jsonl
#
# 产出（均在项目 data/ 下）:
#   data/splits/videomae_*_windows.jsonl / videomae_dataset_ready.json
#   data/videomae_droid_clips/train/*.tar  每条 clip.npy (8,224,224,3) + label
#   data/videomae_droid_clips/val/*.tar
#
# 注意: 默认 clip 总量上限 400GB（MAX_CLIP_GB=400），输出在 VamVerl/data/
#
# 用法:
#   bash scripts/prep_component2_data_local.sh          # 全量
#   bash scripts/prep_component2_data_local.sh smoke    # 小规模冒烟（20 train + 5 val episode）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SMOKE="${1:-}"
MAX_TRAIN=0
MAX_VAL=0
SHARD_SIZE=512
MAX_CLIP_GB="${MAX_CLIP_GB:-400}"

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-vamverl}"
DROID_ROOT="${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}"
SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
CLIP_DIR="${VIDEOMAE_CLIP_DIR:-$ROOT/data/videomae_droid_clips}"

if [[ "${SMOKE}" == "smoke" ]]; then
  MAX_TRAIN="${SMOKE_MAX_TRAIN:-20}"
  MAX_VAL="${SMOKE_MAX_VAL:-5}"
  SHARD_SIZE="${SMOKE_SHARD_SIZE:-64}"
  SMOKE_MAX_EPISODES="${SMOKE_MAX_EPISODES:-500}"
  SPLIT_DIR="${ROOT}/data/splits_smoke"
  CLIP_DIR="${ROOT}/data/videomae_droid_clips_smoke"
  MAX_CLIP_GB="${SMOKE_MAX_CLIP_GB:-0}"
fi

mkdir -p "${SPLIT_DIR}" "${CLIP_DIR}/train" "${CLIP_DIR}/val"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi

export DROID_DATA_ROOT="${DROID_ROOT}"
export DROID_SPLIT_DIR="${SPLIT_DIR}"
export VIDEOMAE_CLIP_DIR="${CLIP_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python - <<PY
from pathlib import Path
from vampo.data.droid_episode_split import (
    SPLIT_MANIFEST,
    TRAIN_FILE,
    build_episode_split_from_dataset,
)

split_dir = Path("${SPLIT_DIR}")
droid_root = Path("${DROID_ROOT}")
has_split = (split_dir / SPLIT_MANIFEST).is_file() or (split_dir / TRAIN_FILE).is_file()
if not has_split:
    max_eps = ${SMOKE_MAX_EPISODES:-0}
    if max_eps <= 0 and "${SMOKE}" == "smoke":
        max_eps = 500
    print(f">>> 生成 episode split（max_episodes={max_eps or 'all'}）")
    split_dir.mkdir(parents=True, exist_ok=True)
    build_episode_split_from_dataset(
        droid_root,
        split_dir,
        max_episodes=max_eps or None,
    )
PY

echo ">>> Component2 预计算 clip 数据集"
if [[ "${SMOKE}" == "smoke" ]]; then
  echo ">>> 模式: smoke (train=${MAX_TRAIN} val=${MAX_VAL} episodes)"
else
  echo ">>> clip 上限: ${MAX_CLIP_GB}GB（train≈$(( MAX_CLIP_GB * 95 / 100 ))GB val≈$(( MAX_CLIP_GB * 5 / 100 ))GB）"
fi
echo ">>> DROID=${DROID_ROOT}"
echo ">>> splits → ${SPLIT_DIR}"
echo ">>> clips → ${CLIP_DIR}"
echo ""

echo ">>> [1/3] 扫描可用 episode"
python -m vampo.reward.build_videomae_ready_split \
  --droid-root "${DROID_ROOT}" \
  --split-dir "${SPLIT_DIR}" \
  --max-train "${MAX_TRAIN}" \
  --max-val "${MAX_VAL}"

echo ""
echo ">>> [2/3] 读 MP4 + 导出 8 帧图片 clip（uint8 224×224）到 tar"
python -m vampo.reward.build_videomae_clip_dataset \
  --droid-root "${DROID_ROOT}" \
  --split-dir "${SPLIT_DIR}" \
  --clip-dir "${CLIP_DIR}" \
  --window 8 \
  --img-size 224 \
  --stride-train 4 \
  --stride-val 1 \
  --finish-margin-k 8 \
  --hard-neg-stride 1 \
  --hard-neg-count 2 \
  --pos-near-count 24 \
  --pos-near-count-val 0 \
  --pos-near-stride 1 \
  --shard-size "${SHARD_SIZE}" \
  --max-clip-gb "${MAX_CLIP_GB}"

echo ""
echo ">>> [3/3] 校验 clip 数据集"
python -m vampo.reward.build_videomae_clip_dataset \
  --split-dir "${SPLIT_DIR}" \
  --clip-dir "${CLIP_DIR}" \
  --verify-only

python - <<PY
import json
from pathlib import Path
split = Path("${SPLIT_DIR}")
clip = Path("${CLIP_DIR}")
ready = json.loads((split / "videomae_dataset_ready.json").read_text())
cm = json.loads((clip / "manifest.json").read_text())
tr = ready["train"]
va = ready["val"]
print()
print(">>> 数据集准备完毕（含像素 clip）✓")
print(f"    train clips {tr['windows']}  success={tr['success_clips']}  failure={tr['failure_clips']}")
print(f"    val clips   {va['windows']}")
print(f"    clip bytes  train={tr['bytes_written']/1e9:.2f}GB  val={va['bytes_written']/1e9:.2f}GB  total={(tr['bytes_written']+va['bytes_written'])/1e9:.2f}GB")
if ready.get("capped"):
    cap = ready.get("max_clip_bytes", 0)
    print(f"    capped at {cap/1e9:.0f}GB limit (partial episodes exported)")
print(f"    tar shards  train={cm['train_glob']}  val={cm['val_glob']}")
print()
print(">>> 开训（训练只读 tar clip，不再读 MP4）:")
print("    sudo bash scripts/setup_nfs_exports_41.sh   # 41 首次")
print("    bash scripts/mount_nfs_cluster4.sh clips")
print("    bash scripts/preflight_component2_train.sh")
print("    bash scripts/train_component2_reward_cluster4.sh")
PY
