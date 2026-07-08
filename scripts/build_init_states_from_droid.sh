#!/usr/bin/env bash
# Build RL init_states from DROID rl_init episode pool (disjoint from hold-out train).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}"
export INIT_STATES_DIR="${INIT_STATES_DIR:-$ROOT/data/init_states}"
export DROID_SPLIT_DIR="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
export MAX_INIT_STATES="${MAX_INIT_STATES:-256}"
export INIT_STATES_CAMERA="${INIT_STATES_CAMERA:-exterior_image_1_left}"
export SPLIT_SEED="${SPLIT_SEED:-42}"
export VAL_RATIO="${VAL_RATIO:-0.05}"
export RL_INIT_SOURCE="${RL_INIT_SOURCE:-val}"
export RL_HOLDOUT_RATIO="${RL_HOLDOUT_RATIO:-0.10}"
cd "$ROOT"
ARGS=(
  --dataset-root "$DROID_DATA_ROOT"
  --output-dir "$INIT_STATES_DIR"
  --split-dir "$DROID_SPLIT_DIR"
  --camera "$INIT_STATES_CAMERA"
  --max-episodes "$MAX_INIT_STATES"
  --seed "$SPLIT_SEED"
  --val-ratio "$VAL_RATIO"
  --rl-init-source "$RL_INIT_SOURCE"
  --rl-holdout-ratio "$RL_HOLDOUT_RATIO"
)
python -m vampo.data.droid_to_init_states "${ARGS[@]}"
python -m vampo.data.check_episode_overlap \
  --init-manifest "$INIT_STATES_DIR/manifest.json" \
  --split-dir "$DROID_SPLIT_DIR" \
  --strict
