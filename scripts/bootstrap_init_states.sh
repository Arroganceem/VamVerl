#!/usr/bin/env bash
# Build or validate init_states from DROID rl_init pool (sourced by train_component3_rl_cluster4.sh).
ensure_init_states() {
  local out="${INIT_STATES_DIR:-$ROOT/data/init_states}"
  local split="${DROID_SPLIT_DIR:-$ROOT/data/splits}"
  python - <<PY
import os
from vampo.data.init_states_bootstrap import ensure_init_states

ensure_init_states(
    "${out}",
    dataset_root=os.environ.get("DROID_DATA_ROOT"),
    max_episodes=int(os.environ.get("MAX_INIT_STATES", "2888")),
    camera=os.environ.get("INIT_STATES_CAMERA", "exterior_image_1_left"),
    force_rebuild=os.environ.get("INIT_STATES_FORCE", "0") == "1",
    split_dir="${split}",
    val_ratio=float(os.environ.get("VAL_RATIO", "0.05")),
    rl_init_source=os.environ.get("RL_INIT_SOURCE", "val"),
    rl_holdout_ratio=float(os.environ.get("RL_HOLDOUT_RATIO", "0.10")),
    check_overlap=True,
)
print("[init_states] ready:", "${out}", "split:", "${split}")
PY
}
