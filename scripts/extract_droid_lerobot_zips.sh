#!/usr/bin/env bash
# Extract ModelScope yangt18/DreamZero-DROID zip bundle into LeRobot layout.
set -euo pipefail

ROOT="${1:-${DROID_DATA_ROOT:-/home/robotem/DATA/droid_lerobot}}"
ZIP_DIR="${ROOT}/droid_lerobot_zip"
LOG="${ROOT}/extract_modelscope.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

if [[ ! -d "$ZIP_DIR" ]]; then
  echo "[extract] no zip dir ($ZIP_DIR); skip (dataset may already be LeRobot layout)"
  exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] unzip not found" >&2
  exit 1
fi

mkdir -p "$ROOT/videos"
: > "$LOG"

log "=== DROID LeRobot extract (ModelScope zips) ==="
log "ROOT=$ROOT"

log "Step 1/3: meta.zip"
if [[ -f "$ROOT/meta/info.json" ]] || [[ -f "$ROOT/meta/episodes.jsonl" ]]; then
  log "  skip (meta already present)"
else
  unzip -o -q "$ZIP_DIR/meta.zip" -d "$ROOT"
  log "  done"
fi

log "Step 2/3: data.zip"
parquet_count="$(find "$ROOT/data" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${parquet_count:-0}" -ge 7000 ]]; then
  log "  skip ($parquet_count parquet files already present)"
else
  unzip -o -q "$ZIP_DIR/data.zip" -d "$ROOT"
  parquet_count="$(find "$ROOT/data" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
  log "  done ($parquet_count parquet files)"
fi

log "Step 3/3: chunk-*.zip -> videos/"
mapfile -t chunks < <(compgen -G "$ZIP_DIR/chunk-*.zip" || true)
total="${#chunks[@]}"
done_count=0
skip_count=0

for i in "${!chunks[@]}"; do
  z="${chunks[$i]}"
  chunk_name="$(basename "$z" .zip)"
  marker="$ROOT/videos/${chunk_name}/.extracted"

  if [[ -f "$marker" ]]; then
    ((skip_count++)) || true
    continue
  fi

  log "  [$((i + 1))/$total] $chunk_name"
  unzip -o -q "$z" -d "$ROOT/videos"
  touch "$marker"
  ((done_count++)) || true
done

mp4_count="$(find "$ROOT/videos" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"

log "=== extract summary ==="
log "parquet: ${parquet_count:-0}"
log "mp4:     ${mp4_count:-0}"
log "chunks extracted this run: $done_count, skipped: $skip_count / $total"
log "=== done ==="
