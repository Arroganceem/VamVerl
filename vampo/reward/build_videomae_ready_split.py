"""Scan DROID split and write episode lists that have local MP4 ready for clip export."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2

from vampo.data.droid_episode_split import load_episode_split
from vampo.data.lerobot_io import resolve_video_path
from vampo.reward.videomae_episode import DEFAULT_CAMERA

logger = logging.getLogger(__name__)

READY_TRAIN = "videomae_ready_train_episodes.json"
READY_VAL = "videomae_ready_val_episodes.json"
READY_MANIFEST = "videomae_ready_manifest.json"


def probe_episode(
    droid_root: Path,
    episode_index: int,
    *,
    camera: str = DEFAULT_CAMERA,
    min_frames: int = 8,
) -> bool:
    video_path = resolve_video_path(droid_root, episode_index, camera)
    if not video_path.is_file():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frames >= min_frames


def filter_ready(
    episodes: list[int],
    droid_root: Path,
    *,
    camera: str,
    min_frames: int,
) -> tuple[list[int], int]:
    ready: list[int] = []
    missing = 0
    total = len(episodes)
    for i, ep in enumerate(episodes):
        if probe_episode(droid_root, int(ep), camera=camera, min_frames=min_frames):
            ready.append(int(ep))
        else:
            missing += 1
        if total >= 5000 and (i + 1) % 5000 == 0:
            logger.info("Scanned %d/%d ready=%d", i + 1, total, len(ready))
    return ready, missing


def build_ready_split(
    *,
    droid_root: str | Path,
    split_dir: str | Path,
    camera: str = DEFAULT_CAMERA,
    min_frames: int = 8,
    max_train: int = 0,
    max_val: int = 0,
) -> dict:
    droid_root = Path(droid_root)
    split_dir = Path(split_dir)
    split = load_episode_split(split_dir)
    train_eps = list(split["train_episodes"])
    val_eps = list(split["val_episodes"])
    if max_train > 0:
        train_eps = train_eps[:max_train]
    if max_val > 0:
        val_eps = val_eps[:max_val]

    ready_train, miss_train = filter_ready(
        train_eps, droid_root, camera=camera, min_frames=min_frames
    )
    ready_val, miss_val = filter_ready(
        val_eps, droid_root, camera=camera, min_frames=min_frames
    )

    manifest = {
        "dataset_root": str(droid_root.resolve()),
        "camera": camera,
        "min_frames": min_frames,
        "train_total": len(train_eps),
        "train_ready": len(ready_train),
        "train_missing": miss_train,
        "val_total": len(val_eps),
        "val_ready": len(ready_val),
        "val_missing": miss_val,
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / READY_TRAIN).write_text(json.dumps(ready_train, indent=2) + "\n")
    (split_dir / READY_VAL).write_text(json.dumps(ready_val, indent=2) + "\n")
    (split_dir / READY_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Ready split: train %d/%d val %d/%d -> %s",
        len(ready_train),
        len(train_eps),
        len(ready_val),
        len(val_eps),
        split_dir,
    )
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Build VideoMAE ready episode lists (local MP4 exists)")
    p.add_argument(
        "--droid-root",
        default=os.environ.get("DROID_DATA_ROOT", "/home/robotem/DATA/droid_lerobot"),
    )
    p.add_argument(
        "--split-dir",
        default=os.environ.get("DROID_SPLIT_DIR", str(root / "data/splits")),
    )
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    args = p.parse_args()
    manifest = build_ready_split(
        droid_root=args.droid_root,
        split_dir=args.split_dir,
        camera=args.camera,
        min_frames=args.min_frames,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    if manifest["train_ready"] == 0:
        raise SystemExit("No train episodes ready — check DROID_DATA_ROOT and video sync")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
