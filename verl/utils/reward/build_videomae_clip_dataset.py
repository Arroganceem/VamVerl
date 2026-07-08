"""Export precomputed 8-frame clips (uint8 pixels) to WebDataset tar shards."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import webdataset as wds

from verl.utils.reward.build_videomae_ready_split import READY_TRAIN, READY_VAL
from verl.utils.reward.videomae_episode import (
    DEFAULT_CAMERA,
    DEFAULT_CROP,
    DEFAULT_MAX_FRAMES,
    export_episode,
    load_episode_meta_map,
    resize_clip_uint8,
)
from verl.utils.reward.videomae_windows import train_window_specs, val_window_specs

logger = logging.getLogger(__name__)

TRAIN_WINDOWS = "videomae_train_windows.jsonl"
VAL_WINDOWS = "videomae_val_windows.jsonl"
TRAIN_SUCCESS = "videomae_train_success.jsonl"
TRAIN_FAILURE = "videomae_train_failure.jsonl"
VAL_SUCCESS = "videomae_val_success.jsonl"
VAL_FAILURE = "videomae_val_failure.jsonl"
EPISODE_OUTCOMES = "videomae_episode_outcomes.json"
WINDOW_MANIFEST = "videomae_window_manifest.json"
DATASET_READY = "videomae_dataset_ready.json"
CLIP_MANIFEST_NAME = "manifest.json"

REQUIRED_FILES = (
    TRAIN_WINDOWS,
    VAL_WINDOWS,
    TRAIN_SUCCESS,
    TRAIN_FAILURE,
    VAL_SUCCESS,
    VAL_FAILURE,
    EPISODE_OUTCOMES,
    WINDOW_MANIFEST,
    DATASET_READY,
)


def verify_dataset_ready(split_dir: str | Path) -> dict[str, Any]:
    split_dir = Path(split_dir)
    missing = [name for name in REQUIRED_FILES if not (split_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Dataset incomplete, missing: {missing}")

    ready = json.loads((split_dir / DATASET_READY).read_text())
    if ready.get("status") != "ready":
        raise RuntimeError(f"Dataset status not ready: {ready.get('status')}")

    train = ready["train"]
    val = ready["val"]
    errors: list[str] = []
    if train["success_clips"] < 1:
        errors.append("train success clips == 0")
    if train["failure_clips"] < 1:
        errors.append("train failure clips == 0")
    if val["windows"] < 1:
        errors.append("val windows == 0")

    for name in (TRAIN_WINDOWS, TRAIN_SUCCESS, TRAIN_FAILURE):
        path = split_dir / name
        if path.stat().st_size == 0:
            errors.append(f"empty file: {name}")

    if errors:
        raise RuntimeError("Dataset verification failed: " + "; ".join(errors))

    ready["verified"] = True
    return ready


def _load_episodes(split_dir: Path, name: str) -> list[int]:
    path = split_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run build_videomae_ready_split first")
    return json.loads(path.read_text())


def _write_jsonl_row(
    fall: TextIO,
    fok: TextIO,
    fbad: TextIO,
    row: dict[str, Any],
) -> tuple[int, int]:
    line = json.dumps(row, separators=(",", ":")) + "\n"
    fall.write(line)
    if row["label"] == 1:
        fok.write(line)
        return 1, 0
    fbad.write(line)
    return 0, 1


def build_clip_dataset(
    *,
    droid_root: str | Path,
    split_dir: str | Path,
    clip_dir: str | Path,
    window: int = 8,
    img_size: int = 224,
    stride_train: int = 4,
    stride_val: int = 1,
    finish_margin_k: int = 0,
    hard_neg_stride: int = 1,
    hard_neg_count: int = 0,
    pos_near_count_train: int = 24,
    pos_near_count_val: int = 24,  # 与 train 对齐：末尾成功几个，开头失败几个
    pos_near_stride: int = 1,
    max_frames: int = DEFAULT_MAX_FRAMES,
    crop: str = DEFAULT_CROP,
    camera: str = DEFAULT_CAMERA,
    shard_size: int = 512,
    max_clip_bytes: int = 0,
    val_byte_ratio: float = 0.05,
) -> dict[str, Any]:
    droid_root = Path(droid_root)
    split_dir = Path(split_dir)
    clip_dir = Path(clip_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "train").mkdir(parents=True, exist_ok=True)
    (clip_dir / "val").mkdir(parents=True, exist_ok=True)

    train_eps = _load_episodes(split_dir, READY_TRAIN)
    val_eps = _load_episodes(split_dir, READY_VAL)
    episode_meta = load_episode_meta_map(droid_root)

    kw = dict(
        window=window,
        finish_margin_k=finish_margin_k,
        hard_neg_stride=hard_neg_stride,
        hard_neg_count=hard_neg_count,
        pos_near_stride=pos_near_stride,
    )
    export_kw = dict(
        camera=camera,
        max_frames=max_frames,
        crop=crop,
        episode_meta=episode_meta,
    )

    outcomes: dict[str, Any] = {
        "train": {"success_episodes": [], "failure_episodes": []},
        "val": {"success_episodes": [], "failure_episodes": []},
    }

    if max_clip_bytes > 0:
        val_budget = int(max_clip_bytes * val_byte_ratio)
        train_budget = max(0, max_clip_bytes - val_budget)
        logger.info(
            "Clip byte cap: total=%.1fGB train=%.1fGB val=%.1fGB",
            max_clip_bytes / 1e9,
            train_budget / 1e9,
            val_budget / 1e9,
        )
    else:
        train_budget = val_budget = 0

    def _export_split(
        episodes: list[int],
        stride: int,
        *,
        mode: str,
        outcome_key: str,
        json_paths: tuple[Path, Path, Path],
        tar_pattern: str,
        pos_near_count: int,
        max_bytes: int,
    ) -> dict[str, int]:
        spec_fn = train_window_specs if mode == "train" else val_window_specs
        all_p, ok_p, bad_p = json_paths
        stats = {
            "episodes": 0,
            "windows": 0,
            "success_clips": 0,
            "failure_clips": 0,
            "episode_success": 0,
            "episode_failure": 0,
            "bytes_written": 0,
            "capped": False,
        }
        seen_eps: set[int] = set()
        clip_idx = 0
        hit_cap = False
        sink = wds.ShardWriter(tar_pattern, maxcount=shard_size)
        with all_p.open("w") as fall, ok_p.open("w") as fok, bad_p.open("w") as fbad:
            for i, ep in enumerate(episodes):
                if hit_cap:
                    break
                ep = int(ep)
                sample = export_episode(droid_root, ep, **export_kw)
                if sample is None:
                    continue
                video, meta = sample
                finish = int(meta["finish_step"])
                success = bool(meta["success"])
                if ep not in seen_eps:
                    seen_eps.add(ep)
                    stats["episodes"] += 1
                    bucket = outcomes[outcome_key]
                    if success:
                        bucket["success_episodes"].append(ep)
                        stats["episode_success"] += 1
                    else:
                        bucket["failure_episodes"].append(ep)
                        stats["episode_failure"] += 1
                specs = spec_fn(
                    finish,
                    success,
                    stride=stride,
                    pos_near_count=pos_near_count,
                    **kw,
                )
                if not specs:
                    continue
                for spec in specs:
                    if hit_cap:
                        break
                    end = int(spec["end"])
                    if end < window or end > len(video):
                        continue
                    raw = video[end - window : end]
                    clip = resize_clip_uint8(raw, img_size)
                    label = int(spec["label"])
                    row = {
                        "episode_index": ep,
                        "end": end,
                        "label": label,
                        "success": success,
                        "complete": success,
                    }
                    clip_meta = {
                        **row,
                        "window": window,
                        "img_size": img_size,
                    }
                    buf = io.BytesIO()
                    np.save(buf, clip, allow_pickle=False)
                    clip_bytes = buf.getvalue()
                    if max_bytes > 0 and stats["bytes_written"] + len(clip_bytes) > max_bytes:
                        stats["capped"] = True
                        hit_cap = True
                        break
                    pos, neg = _write_jsonl_row(fall, fok, fbad, row)
                    stats["windows"] += 1
                    stats["success_clips"] += pos
                    stats["failure_clips"] += neg
                    stats["bytes_written"] += len(clip_bytes)
                    sink.write(
                        {
                            "__key__": f"c{clip_idx:08d}",
                            "clip.npy": clip_bytes,
                            "meta.json": json.dumps(clip_meta).encode("utf-8"),
                        }
                    )
                    clip_idx += 1
                if (i + 1) % 500 == 0:
                    logger.info(
                        "%s clips: episodes %d/%d windows=%d",
                        mode,
                        i + 1,
                        len(episodes),
                        stats["windows"],
                    )
        sink.close()
        if stats["capped"]:
            logger.warning(
                "%s export capped at %.2fGB (limit %.2fGB)",
                mode,
                stats["bytes_written"] / 1e9,
                max_bytes / 1e9 if max_bytes > 0 else 0.0,
            )
        return stats

    train_stats = _export_split(
        train_eps,
        stride_train,
        mode="train",
        outcome_key="train",
        json_paths=(
            split_dir / TRAIN_WINDOWS,
            split_dir / TRAIN_SUCCESS,
            split_dir / TRAIN_FAILURE,
        ),
        tar_pattern=str(clip_dir / "train" / "clips_%05d.tar"),
        pos_near_count=pos_near_count_train,
        max_bytes=train_budget,
    )
    val_stats = _export_split(
        val_eps,
        stride_val,
        mode="val",
        outcome_key="val",
        json_paths=(
            split_dir / VAL_WINDOWS,
            split_dir / VAL_SUCCESS,
            split_dir / VAL_FAILURE,
        ),
        tar_pattern=str(clip_dir / "val" / "clips_%05d.tar"),
        pos_near_count=pos_near_count_val,
        max_bytes=val_budget,
    )

    (split_dir / EPISODE_OUTCOMES).write_text(json.dumps(outcomes, indent=2) + "\n")

    window_manifest = {
        "dataset_root": str(droid_root.resolve()),
        "clip_dir": str(clip_dir.resolve()),
        "window": window,
        "img_size": img_size,
        "shard_size": shard_size,
        "pos_near_count_train": pos_near_count_train,
        "pos_near_count_val": pos_near_count_val,
        "pos_near_stride": pos_near_stride,
        "finish_margin_k": finish_margin_k,
        "max_clip_bytes": max_clip_bytes,
        "val_byte_ratio": val_byte_ratio,
        "train": train_stats,
        "val": val_stats,
        "files": {
            "all_train": TRAIN_WINDOWS,
            "all_val": VAL_WINDOWS,
            "train_success": TRAIN_SUCCESS,
            "train_failure": TRAIN_FAILURE,
            "val_success": VAL_SUCCESS,
            "val_failure": VAL_FAILURE,
            "train_clips_glob": str(clip_dir / "train" / "*.tar"),
            "val_clips_glob": str(clip_dir / "val" / "*.tar"),
        },
    }
    (split_dir / WINDOW_MANIFEST).write_text(json.dumps(window_manifest, indent=2) + "\n")

    clip_manifest = {
        "train_glob": str(clip_dir / "train" / "*.tar"),
        "val_glob": str(clip_dir / "val" / "*.tar"),
        "clip_key": "clip.npy",
        "meta_key": "meta.json",
        "window": window,
        "img_size": img_size,
        "train_clips": train_stats["windows"],
        "val_clips": val_stats["windows"],
        "train_bytes": train_stats["bytes_written"],
        "val_bytes": val_stats["bytes_written"],
        "max_clip_bytes": max_clip_bytes,
        "capped": bool(train_stats.get("capped") or val_stats.get("capped")),
    }
    (clip_dir / CLIP_MANIFEST_NAME).write_text(json.dumps(clip_manifest, indent=2) + "\n")

    ready = {
        "status": "ready",
        "dataset_root": str(droid_root.resolve()),
        "clip_dir": str(clip_dir.resolve()),
        "labels": "prep_fixed",
        "pixels": "precomputed_uint8_clips",
        "max_clip_bytes": max_clip_bytes,
        "capped": bool(train_stats.get("capped") or val_stats.get("capped")),
        "success_definition": (
            "label=1: video end (finish + pos_near); "
            "label=0: video start, same count as success (no half-video split)"
        ),
        "train": train_stats,
        "val": val_stats,
        "episode_outcomes": {
            "train_success_eps": len(outcomes["train"]["success_episodes"]),
            "train_failure_eps": len(outcomes["train"]["failure_episodes"]),
            "val_success_eps": len(outcomes["val"]["success_episodes"]),
            "val_failure_eps": len(outcomes["val"]["failure_episodes"]),
        },
        "files": list(REQUIRED_FILES) + [f"clips/{CLIP_MANIFEST_NAME}"],
    }
    (split_dir / DATASET_READY).write_text(json.dumps(ready, indent=2) + "\n")
    logger.info(
        "Clip dataset: train=%d val=%d bytes train=%.1fGB val=%.1fGB -> %s",
        train_stats["windows"],
        val_stats["windows"],
        train_stats["bytes_written"] / 1e9,
        val_stats["bytes_written"] / 1e9,
        clip_dir,
    )
    return ready


def verify_clip_dataset(split_dir: str | Path, clip_dir: str | Path) -> dict[str, Any]:
    split_dir = Path(split_dir)
    clip_dir = Path(clip_dir)
    ready = verify_dataset_ready(split_dir)
    clip_manifest_path = clip_dir / CLIP_MANIFEST_NAME
    if not clip_manifest_path.is_file():
        raise RuntimeError(f"Missing clip manifest: {clip_manifest_path}")
    cm = json.loads(clip_manifest_path.read_text())
    import glob

    train_shards = glob.glob(cm["train_glob"])
    val_shards = glob.glob(cm["val_glob"])
    if not train_shards or not val_shards:
        raise RuntimeError(
            f"Missing clip tar shards train={len(train_shards)} val={len(val_shards)}"
        )
    ready["clip_manifest"] = cm
    ready["clip_shards"] = {"train": len(train_shards), "val": len(val_shards)}
    ready["verified_clips"] = True
    return ready


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Build precomputed VideoMAE clip dataset (pixels in tar)")
    p.add_argument("--droid-root", default=os.environ.get("DROID_DATA_ROOT", "/home/robotem/DATA/droid_lerobot"))
    p.add_argument("--split-dir", default=os.environ.get("DROID_SPLIT_DIR", str(root / "data/splits")))
    p.add_argument(
        "--clip-dir",
        default=os.environ.get("VIDEOMAE_CLIP_DIR", str(root / "data/videomae_droid_clips")),
    )
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--stride-train", type=int, default=4)
    p.add_argument("--stride-val", type=int, default=1)
    p.add_argument("--finish-margin-k", type=int, default=8)
    p.add_argument("--hard-neg-stride", type=int, default=1)
    p.add_argument("--hard-neg-count", type=int, default=2)
    p.add_argument(
        "--pos-near-count",
        type=int,
        default=24,
        help="extra success windows before finish (total success = 1+count; failure = same count from start)",
    )
    p.add_argument(
        "--pos-near-count-val",
        type=int,
        default=24,
        help="val: same rule as train (end success / start failure, equal counts)",
    )
    p.add_argument("--pos-near-stride", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument(
        "--max-clip-gb",
        type=float,
        default=float(os.environ.get("MAX_CLIP_GB", "400")),
        help="Max total clip bytes in GB (0=unlimited). Train/val split 95/5.",
    )
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    max_clip_bytes = int(args.max_clip_gb * 1e9) if args.max_clip_gb > 0 else 0

    if args.verify_only:
        print(json.dumps(verify_clip_dataset(args.split_dir, args.clip_dir), indent=2))
        return

    ready = build_clip_dataset(
        droid_root=args.droid_root,
        split_dir=args.split_dir,
        clip_dir=args.clip_dir,
        window=args.window,
        img_size=args.img_size,
        stride_train=args.stride_train,
        stride_val=args.stride_val,
        finish_margin_k=args.finish_margin_k,
        hard_neg_stride=args.hard_neg_stride,
        hard_neg_count=args.hard_neg_count,
        pos_near_count_train=args.pos_near_count,
        pos_near_count_val=args.pos_near_count_val,
        pos_near_stride=args.pos_near_stride,
        shard_size=args.shard_size,
        max_clip_bytes=max_clip_bytes,
    )
    ready = verify_clip_dataset(args.split_dir, args.clip_dir)
    print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
