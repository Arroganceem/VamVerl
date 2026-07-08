"""Precomputed clip WebDataset for VideoMAE success classification."""

from __future__ import annotations

import glob
import io
import json
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset

from vampo.reward.videomae_load import feature_extractor


def resolve_shard_globs(pattern: str | list[str]) -> list[str]:
    if isinstance(pattern, list):
        shards: list[str] = []
        for item in pattern:
            shards.extend(resolve_shard_globs(item))
        return sorted(set(shards))
    if any(ch in pattern for ch in "*?[]"):
        return sorted(glob.glob(pattern, recursive=True))
    return sorted(glob.glob(pattern))


def collate_clips(batch: list[tuple[torch.Tensor, int, dict[str, Any]]]):
    vids = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta_keys = batch[0][2].keys()
    meta = {k: [b[2][k] for b in batch] for k in meta_keys}
    return vids, ys, meta


class PrecomputedClipDataset(IterableDataset):
    """Read pre-exported 8-frame uint8 clips from WebDataset tar (no MP4 at train time)."""

    def __init__(
        self,
        shard_globs: list[str],
        *,
        img_size: int = 224,
        mode: str = "train",
        backbone: str | None = None,
        shuffle_buf: int = 512,
        use_resample: bool = True,
    ):
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be train|val, got {mode!r}")
        if not shard_globs:
            raise ValueError("shard_globs is empty")
        self.mode = mode
        self.fe = feature_extractor(backbone, img_size=img_size)

        if mode == "train":
            shard_source = (
                wds.ResampledShards(shard_globs, seed=42)
                if use_resample
                else wds.SimpleShardList(shard_globs)
            )
            pipeline: list[Any] = [
                shard_source,
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("clip.npy", "meta.json"),
                self._yield_clip,
            ]
            if shuffle_buf > 0:
                pipeline.append(wds.shuffle(shuffle_buf, initial=max(1, shuffle_buf // 4)))
        else:
            pipeline = [
                wds.SimpleShardList(shard_globs),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("clip.npy", "meta.json"),
                self._yield_clip,
            ]
        self.pipeline = wds.DataPipeline(*pipeline)

    def __iter__(self):
        return iter(self.pipeline)

    def _yield_clip(
        self, stream: Iterable[tuple[bytes, bytes]]
    ) -> Iterator[tuple[torch.Tensor, int, dict[str, Any]]]:
        for clip_bytes, meta_bytes in stream:
            clip = np.load(io.BytesIO(clip_bytes))
            meta = json.loads(meta_bytes.decode())
            frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
            tensor = self.fe(frames, return_tensors="pt")["pixel_values"][0]
            label = int(meta["label"])
            yield (
                tensor,
                label,
                {
                    "episode_index": int(meta.get("episode_index", -1)),
                    "video_end": int(meta.get("end", -1)),
                    "label": label,
                    "complete": bool(meta.get("complete", label == 1)),
                },
            )
