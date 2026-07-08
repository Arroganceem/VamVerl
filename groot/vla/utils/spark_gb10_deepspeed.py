"""GB10 UMA: DeepSpeed ZeRO init calls pynvml.nvmlDeviceGetMemoryInfo → Not Supported.

Patch pynvml to fall back to torch.cuda.mem_get_info(). Safe for multi-node NCCL
(unlike LD_LIBRARY_PATH replacement of libnvidia-ml.so.1).
"""

from __future__ import annotations

import os
from collections import namedtuple
from typing import Any

_PATCHED = False
NvmlMemoryInfo = namedtuple("NvmlMemoryInfo", ("total", "free", "used"))


def _needs_patch() -> bool:
    if os.environ.get("DREAMZERO_DISABLE_GB10_DEEPSPEED_PATCH") == "1":
        return False
    if os.environ.get("DREAMZERO_ENABLE_GB10_DEEPSPEED_PATCH") == "1":
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetMemoryInfo(handle)
        return False
    except Exception:
        return True


def _memory_info_from_torch(device: int = 0) -> NvmlMemoryInfo:
    import torch

    if not torch.cuda.is_available():
        return NvmlMemoryInfo(total=0, free=0, used=0)
    free, total = torch.cuda.mem_get_info(device)
    return NvmlMemoryInfo(total=total, free=free, used=total - free)


def apply_spark_gb10_deepspeed_patch(force: bool = False) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not force and not _needs_patch():
        return False

    import pynvml

    original = pynvml.nvmlDeviceGetMemoryInfo

    def patched_get_memory_info(handle: Any) -> NvmlMemoryInfo:
        try:
            return original(handle)
        except pynvml.NVMLError_NotSupported:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            return _memory_info_from_torch(local_rank)

    pynvml.nvmlDeviceGetMemoryInfo = patched_get_memory_info  # type: ignore[method-assign]
    _PATCHED = True
    return True
