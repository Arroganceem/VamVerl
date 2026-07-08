# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import torch
import torch.distributed as dist


def _process_rss_gb() -> float | None:
    """Resident set size (GB). On GB10 unified memory this tracks the shared pool."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        parts = [head]
        rss = _process_rss_gb()
        if rss is not None:
            parts.append(f"process RSS (GB): {rss:.2f}")
        if torch.cuda.is_available():
            parts.append(
                f"cuda allocated/reserved (GB): "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f}/"
                f"{torch.cuda.memory_reserved() / 1024**3:.2f}"
            )
        message = ", ".join(parts)

        if logger is None:
            print(message, flush=True)
        else:
            logger.log(msg=message, level=level)


def trim_process_heap() -> None:
    """Return freed heap pages to OS (helps GB10 unified memory reclaim after large peaks)."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
