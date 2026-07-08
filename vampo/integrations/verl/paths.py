"""Path resolution for verl workers (offline cluster, no HF hub)."""

from __future__ import annotations

import os


def resolve_tokenizer_path(explicit: str | None = None) -> str | None:
    """Return a local umt5-xxl tokenizer dir when HF hub is unavailable."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for key in ("TOKENIZER_PATH",):
        val = os.environ.get(key)
        if val:
            candidates.append(val)
    wan21 = os.environ.get("WAN21_DIR")
    if wan21:
        candidates.append(os.path.join(wan21, "google", "umt5-xxl"))
    candidates.append("/home/robotem/Models/umt5-xxl")
    wan22 = os.environ.get("WAN22_DIR")
    if wan22:
        candidates.append(os.path.join(wan22, "google", "umt5-xxl"))

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(os.path.join(path, "tokenizer_config.json")):
            return path
    return explicit
