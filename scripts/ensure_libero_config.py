#!/usr/bin/env python3
"""Create ~/.libero/config.yaml without interactive prompts (VAMPO does not use LIBERO)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def ensure_libero_config() -> None:
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.is_file():
        return

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        return

    libero_root = Path(spec.submodule_search_locations[0])
    benchmark_root = libero_root / "libero"
    default_paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(libero_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }

    config_dir.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        config_file.write_text(yaml.dump(default_paths), encoding="utf-8")
    else:
        lines = [f"{key}: {value}\n" for key, value in default_paths.items()]
        config_file.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ensure_libero_config()
