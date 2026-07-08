"""CLI for staged VamVerl verification (no full training required)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from verl.trainer.preflight.staged_checks import (
    STAGE_NAMES,
    print_staged_report,
    run_stages,
    summarize_stages,
)


def _parse_stages(through: int | None, only: str | None) -> list[int]:
    if only is not None:
        parts = [p.strip() for p in only.split(",") if p.strip()]
        stages: list[int] = []
        for part in parts:
            if part.isdigit():
                stages.append(int(part))
            elif part in STAGE_NAMES.values():
                stages.append(next(k for k, v in STAGE_NAMES.items() if v == part))
            else:
                raise ValueError(f"Unknown stage {part!r}; use 0-5 or {list(STAGE_NAMES.values())}")
        return sorted(set(stages))
    end = 3 if through is None else through
    return list(range(0, end + 1))


def main() -> None:
    root = Path(os.environ.get("VAMVERL_ROOT", Path(__file__).resolve().parents[3]))
    p = argparse.ArgumentParser(
        description="VamVerl staged verification (static → data → RL mock → config → VideoMAE)"
    )
    p.add_argument("--config", default=str(root / "configs/vampo_ppo_trainer.yaml"))
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/home/robotem/Models/DreamZero-DROID"))
    p.add_argument("--droid-root", default=os.environ.get("DROID_DATA_ROOT", "/home/robotem/DATA/droid_lerobot"))
    p.add_argument("--init-dir", default=os.environ.get("INIT_STATES_DIR", str(root / "data/init_states")))
    p.add_argument("--split-dir", default=os.environ.get("DROID_SPLIT_DIR", str(root / "data/splits")))
    p.add_argument(
        "--through",
        type=int,
        default=None,
        help="运行 stage 0..N（默认 0..3，不含 VideoMAE GPU 冒烟）",
    )
    p.add_argument(
        "--only",
        help="仅运行指定阶段，逗号分隔（名称或编号），如 2,4 或 rl-mock,videomae",
    )
    p.add_argument(
        "--videomae-device",
        default=os.environ.get("VIDEOMAE_VERIFY_DEVICE", "cpu"),
        help="Stage 4 VideoMAE 推理设备（默认 cpu，避免 OOM）",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    try:
        stages = _parse_stages(args.through, args.only)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    results = run_stages(
        stages,
        root=root,
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        droid_root=Path(args.droid_root),
        init_dir=Path(args.init_dir),
        split_dir=Path(args.split_dir),
        videomae_device=args.videomae_device,
    )

    if args.json:
        github_ready, train_ready = summarize_stages(results, stages)
        print(
            json.dumps(
                {
                    "stages": stages,
                    "github_ready": github_ready,
                    "train_ready": train_ready,
                    "checks": [asdict(r) for r in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print("=== VamVerl Staged Verification ===")
        print_staged_report(results, stages_run=stages)

    github_ready, _ = summarize_stages(results, stages)
    if not github_ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
