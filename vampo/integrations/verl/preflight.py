"""Pre-flight checks before verl GRPO + PPO training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CheckStatus = Literal["ok", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    hint: str = ""


def _ok(name: str, message: str) -> CheckResult:
    return CheckResult(name, "ok", message)


def _warn(name: str, message: str, hint: str = "") -> CheckResult:
    return CheckResult(name, "warn", message, hint)


def _fail(name: str, message: str, hint: str = "") -> CheckResult:
    return CheckResult(name, "fail", message, hint)


def check_python_imports() -> CheckResult:
    missing: list[str] = []
    for mod in ("torch", "numpy", "omegaconf", "ray"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return _fail(
            "python_imports",
            f"缺少依赖: {', '.join(missing)}",
            "pip install -e \".[verl,vla]\"",
        )
    try:
        __import__("verl")
    except ImportError as exc:
        return _fail(
            "python_imports",
            f"verl 不可导入: {exc}",
            "pip install -e \".[verl,vla]\"",
        )
    return _ok("python_imports", "torch / ray / verl 可导入")


def check_cuda() -> CheckResult:
    try:
        import torch

        if not torch.cuda.is_available():
            return _warn(
                "cuda",
                "CUDA 不可用（将极慢或无法训练 VLA）",
                "确认 nvidia-smi 正常且 PyTorch 为 GPU 版本",
            )
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return _ok("cuda", f"{name} · {total_gb:.1f} GiB")
    except Exception as exc:
        return _warn("cuda", f"无法检测 CUDA: {exc}")


def _load_reward_settings(config_path: Path) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    reward = OmegaConf.select(cfg, "actor_rollout_ref.reward") or {}
    backend = str(reward.get("backend", "videomae")).lower()
    return {
        "backend": backend,
        "videomae_checkpoint": (
            reward.get("videomae_checkpoint")
            or os.environ.get("VIDEOMAE_CKPT")
        ),
        "hf_model_id": (
            reward.get("hf_model_id")
            or reward.get("hf_model_path")
            or os.environ.get("VIDEOMAE_BACKBONE")
        ),
    }


def check_config(config_path: Path) -> CheckResult:
    if not config_path.is_file():
        return _fail(
            "config",
            f"配置文件不存在: {config_path}",
            "使用 configs/vampo_ppo_trainer.yaml",
        )
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        reward = OmegaConf.select(cfg, "actor_rollout_ref.reward") or {}
        backend = str(reward.get("backend", "videomae")).lower()
        model_path = OmegaConf.select(cfg, "vla.model_path") or OmegaConf.select(
            cfg, "actor_rollout_ref.model.path"
        )
        ckpt = reward.get("videomae_checkpoint") or os.environ.get("VIDEOMAE_CKPT")
        backbone = (
            reward.get("hf_model_id")
            or reward.get("hf_model_path")
            or os.environ.get("VIDEOMAE_BACKBONE")
        )
        return _ok(
            "config",
            f"{config_path.name} · model={model_path} · reward={backend} · "
            f"backbone={backbone or 'default'} · ckpt={ckpt}",
        )
    except Exception as exc:
        return _fail("config", f"无法解析配置: {exc}")


def check_vla_model(model_path: Path) -> CheckResult:
    if not model_path.is_dir():
        return _fail(
            "vla_model",
            f"DreamZero 基座目录不存在: {model_path}",
            "export MODEL_PATH=/path/to/DreamZero-DROID",
        )
    markers = ["config.json", "experiment_cfg"]
    found = [m for m in markers if (model_path / m).exists()]
    if not found:
        return _fail(
            "vla_model",
            f"目录缺少 config.json / experiment_cfg: {model_path}",
        )
    shards = list(model_path.glob("model-*.safetensors")) + list(
        model_path.glob("pytorch_model*.bin")
    )
    if not shards:
        return _warn(
            "vla_model",
            f"未找到 weight shard，请确认 ckpt 完整: {model_path}",
        )
    return _ok("vla_model", f"{model_path} · {len(shards)} shard(s)")


def check_droid_data(droid_root: Path) -> CheckResult:
    if not droid_root.is_dir():
        return _fail(
            "droid_data",
            f"DROID 数据目录不存在: {droid_root}",
            "export DROID_DATA_ROOT=/home/robotem/DATA/droid_lerobot",
        )
    episodes = droid_root / "meta" / "episodes.jsonl"
    if not episodes.is_file():
        return _fail(
            "droid_data",
            f"缺少 LeRobot episodes: {episodes}",
        )
    n_lines = sum(1 for _ in episodes.open())
    return _ok("droid_data", f"{droid_root} · {n_lines} episodes")


def check_episode_split(split_dir: Path) -> CheckResult:
    manifest = split_dir / "droid_episode_split.json"
    rl_init = split_dir / "rl_init_episodes.json"
    if not manifest.is_file() and not rl_init.is_file():
        return _fail(
            "episode_split",
            f"episode split 未生成: {split_dir}",
            "bash scripts/build_init_states_from_droid.sh",
        )
    try:
        from vampo.data.droid_episode_split import load_episode_split

        split = load_episode_split(split_dir)
        return _ok(
            "episode_split",
            f"train={len(split['train_episodes'])} val={len(split['val_episodes'])} "
            f"rl_init={len(split['rl_init_episodes'])} source={split.get('rl_init_source')}",
        )
    except Exception as exc:
        return _fail("episode_split", str(exc), "bash scripts/build_init_states_from_droid.sh")


def check_init_states(init_dir: Path) -> CheckResult:
    try:
        from vampo.data.init_states_bootstrap import validate_init_states

        validate_init_states(init_dir)
        manifest = json.loads((init_dir / "manifest.json").read_text())
        entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
        return _ok("init_states", f"{init_dir} · {len(entries)} entries")
    except Exception as exc:
        return _fail(
            "init_states",
            str(exc),
            "bash scripts/build_init_states_from_droid.sh",
        )


def check_init_overlap(init_dir: Path, split_dir: Path) -> CheckResult:
    manifest = init_dir / "manifest.json"
    if not manifest.is_file():
        return _warn("init_overlap", "跳过（init_states 未就绪）")
    try:
        from vampo.data.check_episode_overlap import check_init_manifest_overlap

        report = check_init_manifest_overlap(manifest, split_dir)
        if report["ok"]:
            return _ok("init_overlap", "rl_init 与 train pool 无泄漏")
        return _fail(
            "init_overlap",
            f"overlap={report['overlap_count']} outside_pool={len(report['not_in_rl_init_pool'])}",
            "bash scripts/build_init_states_from_droid.sh",
        )
    except Exception as exc:
        return _fail("init_overlap", str(exc))


def check_videomae_checkpoint(checkpoint_path: str | None) -> CheckResult:
    if not checkpoint_path:
        return _fail(
            "videomae_ckpt",
            "未配置 videomae_checkpoint / VIDEOMAE_CKPT",
            "export VIDEOMAE_CKPT=/path/to/videomae_droid.pth 或在 yaml 中设置",
        )
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        return _fail(
            "videomae_ckpt",
            f"VideoMAE checkpoint 不存在: {ckpt}",
            "从 WMPO 训练或拷贝 checkpoint；见 WMPO/reward_model/videomae.py",
        )
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    return _ok("videomae_ckpt", f"{ckpt} · {size_mb:.1f} MiB")


def check_videomae_backbone(hf_model_id: str | None = None) -> CheckResult:
    try:
        from vampo.reward.videomae_reward import resolve_videomae_backbone

        path, _ = resolve_videomae_backbone(hf_model_id)
        cfg = Path(path) / "config.json"
        if not cfg.is_file():
            return _fail(
                "videomae_backbone",
                f"本地 backbone 缺少 config.json: {path}",
                "确认 /home/robotem/Models/videomae-base 已完整下载",
            )
        return _ok("videomae_backbone", path)
    except FileNotFoundError as exc:
        return _fail(
            "videomae_backbone",
            str(exc),
            "export VIDEOMAE_BACKBONE=/home/robotem/Models/videomae-base",
        )
    except Exception as exc:
        return _fail("videomae_backbone", f"无法解析 backbone: {exc}")


def check_videomae_smoke(
    checkpoint_path: str,
    *,
    hf_model_id: str | None = None,
    device: str | None = None,
) -> CheckResult:
    try:
        import numpy as np
        from vampo.reward.factory import build_reward_model

        reward_cfg: dict = {
            "backend": "videomae",
            "videomae_checkpoint": checkpoint_path,
            "device": device or ("cuda" if _cuda_available() else "cpu"),
            "batch_size": 8,
        }
        if hf_model_id:
            reward_cfg["hf_model_id"] = hf_model_id
        rm = build_reward_model({"reward": reward_cfg})
        video = np.random.randint(0, 255, (40, 128, 128, 3), dtype=np.uint8)
        result = rm.predict_success(video)
        return _ok(
            "videomae_smoke",
            f"predict_success OK · complete={result.complete} finish_frame={result.finish_step}",
        )
    except Exception as exc:
        return _fail(
            "videomae_smoke",
            f"VideoMAE reward 冒烟失败: {exc}",
            "确认 checkpoint 与 transformers VideoMAE 依赖可用",
        )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def run_preflight(
    *,
    root: Path,
    config_path: Path,
    model_path: Path,
    droid_root: Path,
    init_dir: Path,
    split_dir: Path,
    skip_reward: bool = False,
    strict_reward: bool = False,
) -> list[CheckResult]:
    reward_settings = _load_reward_settings(config_path)
    reward_backend = reward_settings["backend"]

    results: list[CheckResult] = [
        check_python_imports(),
        check_cuda(),
        check_config(config_path),
        check_vla_model(model_path),
        check_droid_data(droid_root),
        check_episode_split(split_dir),
        check_init_states(init_dir),
        check_init_overlap(init_dir, split_dir),
    ]

    if skip_reward:
        results.append(
            _warn(
                "reward_smoke",
                f"已跳过（--skip-reward）· backend={reward_backend}",
                "训练前请运行: bash scripts/preflight_rl.sh --strict-reward",
            )
        )
        return results

    backbone = reward_settings.get("hf_model_id")
    results.append(check_videomae_backbone(str(backbone) if backbone else None))
    ckpt = reward_settings.get("videomae_checkpoint")
    ckpt_check = check_videomae_checkpoint(str(ckpt) if ckpt else None)
    results.append(ckpt_check)
    if strict_reward:
        if ckpt_check.status == "fail":
            results.append(
                _fail("videomae_smoke", "跳过冒烟（checkpoint 未就绪）", ckpt_check.hint)
            )
        else:
            results.append(
                check_videomae_smoke(
                    str(ckpt),
                    hf_model_id=str(backbone) if backbone else None,
                )
            )
    return results


def summarize(results: list[CheckResult]) -> tuple[bool, bool]:
    """Return (ready_for_prep, ready_for_train)."""
    blocking = [r for r in results if r.status == "fail"]
    prep_ok = len(blocking) == 0
    reward_checks = [r for r in results if r.name.startswith("videomae")]
    train_ok = prep_ok and bool(reward_checks) and all(r.status == "ok" for r in reward_checks)
    if not reward_checks:
        train_ok = prep_ok
    return prep_ok, train_ok


def print_report(results: list[CheckResult], *, strict_reward: bool) -> None:
    icons = {"ok": "✓", "warn": "!", "fail": "✗"}
    for r in results:
        line = f"  [{icons[r.status]}] {r.name}: {r.message}"
        print(line)
        if r.hint and r.status != "ok":
            print(f"      → {r.hint}")

    prep_ok, train_ok = summarize(results)
    print()
    if train_ok:
        print("结论: 可以启动 RL 训练 → bash scripts/train_component3_rl_cluster4.sh")
    elif prep_ok and not strict_reward:
        print("结论: 本地数据/环境就绪；reward 服务未做 strict 检查")
        print("      就绪后运行: bash scripts/preflight_rl.sh --strict-reward")
        print("      然后启动:   bash scripts/train_component3_rl_cluster4.sh")
    else:
        print("结论: 尚有阻塞项，请按上方提示修复")


def main() -> None:
    root = Path(os.environ.get("VAMVERL_ROOT", Path(__file__).resolve().parents[3]))
    p = argparse.ArgumentParser(description="VamVerl RL training preflight checks")
    p.add_argument("--config", default=str(root / "configs/vampo_ppo_trainer.yaml"))
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/home/robotem/Models/DreamZero-DROID"))
    p.add_argument("--droid-root", default=os.environ.get("DROID_DATA_ROOT", "/home/robotem/DATA/droid_lerobot"))
    p.add_argument("--init-dir", default=os.environ.get("INIT_STATES_DIR", str(root / "data/init_states")))
    p.add_argument("--split-dir", default=os.environ.get("DROID_SPLIT_DIR", str(root / "data/splits")))
    p.add_argument(
        "--skip-reward",
        action="store_true",
        help="跳过 VideoMAE reward 检查",
    )
    p.add_argument(
        "--strict-reward",
        action="store_true",
        help="要求 VideoMAE backbone + checkpoint 冒烟测试通过（训练前必跑）",
    )
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args()

    results = run_preflight(
        root=root,
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        droid_root=Path(args.droid_root),
        init_dir=Path(args.init_dir),
        split_dir=Path(args.split_dir),
        skip_reward=args.skip_reward,
        strict_reward=args.strict_reward,
    )

    if args.json:
        prep_ok, train_ok = summarize(results)
        print(
            json.dumps(
                {
                    "ready_for_prep": prep_ok,
                    "ready_for_train": train_ok,
                    "checks": [asdict(r) for r in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print("=== VamVerl RL Preflight ===")
        print_report(results, strict_reward=args.strict_reward)

    _, train_ok = summarize(results)
    if args.strict_reward and not train_ok:
        sys.exit(1)
    prep_ok, _ = summarize(results)
    if not prep_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
