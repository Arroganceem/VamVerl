"""Staged verification checks (no full RL training / 14B load required)."""

from __future__ import annotations

import compileall
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.preflight.dreamzero_preflight import (
    CheckResult,
    _fail,
    _ok,
    _warn,
    check_config,
    check_cuda,
    check_droid_data,
    check_episode_split,
    check_init_overlap,
    check_init_states,
    check_python_imports,
    check_videomae_backbone,
    check_videomae_checkpoint,
    check_videomae_smoke,
    check_vla_model,
    run_preflight,
)
from verl.utils.vla.proto_adapter import trajectories_to_dataproto
from verl.trainer.ppo.dreamzero_reward_manager import DreamZeroRewardManager
from verl.trainer.ppo.dreamzero_grpo import compute_dreamzero_grpo_from_batch
from verl.utils.vla.log_prob_utils import (
    apply_recomputed_log_prob_fields,
    build_rollout_log_prob_tensors,
    log_probs_degenerate,
    rebuild_log_prob_fields_from_old,
)
from verl.utils.vla.trajectory import ChunkRecord, Trajectory

STAGE_NAMES = {
    0: "static",
    1: "data",
    2: "rl-mock",
    3: "config",
    4: "videomae",
    5: "preflight-strict",
}

DREAMZERO_IMPORT_MODULES = (
    "verl.utils.reward.base",
    "verl.utils.reward.factory",
    "verl.utils.reward.wmpo_sliding",
    "verl.utils.reward.videomae_reward",
    "verl.utils.vla.log_prob_utils",
    "verl.utils.vla.proto_adapter",
    "verl.trainer.ppo.dreamzero_grpo",
    "verl.trainer.ppo.dreamzero_reward_manager",
    "verl.trainer.preflight.dreamzero_preflight",
    "verl.trainer.main_dreamzero_ppo",
    "verl.workers.rollout.imagination.rollout",
)

VAMPO_IMPORT_MODULES = DREAMZERO_IMPORT_MODULES  # legacy alias


def _make_mock_trajectory(
    *,
    uid: str,
    init_state_id: str,
    flow_log_probs: list[float],
    complete: bool = False,
    finish_step: int = 1,
) -> Trajectory:
    chunks: list[ChunkRecord] = []
    for lp in flow_log_probs:
        chunks.append(
            ChunkRecord(
                obs={"state": np.zeros(8, dtype=np.float32)},
                action=np.zeros((4, 7), dtype=np.float32),
                video_frames=np.zeros((4, 64, 64, 3), dtype=np.uint8),
                flow_log_prob=lp,
            )
        )
    return Trajectory(
        init_state_id=init_state_id,
        prompt="pick up the object",
        uid=uid,
        chunks=chunks,
        complete=complete,
        finish_step=finish_step,
    )


def check_static_compile(root: Path) -> CheckResult:
    targets = [root / "verl"]
    if (root / "eval_utils").is_dir():
        targets.append(root / "eval_utils")
    failed: list[str] = []
    for target in targets:
        ok = compileall.compile_dir(
            str(target),
            quiet=1,
            force=False,
            legacy=False,
        )
        if not ok:
            failed.append(target.name)
    if failed:
        return _fail(
            "static_compile",
            f"compileall 失败: {', '.join(failed)}",
            f"python -m compileall {root / 'verl'}",
        )
    return _ok("static_compile", f"compileall OK · {', '.join(t.name for t in targets)}")


def check_static_imports() -> CheckResult:
    missing: list[str] = []
    for mod in VAMPO_IMPORT_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{mod} ({exc})")
    if missing:
        return _fail(
            "static_imports",
            f"导入失败 {len(missing)} 项: {missing[0]}",
            "pip install -e \".[verl,vla]\"",
        )
    return _ok("static_imports", f"{len(VAMPO_IMPORT_MODULES)} 个核心模块可导入")


def check_rl_pipeline_mock() -> CheckResult:
    """Mock rollout → DataProto → sparse reward → GRPO (CPU, no VLA)."""
    try:
        action_horizon = 2
        action_dim = 4
        action_flat = action_horizon * action_dim
        max_wm = 2

        trajs = [
            _make_mock_trajectory(
                uid="group-a",
                init_state_id="s0",
                flow_log_probs=[-1.2, -0.8],
                complete=False,
                finish_step=2,
            ),
            _make_mock_trajectory(
                uid="group-a",
                init_state_id="s0",
                flow_log_probs=[-2.5, -1.1],
                complete=False,
                finish_step=2,
            ),
            _make_mock_trajectory(
                uid="group-b",
                init_state_id="s1",
                flow_log_probs=[-0.5, -0.3],
                complete=True,
                finish_step=2,
            ),
            _make_mock_trajectory(
                uid="group-b",
                init_state_id="s1",
                flow_log_probs=[-1.0, -0.9],
                complete=False,
                finish_step=2,
            ),
        ]

        built = build_rollout_log_prob_tensors(trajs, max_wm, action_flat, device="cpu")
        if built is None:
            return _fail("rl_pipeline_mock", "build_rollout_log_prob_tensors 返回 None")
        per_step, old_log_probs, scalar = built
        if log_probs_degenerate(old_log_probs, scalar):
            return _fail("rl_pipeline_mock", "mock log_prob 被判定为 degenerate")

        proto = trajectories_to_dataproto(
            trajs,
            action_horizon=action_horizon,
            action_dim=action_dim,
            feat_dim=16,
            device="cpu",
        )
        # verl hybrid: actor.compute_log_prob fills old_log_probs after rollout
        proto.batch["old_log_probs"] = old_log_probs
        apply_recomputed_log_prob_fields(proto)
        proto.non_tensor_batch["uid"] = np.array([t.uid for t in trajs], dtype=object)

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"model": {"action_token_len": action_flat}},
                "verifier": {"reward_coef": 1.0},
            }
        )
        rm = DreamZeroRewardManager(num_examine=0, config=config)
        reward_dict, _metrics = rm(proto)
        proto.batch["token_level_rewards"] = reward_dict["all"]

        adv, _ret = compute_dreamzero_grpo_from_batch(proto, config)
        adv_flat = adv.sum(dim=-1).detach().cpu().numpy()
        if not np.all(np.isfinite(adv_flat)):
            return _fail("rl_pipeline_mock", "GRPO advantage 含 NaN/Inf")

        # group-a: tied reward (0) → tie-break by log_prob → non-zero spread
        spread_a = float(adv_flat[0] - adv_flat[1])
        if abs(spread_a) < 1e-6:
            return _fail(
                "rl_pipeline_mock",
                "group-a tie-break 未产生 advantage 分化 "
                f"(adv={adv_flat[:2].tolist()})",
            )

        # group-b: reward 0 vs 1 → should differ
        if abs(float(adv_flat[2]) - float(adv_flat[3])) < 1e-6:
            return _fail(
                "rl_pipeline_mock",
                "group-b reward 分化未反映到 advantage",
            )

        recovered, recovered_scalar = rebuild_log_prob_fields_from_old(
            old_log_probs, max_wm, action_flat
        )
        if not torch.allclose(recovered, per_step) or not torch.allclose(
            recovered_scalar, scalar
        ):
            return _fail("rl_pipeline_mock", "rebuild_log_prob_fields_from_old 不一致")

        return _ok(
            "rl_pipeline_mock",
            f"proto/reward/GRPO OK · group-a adv_spread={spread_a:.4f} "
            f"log_prob_scalar={scalar.tolist()}",
        )
    except Exception as exc:
        return _fail("rl_pipeline_mock", f"RL mock 链路异常: {exc}")


def check_config_schema(config_path: Path) -> CheckResult:
    if not config_path.is_file():
        return _fail("config_schema", f"配置不存在: {config_path}")
    try:
        cfg = OmegaConf.load(config_path)
        required = [
            "vla.model_path",
            "actor_rollout_ref.reward.backend",
            "actor_rollout_ref.rollout.max_wm_steps",
            "algorithm.adv_estimator",
            "trainer.n_gpus_per_node",
        ]
        missing = [k for k in required if OmegaConf.select(cfg, k) is None]
        if missing:
            return _fail(
                "config_schema",
                f"缺少字段: {', '.join(missing)}",
            )
        adv = str(OmegaConf.select(cfg, "algorithm.adv_estimator")).lower()
        if adv not in {"grpo", "gae", "vampo_grpo"}:
            return _warn(
                "config_schema",
                f"adv_estimator={adv!r} 非典型 GRPO 配置",
            )
        backend = str(OmegaConf.select(cfg, "actor_rollout_ref.reward.backend")).lower()
        if backend not in {"videomae", "video_mae"}:
            return _warn(
                "config_schema",
                f"reward.backend={backend!r} 当前仅 videomae 在 factory 中实现",
            )
        return _ok(
            "config_schema",
            f"{config_path.name} · adv={adv} · reward={backend}",
        )
    except Exception as exc:
        return _fail("config_schema", str(exc))


def check_hydra_entry(root: Path) -> CheckResult:
    try:
        from verl.trainer.main_dreamzero_ppo import main_hydra

        if not callable(main_hydra):
            return _fail("hydra_entry", "main_hydra 不可调用")
        return _ok("hydra_entry", "main_dreamzero_ppo.main_hydra 可导入")
    except Exception as exc:
        return _fail(
            "hydra_entry",
            f"训练入口导入失败: {exc}",
            "pip install -e \".[verl,vla]\"",
        )


def run_stage(
    stage: int,
    *,
    root: Path,
    config_path: Path,
    model_path: Path,
    droid_root: Path,
    init_dir: Path,
    split_dir: Path,
    videomae_device: str | None = None,
) -> list[CheckResult]:
    if stage == 0:
        return [check_static_compile(root), check_static_imports(), check_python_imports()]
    if stage == 1:
        return [
            check_cuda(),
            check_config(config_path),
            check_vla_model(model_path),
            check_droid_data(droid_root),
            check_episode_split(split_dir),
            check_init_states(init_dir),
            check_init_overlap(init_dir, split_dir),
        ]
    if stage == 2:
        return [check_rl_pipeline_mock()]
    if stage == 3:
        return [check_config_schema(config_path), check_hydra_entry(root)]
    if stage == 4:
        from verl.trainer.preflight.dreamzero_preflight import _load_reward_settings

        settings = _load_reward_settings(config_path)
        backbone = settings.get("hf_model_id")
        ckpt = settings.get("videomae_checkpoint")
        results = [
            check_videomae_backbone(str(backbone) if backbone else None),
            check_videomae_checkpoint(str(ckpt) if ckpt else None),
        ]
        ckpt_res = results[-1]
        if ckpt_res.status == "ok":
            results.append(
                check_videomae_smoke(
                    str(ckpt),
                    hf_model_id=str(backbone) if backbone else None,
                    device=videomae_device or "cpu",
                )
            )
        return results
    if stage == 5:
        return run_preflight(
            root=root,
            config_path=config_path,
            model_path=model_path,
            droid_root=droid_root,
            init_dir=init_dir,
            split_dir=split_dir,
            skip_reward=False,
            strict_reward=True,
        )
    raise ValueError(f"Unknown stage {stage}")


def run_stages(
    stages: list[int],
    *,
    root: Path,
    config_path: Path,
    model_path: Path,
    droid_root: Path,
    init_dir: Path,
    split_dir: Path,
    videomae_device: str | None = None,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for stage in stages:
        results.extend(
            run_stage(
                stage,
                root=root,
                config_path=config_path,
                model_path=model_path,
                droid_root=droid_root,
                init_dir=init_dir,
                split_dir=split_dir,
                videomae_device=videomae_device,
            )
        )
    return results


def summarize_stages(
    results: list[CheckResult],
    stages_run: list[int] | None = None,
) -> tuple[bool, bool]:
    """Return (github_ready, train_ready)."""
    blocking = [r for r in results if r.status == "fail"]
    github_ready = len(blocking) == 0
    if not github_ready or not stages_run:
        return github_ready, False

    max_stage = max(stages_run)
    if max_stage >= 5:
        return github_ready, True

    train_required = {
        "vla_model",
        "droid_data",
        "episode_split",
        "init_states",
        "videomae_smoke",
    }
    ok_names = {r.name for r in results if r.status == "ok"}
    if max_stage >= 4 and train_required.issubset(ok_names):
        return github_ready, True
    return github_ready, False


def print_staged_report(
    results: list[CheckResult],
    *,
    stages_run: list[int],
) -> None:
    icons = {"ok": "✓", "warn": "!", "fail": "✗"}
    print(f"阶段: {', '.join(f'{s}={STAGE_NAMES[s]}' for s in stages_run)}")
    print()
    for r in results:
        print(f"  [{icons[r.status]}] {r.name}: {r.message}")
        if r.hint and r.status != "ok":
            print(f"      → {r.hint}")
    github_ready, train_ready = summarize_stages(results, stages_run)
    print()
    if train_ready:
        print("结论: 代码 + 数据 + VideoMAE 就绪 → 可启动 cluster 训练")
    elif github_ready:
        print("结论: 代码完整性验证通过（GitHub 可上传 / CI 可绿）")
        print("      开训前补跑: bash scripts/preflight/verify_staged.sh --through 5")
    else:
        print("结论: 存在阻塞项，请按上方提示修复后再上传或开训")
