"""Hydra entry for VAMPO + verl GRPO/PPO training."""

from __future__ import annotations

import json
import os

import ray
from omegaconf import OmegaConf


class _DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0


@ray.remote
def main_task(config):
    from pprint import pprint

    from verl.trainer.ppo.ray_trainer import RayTrainer, ResourcePoolManager, Role
    from vampo.integrations.verl.reward_manager import VAMPORewardManager
    from vampo.integrations.verl.worker import VAMPOActorRolloutRefWorker
    from vampo.integrations.verl.ray_worker_group import VAMPORayWorkerGroup

    ray_worker_group_cls = VAMPORayWorkerGroup

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    tokenizer = _DummyTokenizer()

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(VAMPOActorRolloutRefWorker),
    }
    if config.algorithm.kl_ctrl.kl_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(VAMPOActorRolloutRefWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {Role.ActorRollout: global_pool_id}
    if Role.RefPolicy in role_worker_mapping:
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = VAMPORewardManager(num_examine=0, config=config)
    val_reward_fn = VAMPORewardManager(num_examine=1, config=config)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


def _resolve_vampo_paths(config) -> None:
    """Use absolute paths from env so Ray tasks/workers don't rely on Hydra cwd."""
    root = os.path.abspath(os.environ.get("VAMVERL_ROOT", os.getcwd()))
    init_dir = os.path.abspath(
        os.environ.get("INIT_STATES_DIR", os.path.join(root, "data", "init_states"))
    )
    OmegaConf.update(config, "data.init_states_dir", init_dir, force_add=False)
    if OmegaConf.select(config, "actor_rollout_ref.data") is not None:
        OmegaConf.update(config, "actor_rollout_ref.data.init_states_dir", init_dir, force_add=False)

    rank0_ip = os.environ.get("VAMPO_RANK0_NODE_IP") or OmegaConf.select(
        config, "trainer.rank0_node_ip", default=None
    )
    if rank0_ip:
        os.environ.setdefault("VAMPO_RANK0_NODE_IP", str(rank0_ip))

    videomae_ckpt = os.environ.get("VIDEOMAE_CKPT")
    if videomae_ckpt and OmegaConf.select(config, "actor_rollout_ref.reward") is not None:
        OmegaConf.update(
            config, "actor_rollout_ref.reward.videomae_checkpoint", videomae_ckpt, force_add=False
        )
    videomae_backbone = os.environ.get("VIDEOMAE_BACKBONE")
    if videomae_backbone and OmegaConf.select(config, "actor_rollout_ref.reward") is not None:
        OmegaConf.update(
            config, "actor_rollout_ref.reward.hf_model_id", videomae_backbone, force_add=False
        )


def _build_runtime_env(config) -> dict:
    root = os.path.abspath(os.environ.get("VAMVERL_ROOT", os.getcwd()))
    runtime_env_path = str(config.trainer.runtime_env)
    if runtime_env_path not in ("none", ""):
        if not os.path.isabs(runtime_env_path):
            candidate = os.path.join(root, runtime_env_path)
            if os.path.isfile(candidate):
                runtime_env_path = candidate
    if runtime_env_path not in ("none", "") and os.path.isfile(runtime_env_path):
        with open(runtime_env_path, "r") as f:
            runtime_env = json.load(f)
    else:
        runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
    env_vars = runtime_env.setdefault("env_vars", {})
    vamverl_root = os.environ.get("VAMVERL_ROOT")
    if vamverl_root and os.path.isdir(vamverl_root):
        env_vars.setdefault("VAMVERL_ROOT", vamverl_root)
        env_vars["PYTHONPATH"] = os.pathsep.join(
            p for p in (vamverl_root, env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))) if p
        )
    for key in (
        "PYTHONPATH",
        "VAMVERL_ROOT",
        "MODEL_PATH",
        "WAN21_DIR",
        "WAN22_DIR",
        "TOKENIZER_PATH",
        "HF_HUB_OFFLINE",
        "DROID_DATA_ROOT",
        "INIT_STATES_DIR",
        "DROID_SPLIT_DIR",
        "VIDEOMAE_CKPT",
        "VIDEOMAE_BACKBONE",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_MODE",
        "VAMPO_NCCL_TIMEOUT_MIN",
        "VAMPO_RANK0_NODE_IP",
        "RAY_HEAD_IP",
        "HEAD_IP",
    ):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val
    return runtime_env


def main(config):
    _resolve_vampo_paths(config)
    if not ray.is_initialized():
        runtime_env = _build_runtime_env(config)
        address = os.environ.get("RAY_ADDRESS")
        if address and address not in ("local", ""):
            ray.init(address=address, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)
    ray.get(main_task.remote(config))


def main_hydra() -> None:
    """Console entry point for ``vampo-train`` (verl GRPO+PPO)."""
    import hydra

    @hydra.main(config_path="../../../configs", config_name="vampo_ppo_trainer", version_base=None)
    def hydra_main(cfg):
        main(cfg)

    hydra_main()


if __name__ == "__main__":
    main_hydra()
