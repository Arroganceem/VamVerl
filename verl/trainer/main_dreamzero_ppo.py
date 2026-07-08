"""Hydra entry for DreamZero + verl GRPO/PPO training."""

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
    from verl.trainer.ppo.dreamzero_reward_manager import DreamZeroRewardManager
    from verl.workers.dreamzero_worker import DreamZeroActorRolloutRefWorker
    from verl.single_controller.ray.dreamzero_worker_group import DreamZeroRayWorkerGroup

    ray_worker_group_cls = DreamZeroRayWorkerGroup

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    tokenizer = _DummyTokenizer()

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(DreamZeroActorRolloutRefWorker),
    }
    if config.algorithm.kl_ctrl.kl_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DreamZeroActorRolloutRefWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {Role.ActorRollout: global_pool_id}
    if Role.RefPolicy in role_worker_mapping:
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = DreamZeroRewardManager(num_examine=0, config=config)
    val_reward_fn = DreamZeroRewardManager(num_examine=1, config=config)
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


def _resolve_vamverl_paths(config) -> None:
    """Use absolute paths from env so Ray tasks/workers don't rely on Hydra cwd."""
    from verl.utils.vamverl_env import FSDP_SHARDED_CHECKPOINT, RANK0_NODE_IP, get, setdefault

    root = os.path.abspath(os.environ.get("VAMVERL_ROOT", os.getcwd()))
    init_dir = os.path.abspath(
        os.environ.get("INIT_STATES_DIR", os.path.join(root, "data", "init_states"))
    )
    OmegaConf.update(config, "data.init_states_dir", init_dir, force_add=False)
    if OmegaConf.select(config, "actor_rollout_ref.data") is not None:
        OmegaConf.update(config, "actor_rollout_ref.data.init_states_dir", init_dir, force_add=False)

    rank0_ip = get(RANK0_NODE_IP) or OmegaConf.select(
        config, "trainer.rank0_node_ip", default=None
    )
    if rank0_ip:
        setdefault(RANK0_NODE_IP, str(rank0_ip))

    sharded_ckpt = OmegaConf.select(
        config, "actor_rollout_ref.actor.fsdp_config.sharded_checkpoint_dir", default=None
    )
    if sharded_ckpt:
        setdefault(FSDP_SHARDED_CHECKPOINT, os.path.abspath(str(sharded_ckpt)))

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


def _apply_cluster_runtime_defaults(config, env_vars: dict) -> None:
    """Ray worker env for multi-node cluster (replaces runtime_env JSON file)."""
    from verl.utils.vamverl_env import NCCL_TIMEOUT_MIN, RANK0_NODE_IP, setdefault

    rank0_ip = OmegaConf.select(config, "trainer.rank0_node_ip", default=None)
    nnodes = int(OmegaConf.select(config, "trainer.nnodes", default=1) or 1)
    if not rank0_ip and nnodes <= 1:
        return
    env_vars.setdefault("NCCL_IB_DISABLE", "0")
    env_vars.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env_vars.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env_vars.setdefault(NCCL_TIMEOUT_MIN, "120")
    if rank0_ip:
        setdefault(RANK0_NODE_IP, str(rank0_ip))
        env_vars.setdefault(RANK0_NODE_IP, str(rank0_ip))


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
    _apply_cluster_runtime_defaults(config, env_vars)
    vamverl_root = os.environ.get("VAMVERL_ROOT")
    if vamverl_root and os.path.isdir(vamverl_root):
        env_vars.setdefault("VAMVERL_ROOT", vamverl_root)
        env_vars["PYTHONPATH"] = os.pathsep.join(
            p for p in (vamverl_root, env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))) if p
        )
    from verl.utils.vamverl_env import RAY_RUNTIME_ENV_KEYS, get

    for key in RAY_RUNTIME_ENV_KEYS:
        val = get(key)
        if val:
            env_vars[key] = val
    return runtime_env


def main(config):
    _resolve_vamverl_paths(config)
    if not ray.is_initialized():
        runtime_env = _build_runtime_env(config)
        address = os.environ.get("RAY_ADDRESS")
        if address and address not in ("local", ""):
            ray.init(address=address, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)
    ray.get(main_task.remote(config))


def main_hydra() -> None:
    """Console entry point for ``vamverl-train`` / ``vampo-train`` (verl GRPO+PPO)."""
    import hydra

    @hydra.main(config_path="../../configs", config_name="vampo_ppo_trainer", version_base=None)
    def hydra_main(cfg):
        main(cfg)

    hydra_main()


if __name__ == "__main__":
    main_hydra()
