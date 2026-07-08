"""verl worker: in-process VLA rollout + PPO (FSDP distributed)."""

from __future__ import annotations

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from vampo.integrations.verl.actor import VAMPODPOActor
from vampo.integrations.verl.paths import resolve_tokenizer_path
from vampo.integrations.verl.fsdp_utils import (
    fsdp_enabled,
    fsdp_post_initialize,
    init_fsdp_device_mesh,
    prepare_vla_for_fsdp_wrap,
    wrap_vla_fsdp,
)
from vampo.integrations.verl.log_prob_utils import (
    apply_recomputed_log_prob_fields,
    log_probs_degenerate,
    log_rollout_log_prob_summary,
)
from vampo.integrations.verl.parallel_utils import set_parallel_strategy_env
from vampo.integrations.verl.rollout import VAMPORollout
from vampo.integrations.verl.vla_policy import VLAPolicyModule
from vampo.reward.factory import build_reward_model
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, dispatch_one_to_all, register
from verl.utils.debug import log_gpu_memory_usage
from verl.workers.hybrid_engine.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def _collect_fsdp_compute_proto(worker_group, output):
    """FSDP ranks run identical compute; return rank-0 payload (avoid N× concat)."""
    del worker_group
    return output[0] if output else output


_VAMPO_FSDP_COMPUTE_PROTO = {
    "dispatch_fn": dispatch_one_to_all,
    "collect_fn": _collect_fsdp_compute_proto,
}


class _DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0


class VAMPOActorRolloutRefWorker(Worker):
    """Actor + rollout worker: shared VLAPolicyModule for rollout and PPO."""

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        nccl_timeout_min = int(os.environ.get("VAMPO_NCCL_TIMEOUT_MIN", "120"))
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            init_kwargs: dict = {
                "backend": backend,
                "timeout": timedelta(minutes=nccl_timeout_min),
            }
            if backend == "nccl" and torch.cuda.is_available():
                init_kwargs["device_id"] = torch.cuda.current_device()
            dist.init_process_group(**init_kwargs)
            print(
                f"[dist] rank{dist.get_rank()}: init_process_group "
                f"backend={backend} timeout={nccl_timeout_min}min",
                flush=True,
            )
        else:
            print(
                f"[dist] rank{dist.get_rank()}: process group already initialized "
                f"(VAMPO_NCCL_TIMEOUT_MIN={nccl_timeout_min} ignored)",
                flush=True,
            )

        set_parallel_strategy_env(config)
        self._use_fsdp = fsdp_enabled(config)
        self.device_mesh = init_fsdp_device_mesh() if self._use_fsdp else None

        # FSDP: all ranks run the same batch (no DP shard); keep yaml batch sizes as-is.

        self.tokenizer = _DummyTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_model_path(self) -> str:
        model_cfg = self.config.model
        path = model_cfg.get("path")
        if path and path != "vla":
            return str(path)
        vla_path = OmegaConf.select(self.config, "vla.model_path", default=None)
        if vla_path:
            return str(vla_path)
        return os.environ.get(
            "MODEL_PATH", "/home/robotem/Models/DreamZero-DROID"
        )

    def _build_policy(self) -> torch.nn.Module:
        model_cfg = self.config.model
        if model_cfg.get("backend", "vla") != "vla":
            raise ValueError(
                f"Unsupported model.backend={model_cfg.get('backend')!r}; only 'vla' is supported"
            )

        rollout_cfg = OmegaConf.to_container(self.config.rollout, resolve=True)
        rl_mode = model_cfg.get("rl_fine_tune_mode")
        if rl_mode is None:
            rl_mode = "lora" if model_cfg.get("keep_lora_trainable", False) else "full"
        keep_lora = model_cfg.get("keep_lora_trainable")
        tokenizer_path = resolve_tokenizer_path(
            model_cfg.get("tokenizer_path")
            or OmegaConf.select(self.config, "vla.tokenizer_path")
        )
        return VLAPolicyModule(
            model_path=self._resolve_model_path(),
            device=self.device,
            action_horizon=int(model_cfg.get("action_horizon", 8)),
            action_dim=int(model_cfg.get("action_dim", 8)),
            imagined_frames=int(rollout_cfg.get("imagined_frames", 8)),
            keep_lora_trainable=keep_lora if keep_lora is not None else None,
            rl_fine_tune_mode=str(rl_mode),
            tune_projector=bool(model_cfg.get("tune_projector", True)),
            tune_diffusion_model=bool(model_cfg.get("tune_diffusion_model", True)),
            primary_camera_key=rollout_cfg.get("primary_camera_key"),
            flow_rl_sigma=model_cfg.get("flow_rl_sigma"),
            flow_rl_video_sigma=model_cfg.get("flow_rl_video_sigma"),
            lazy_load=self._use_fsdp,
            defer_post_initialize=self._use_fsdp,
            tokenizer_path_override=tokenizer_path,
        )

    def _maybe_wrap_fsdp(self, policy: torch.nn.Module) -> torch.nn.Module:
        if not self._use_fsdp or not isinstance(policy, VLAPolicyModule):
            return policy
        fsdp_cfg = OmegaConf.to_container(self.config.actor.fsdp_config, resolve=True)
        vla = policy.groot.trained_model
        prepare_vla_for_fsdp_wrap(vla)
        dist.barrier()
        policy.groot.trained_model = wrap_vla_fsdp(
            vla,
            fsdp_config=fsdp_cfg,
            device_mesh=self.device_mesh,
        )
        fsdp_post_initialize(policy.groot.trained_model)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After VLA FSDP wrap", logger=logger)
        return policy

    def _build_reward_model(self):
        return build_reward_model(self.config)

    def _rollout_reward_model(self):
        if not self._is_rollout or self.rollout is None:
            return None
        return self.rollout.rollout.reward_model

    def _offload_rollout_reward(self) -> None:
        rm = self._rollout_reward_model()
        if rm is not None and hasattr(rm, "offload"):
            rm.offload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cuda_cleanup(self, tag: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            log_gpu_memory_usage(tag, logger=logger)

    def _build_rollout(self, policy_module: torch.nn.Module):
        rollout_cfg = OmegaConf.to_container(self.config.rollout, resolve=True)
        rollout_cfg["device"] = str(self.device)
        vla_module = policy_module if isinstance(policy_module, VLAPolicyModule) else None
        rollout = VAMPORollout(
            config=rollout_cfg,
            init_states_dir=self.config.data.init_states_dir,
            reward_model=self._build_reward_model(),
            policy_module=vla_module,
        )
        return rollout, BaseShardingManager()

    def _optimizer_params(self, policy: torch.nn.Module):
        return policy.trainable_parameters_list()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        policy = self._build_policy()
        if isinstance(policy, VLAPolicyModule):
            trainable = policy.trainable_parameters_list()
            n_trainable = sum(p.numel() for p in trainable)
            print(
                f"VLA RL setup: {len(trainable)} trainable tensors, "
                f"{n_trainable:,} parameters (rl_mode={policy.rl_fine_tune_mode})",
                flush=True,
            )
            if n_trainable == 0 and self._is_actor:
                raise RuntimeError(
                    "No trainable parameters after RL setup; "
                    "check rl_fine_tune_mode / keep_lora_trainable."
                )
        policy = self._maybe_wrap_fsdp(policy)
        rl_ckpt = self.config.model.get("rl_checkpoint")
        if rl_ckpt and isinstance(policy, VLAPolicyModule) and os.path.isfile(str(rl_ckpt)):
            policy.load_rl_checkpoint(str(rl_ckpt))
        self.actor_module = policy
        self.actor_module_fsdp = policy

        if self._is_actor:
            optim_cfg = self.config.actor.optim
            self.actor_optimizer = torch.optim.AdamW(
                self._optimizer_params(policy),
                lr=float(optim_cfg.lr),
                weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
            )
            self.actor_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor_optimizer, factor=1.0)
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = VAMPODPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout(policy)

        if self._is_ref:
            ref_policy = self._build_policy()
            ref_policy = self._maybe_wrap_fsdp(ref_policy)
            self.ref_module_fsdp = ref_policy
            ref_params = self._optimizer_params(self.ref_module_fsdp)
            ref_opt = torch.optim.AdamW(ref_params, lr=1e-5)
            self.ref_policy = VAMPODPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp, actor_optimizer=ref_opt
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()

    @register(dispatch_mode=_VAMPO_FSDP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        data = data.to(self.device)
        metrics = self.actor.update_policy(data=data)
        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr(1e-4)"] = lr * 1e4
        self._cuda_cleanup("After update_actor")
        output = DataProto(meta_info={"metrics": metrics})
        return output.to("cpu")

    @register(dispatch_mode=_VAMPO_FSDP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        assert self._is_actor
        data = data.to(self.device)
        metrics = self.actor.compute_entropy(batch_data=data)
        output = DataProto(meta_info={"metrics": metrics})
        return output.to("cpu")

    @register(dispatch_mode=_VAMPO_FSDP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(self.device)
        recompute_log_prob = prompts.meta_info.get("recompute_log_prob", True)

        with self.sharding_manager:
            log_gpu_memory_usage("Before vampo rollout", logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.sharding_manager.postprocess_data(output)

        self._offload_rollout_reward()

        if dist.get_rank() == 0:
            log_rollout_log_prob_summary(
                output.batch.get("rollout_log_probs"),
                output.batch.get("rollout_log_prob_scalar"),
                phase="rollout",
            )

        if self._is_actor and recompute_log_prob:
            output.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info["temperature"] = self.config.rollout.temperature
            output.meta_info["use_dynamic_bsz"] = bool(
                getattr(self.config.rollout, "log_prob_use_dynamic_bsz", False)
            )
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch["old_log_probs"] = old_log_probs
            apply_recomputed_log_prob_fields(output)
            if dist.get_rank() == 0:
                if log_probs_degenerate(
                    output.batch.get("old_log_probs"),
                    output.batch.get("rollout_log_prob_scalar"),
                ):
                    print(
                        "VAMPO ERROR: actor.compute_log_prob degenerate; "
                        "check flow_traces in rollout batch",
                        flush=True,
                    )
                else:
                    log_rollout_log_prob_summary(
                        output.batch["rollout_log_probs"],
                        output.batch["rollout_log_prob_scalar"],
                        phase="recomputed",
                    )

        if dist.get_rank() == 0 and "complete" in output.batch:
            from vampo.integrations.verl.train_progress import log_batch_rewards

            n_samples = int(
                prompts.meta_info.get("n_samples")
                or getattr(self.config.data, "n_samples", 1)
                or 1
            )
            log_batch_rewards(output, phase="worker_rollout", n_samples=n_samples)

        self._cuda_cleanup("After generate_sequences")
        return output.to("cpu")

    @register(dispatch_mode=_VAMPO_FSDP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        data = data.to(self.device)
        ref_log_prob = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob})
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        os.makedirs(local_path, exist_ok=True)
        if dist.get_rank() == 0:
            if isinstance(self.actor_module, VLAPolicyModule):
                self.actor_module.save_rl_checkpoint(os.path.join(local_path, "policy.pt"))
            else:
                torch.save(self.actor_module.state_dict(), os.path.join(local_path, "policy.pt"))
        dist.barrier()
