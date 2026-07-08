"""verl rollout using ImaginationRollout + in-process VLA policy."""

from __future__ import annotations

import torch

from vampo.imagination.policy.runner import PolicyRunner
from vampo.integrations.verl.proto_adapter import trajectories_to_dataproto
from vampo.integrations.verl.vla_backend import InProcessVLABackend
from vampo.integrations.verl.vla_policy import VLAPolicyModule
from vampo.imagination.rollout import ImaginationRollout, InitStateStore
from vampo.reward.base import BaseRewardModel
from verl import DataProto
from verl.workers.rollout.base import BaseRollout


def _build_rollout_backend(config: dict, policy_module: VLAPolicyModule | None = None):
    backend_name = config.get("policy_backend", "vla")
    if backend_name == "vla":
        if policy_module is None:
            raise ValueError("policy_backend=vla requires shared VLAPolicyModule")
        return InProcessVLABackend(policy_module, prompt=config.get("task_prompt", ""))
    raise ValueError(
        f"Unknown rollout.policy_backend={backend_name!r}; use vla (in-process DreamZero)"
    )


class VAMPORollout(BaseRollout):
    """Rollout imagined trajectories via VLA and emit verl DataProto batches."""

    def __init__(
        self,
        config,
        init_states_dir: str,
        reward_model: BaseRewardModel | None = None,
        policy_module: VLAPolicyModule | None = None,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.init_store = InitStateStore(init_states_dir)
        if len(self.init_store) == 0:
            raise RuntimeError(
                f"No init states under {init_states_dir}; "
                "run: python -m vampo.data.init_states_bootstrap"
            )
        if reward_model is None:
            raise RuntimeError(
                "Reward model required for verl training; configure reward.videomae_checkpoint."
            )
        self.action_horizon = int(config.get("action_horizon", 8))
        self.action_dim = int(config.get("action_dim", 8))
        self.max_wm_steps = int(config.get("max_wm_steps", 8))
        self.primary_camera_key = config.get("primary_camera_key")
        self.task_prompt = config.get("task_prompt", "")

        backend = _build_rollout_backend(config, policy_module=policy_module)
        runner = PolicyRunner(backend)
        self.rollout = ImaginationRollout(
            policy=runner,
            reward_model=reward_model,
            max_wm_steps=self.max_wm_steps,
            primary_camera_key=self.primary_camera_key,
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        meta = prompts.meta_info
        n_samples = int(meta.get("n_samples", 1))
        init_indices = prompts.batch["init_index"].cpu().reshape(-1).tolist()

        trajectories = []
        for init_index in init_indices:
            state_id, obs, prompt = self.init_store.get(int(init_index))
            prompt = prompt or self.task_prompt
            trajectories.extend(
                self.rollout.rollout_group(obs, prompt, state_id, n_samples)
            )

        proto = trajectories_to_dataproto(
            trajectories,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            reuse_trace_log_prob=bool(self.config.get("reuse_trace_log_prob", False)),
            action_sigma=float(
                getattr(self.policy_module, "flow_rl_sigma", None)
                or self.config.get("flow_rl_sigma", 0.05)
            ),
            video_sigma=float(
                getattr(self.policy_module, "flow_rl_video_sigma", None)
                or self.config.get("flow_rl_video_sigma", 0.05)
            ),
        )
        proto.batch["state_id"] = prompts.batch["state_id"].repeat_interleave(n_samples, dim=0)
        return proto
