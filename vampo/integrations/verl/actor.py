"""Continuous-action PPO actor for VAMPO (VLA + LoRA / full fine-tune)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from vampo.integrations.verl.vla_policy import VLAPolicyModule
from verl import DataProto
from verl.workers.actor import BasePPOActor


class VAMPODPOActor(BasePPOActor):
    """PPO actor with flow-matching log-probs on continuous actions."""

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer | None = None):
        super().__init__(config)
        if not isinstance(actor_module, VLAPolicyModule):
            raise TypeError(f"VAMPODPOActor expects VLAPolicyModule, got {type(actor_module)}")
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.action_token_len = int(config.get("action_token_len", 64))

    def _flat_actions(self, responses: Tensor) -> Tuple[Tensor, int]:
        batch_size, traj_len = responses.shape[0], responses.shape[1]
        flat = responses.reshape(batch_size, traj_len, -1)
        return flat, traj_len

    def _response_mask(self, finish_step: Tensor, traj_len: int, action_flat: int, device: torch.device) -> Tensor:
        response_length = traj_len * action_flat
        finish = finish_step * self.action_token_len
        steps = torch.arange(response_length, device=device)
        steps_expanded = steps.unsqueeze(0).expand(finish_step.size(0), -1)
        return steps_expanded < finish.unsqueeze(1)

    def _log_probs_for_batch(
        self,
        data: DataProto,
        actions: Tensor,
        traj_len: int,
        *,
        enable_grad: bool = True,
    ) -> Tensor:
        obs_chunks = data.non_tensor_batch["obs_chunks"]
        batch_size = actions.shape[0]
        prompts = list(data.non_tensor_batch.get("prompts", [""] * batch_size))
        flow_traces = data.non_tensor_batch.get("flow_traces")
        return self.actor_module.log_prob_from_batch(
            obs_chunks,
            actions,
            prompts=prompts,
            flow_traces_batch=flow_traces,
            enable_grad=enable_grad,
        )

    def _ppo_loss_for_micro_batch(
        self,
        data: DataProto,
        batch: dict,
        actions: Tensor,
        traj_len: int,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, action_flat = actions.shape[0], actions.shape[-1]
        response_length = traj_len * action_flat
        finish = batch["finish_step"] * self.action_token_len
        steps = torch.arange(response_length, device=actions.device)
        steps_expanded = steps.unsqueeze(0).expand(batch_size, -1)
        response_mask = steps_expanded < finish.unsqueeze(1)

        new_log_probs = self._log_probs_for_batch(data, actions, traj_len)
        old_log_prob = batch["old_log_probs"]
        advantages = batch["advantages"]

        ratio = torch.exp(new_log_probs - old_log_prob)
        clip_high = self.config.get("clip_ratio_high", 0.28)
        clip_low = self.config.get("clip_ratio_low", 0.2)
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
        pg_loss = torch.max(pg1, pg2)
        pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp_min(1.0)

        ent = self.actor_module.flow_entropy_per_wm_step()
        entropy = torch.full(
            (batch_size, response_length),
            ent / max(action_flat, 1),
            device=actions.device,
            dtype=torch.float32,
        )
        entropy_loss = (entropy * response_mask).sum() / response_mask.sum().clamp_min(1.0)

        return pg_loss, entropy_loss

    def compute_log_prob(self, data: DataProto) -> Tensor:
        """verl HybridEngine: recompute ``old_log_probs`` via actor forward after rollout."""
        self.actor_module.eval()
        micro_batch_size = int(data.meta_info["micro_batch_size"])
        _ = data.meta_info.get("temperature", 1.0)
        if data.meta_info.get("use_dynamic_bsz", False):
            raise NotImplementedError("VAMPO flow log_prob does not support use_dynamic_bsz yet")

        batch_size = int(data.batch.batch_size[0])
        log_probs_lst: list[Tensor] = []
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_data = data.slice(slice(start, end))
            actions, traj_len = self._flat_actions(micro_data.batch["responses"])
            with torch.no_grad():
                log_probs = self._log_probs_for_batch(
                    micro_data, actions, traj_len, enable_grad=False
                )
            log_probs_lst.append(log_probs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(log_probs_lst, dim=0).detach()

    def compute_entropy(self, batch_data: DataProto) -> Dict:
        batch = batch_data.select(batch_keys=["responses", "obs_features", "finish_step"]).batch
        actions, traj_len = self._flat_actions(batch["responses"])
        batch_size, action_flat = actions.shape[0], actions.shape[-1]
        ent = self.actor_module.flow_entropy_per_wm_step()
        entropy = torch.full(
            (batch_size, traj_len * action_flat),
            ent / max(action_flat, 1),
            device=actions.device,
            dtype=torch.float32,
        )
        mask = self._response_mask(
            batch["finish_step"], traj_len, action_flat, entropy.device
        )
        return {"actor/entropy": (entropy * mask).sum() / mask.sum().clamp_min(1.0)}

    def update_policy(self, data: DataProto) -> Dict:
        self.actor_module.train()
        assert self.actor_optimizer is not None

        if "temperature" not in data.meta_info:
            data.meta_info["temperature"] = 1.0

        batch_size = int(data.batch.batch_size[0])
        micro_bs = max(1, int(self.config.get("ppo_micro_batch_size", 1)))
        entropy_coeff = self.config.get("entropy_coeff", 0.001)
        batch_keys = [
            "responses",
            "obs_features",
            "old_log_probs",
            "advantages",
            "finish_step",
        ]
        num_micro_batches = max(1, (batch_size + micro_bs - 1) // micro_bs)

        self.actor_optimizer.zero_grad(set_to_none=True)
        pg_loss_sum = 0.0
        entropy_loss_sum = 0.0
        num_micro = 0

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            micro_data = data.slice(slice(start, end))
            micro_batch = micro_data.select(batch_keys=batch_keys).batch
            actions, traj_len = self._flat_actions(micro_batch["responses"])

            pg_loss, entropy_loss = self._ppo_loss_for_micro_batch(
                micro_data, micro_batch, actions, traj_len
            )
            scaled = (pg_loss - entropy_coeff * entropy_loss) / num_micro_batches
            scaled.backward()

            pg_loss_sum += float(pg_loss.detach())
            entropy_loss_sum += float(entropy_loss.detach())
            num_micro += 1
            print(
                f"VAMPO update micro {num_micro}/{num_micro_batches} "
                f"pg_loss={float(pg_loss.detach()):.6f} "
                f"entropy={float(entropy_loss.detach()):.6f}",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fsdp_vla = None
        if isinstance(self.actor_module.groot.trained_model, FSDP):
            fsdp_vla = self.actor_module.groot.trained_model
        if fsdp_vla is not None:
            grad_norm = fsdp_vla.clip_grad_norm_(max_norm=self.config.get("grad_clip", 1.0))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.trainable_parameters_list(),
                max_norm=self.config.get("grad_clip", 1.0),
            )
        self.actor_optimizer.step()

        return {
            "actor/pg_loss": pg_loss_sum / max(num_micro, 1),
            "actor/entropy_loss": entropy_loss_sum / max(num_micro, 1),
            "actor/grad_norm": float(grad_norm),
            "actor/micro_batches": num_micro,
        }
