"""In-process VLA policy with joint action+video flow chain log-prob for PPO."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tianshou.data import Batch

from verl.workers.rollout.imagination.policy.obs_utils import convert_rl_obs_to_vla_obs
from verl.utils.vla.flow_log_prob import compute_joint_flow_log_prob_from_paths
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


@dataclass
class FlowJointTrace:
    """Joint action + video latent flow paths from one WM step."""

    action_path: torch.Tensor  # [K+1, B, H, D]
    action_eps: torch.Tensor  # [K, B, H, D]
    video_path: torch.Tensor  # [K+1, B, T, C, H, W]
    video_eps: torch.Tensor  # [K, B, T, C, H, W]

    @staticmethod
    def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
        if t.ndim >= 2 and t.shape[1] == 1:
            return t[:, 0]
        return t

    @staticmethod
    def _restore_batch_dim(t: torch.Tensor) -> torch.Tensor:
        """Re-insert B=1 dim dropped by _squeeze_batch for log_prob replay."""
        t = torch.as_tensor(t)
        if t.ndim == 3:
            return t.unsqueeze(1)
        if t.ndim == 5:
            return t.unsqueeze(1)
        return t

    def with_batch_dim(self) -> FlowJointTrace:
        return FlowJointTrace(
            action_path=self._restore_batch_dim(self.action_path),
            action_eps=self._restore_batch_dim(self.action_eps),
            video_path=self._restore_batch_dim(self.video_path),
            video_eps=self._restore_batch_dim(self.video_eps),
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.action_path.cpu().numpy(),
            self.action_eps.cpu().numpy(),
            self.video_path.cpu().numpy(),
            self.video_eps.cpu().numpy(),
        )

    @classmethod
    def from_numpy(
        cls,
        action_path: np.ndarray,
        action_eps: np.ndarray,
        video_path: np.ndarray,
        video_eps: np.ndarray,
    ) -> FlowJointTrace:
        trace = cls(
            action_path=torch.as_tensor(np.ascontiguousarray(action_path.copy()), dtype=torch.float32),
            action_eps=torch.as_tensor(np.ascontiguousarray(action_eps.copy()), dtype=torch.float32),
            video_path=torch.as_tensor(np.ascontiguousarray(video_path.copy()), dtype=torch.float32),
            video_eps=torch.as_tensor(np.ascontiguousarray(video_eps.copy()), dtype=torch.float32),
        )
        return trace.with_batch_dim()

    @classmethod
    def from_dict(cls, data: dict) -> FlowJointTrace:
        if "video_path" in data and "video_eps" in data:
            return cls.from_numpy(
                data["action_path"] if "action_path" in data else data["path"],
                data["action_eps"] if "action_eps" in data else data["eps"],
                data["video_path"],
                data["video_eps"],
            )
        raise ValueError("flow trace dict must include video_path and video_eps")


# Backward-compatible alias
FlowActionTrace = FlowJointTrace


def _flat_action_from_act_dict(act_dict: dict, action_horizon: int, action_dim: int) -> torch.Tensor:
    joint = None
    gripper = None
    for key, value in act_dict.items():
        if "joint_position" in key:
            joint = value
        elif "gripper" in key:
            gripper = value
    if joint is None:
        return torch.zeros(action_horizon * action_dim, dtype=torch.float32)
    if isinstance(joint, torch.Tensor):
        joint = joint.detach().float().cpu()
    else:
        joint = torch.as_tensor(joint, dtype=torch.float32)
    if joint.ndim == 1:
        joint = joint.unsqueeze(0)
    if gripper is not None:
        if isinstance(gripper, torch.Tensor):
            gripper = gripper.detach().float().cpu()
        else:
            gripper = torch.as_tensor(gripper, dtype=torch.float32)
        if gripper.ndim == 1:
            gripper = gripper.unsqueeze(-1)
        act = torch.cat([joint, gripper], dim=-1)
    else:
        pad = torch.zeros(joint.shape[0], 1, dtype=torch.float32)
        act = torch.cat([joint, pad], dim=-1)
    flat = act.reshape(-1)
    target = action_horizon * action_dim
    if flat.numel() > target:
        flat = flat[:target]
    elif flat.numel() < target:
        flat = torch.cat([flat, torch.zeros(target - flat.numel())])
    return flat


def _log_and_verify_rl_setup(
    module: "DreamZeroPolicyModule",
    *,
    mode: str,
    tune_projector: bool,
    tune_diffusion_model: bool,
) -> None:
    """verl 层校验：yaml 的 LoRA/full 配置是否在 Groot 加载后真正生效。"""
    trainable = module.trainable_parameters_list()
    n_params = sum(p.numel() for p in trainable)
    lora_tensors = sum(
        1
        for name, p in module.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    )
    projector_tensors = sum(
        1
        for name, p in module.named_parameters()
        if p.requires_grad
        and any(part in name for part in ("action_decoder", "action_encoder", "state_encoder"))
    )
    print(
        f"[verl/VLA] RL mode={mode!r}: {len(trainable)} trainable tensors, "
        f"{n_params:,} params (lora={lora_tensors}, projector={projector_tensors})",
        flush=True,
    )
    if mode == "lora" and n_params == 0:
        raise RuntimeError(
            "[verl/VLA] rl_fine_tune_mode=lora but no trainable parameters after model load."
        )
    if mode == "lora" and tune_diffusion_model and lora_tensors == 0:
        raise RuntimeError(
            "[verl/VLA] tune_diffusion_model=true but no LoRA weights are trainable."
        )
    if mode == "lora" and tune_projector and projector_tensors == 0:
        raise RuntimeError(
            "[verl/VLA] tune_projector=true but no projector weights are trainable."
        )


class DreamZeroPolicyModule(nn.Module):
    """Shared VLA for rollout + PPO.

    verl 框架 LoRA 支持（分工）：
    - **本层 (vampo/integrations/verl)**：yaml 配置、optimizer 只更新 trainable、
      save_rl_checkpoint 只存 LoRA/projector 权重、PPO backward。
    - **GrootSimPolicy (groot/sim_policy.py)**：加载 checkpoint 后按 rl_mode 注入 LoRA /
      解冻 projector；DreamZero-DROID 等 dense checkpoint (train_architecture=full) 需在
      rl_mode=lora 时主动 inject LoRA adapter。
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device | str,
        action_horizon: int = 8,
        action_dim: int = 8,
        imagined_frames: int = 8,
        keep_lora_trainable: bool | None = None,
        rl_fine_tune_mode: str = "full",
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        primary_camera_key: str | None = None,
        flow_rl_sigma: float | None = None,
        flow_rl_video_sigma: float | None = None,
        lazy_load: bool = False,
        defer_post_initialize: bool = False,
        tokenizer_path_override: str | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_flat = action_horizon * action_dim
        self.imagined_frames = imagined_frames
        self.primary_camera_key = primary_camera_key
        self.device = torch.device(device)
        self._video_latent_numel = 0

        mode = str(rl_fine_tune_mode).lower()
        if keep_lora_trainable is not None:
            mode = "lora" if keep_lora_trainable else mode
        self.rl_fine_tune_mode = mode
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.groot = GrootSimPolicy(
            embodiment_tag=EmbodimentTag.OXE_DROID,
            model_path=model_path,
            device=str(self.device),
            device_mesh=None,
            lazy_load=lazy_load,
            defer_post_initialize=defer_post_initialize,
            tokenizer_path_override=tokenizer_path_override,
            keep_lora_trainable=(mode == "lora"),
            rl_fine_tune_mode=mode,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )
        ah = self.groot.trained_model.action_head
        if flow_rl_sigma is not None:
            ah.flow_rl_sigma = float(flow_rl_sigma)
        if flow_rl_video_sigma is not None:
            ah.flow_rl_video_sigma = float(flow_rl_video_sigma)
        elif flow_rl_sigma is not None:
            ah.flow_rl_video_sigma = float(flow_rl_sigma)
        self.flow_rl_sigma = float(ah.flow_rl_sigma)
        self.flow_rl_video_sigma = float(ah.flow_rl_video_sigma)
        self._video_buffer: list[torch.Tensor] = []
        _log_and_verify_rl_setup(
            self,
            mode=mode,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )

    @property
    def flow_steps(self) -> int:
        return int(self.groot.trained_model.action_head.num_inference_steps)

    def trainable_parameters_list(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def reset_episode(self) -> None:
        self._clear_action_head_episode_state()

    def _clear_action_head_episode_state(self) -> None:
        """Release Wan KV / language caches (GB10 unified memory — drop before update_actor)."""
        self._video_buffer = []
        ah = self.groot.trained_model.action_head
        if hasattr(ah, "current_start_frame"):
            ah.current_start_frame = 0
        ah.language = None
        ah.kv_cache1 = None
        ah.kv_cache_neg = None
        ah.crossattn_cache = None
        ah.crossattn_cache_neg = None
        if hasattr(ah, "clip_feas"):
            ah.clip_feas = None
        if hasattr(ah, "ys"):
            ah.ys = None
        if hasattr(ah, "skip_countdown"):
            ah.skip_countdown = 0

    @staticmethod
    def _trace_log_prob_valid(lp: torch.Tensor) -> bool:
        if lp.numel() == 0:
            return False
        vals = lp.detach().float().cpu()
        if vals.numel() != 1:
            return bool(torch.isfinite(vals).all() and float(vals.abs().max()) > 0)
        return bool(torch.isfinite(vals).all() and float(vals.abs()) > 0)

    def _log_prob_from_trace(self, trace: FlowJointTrace) -> torch.Tensor:
        """Exact log π from stored path/ε (no DiT re-forward)."""
        ap, ae, vp, ve = trace.to_numpy()
        if ap.size == 0 or vp.size == 0 or ae.size == 0 or ve.size == 0:
            return torch.tensor(float("nan"), dtype=torch.float32, device=self.device)
        if not (
            np.isfinite(ap).all()
            and np.isfinite(ae).all()
            and np.isfinite(vp).all()
            and np.isfinite(ve).all()
        ):
            return torch.tensor(float("nan"), dtype=torch.float32, device=self.device)
        val = compute_joint_flow_log_prob_from_paths(
            ap,
            ae,
            vp,
            ve,
            action_sigma=self.flow_rl_sigma,
            video_sigma=self.flow_rl_video_sigma,
        )
        if not math.isfinite(val):
            return torch.tensor(float("nan"), dtype=torch.float32, device=self.device)
        return torch.tensor(val, dtype=torch.float32, device=self.device)

    def _prepare_obs(self, obs: dict, prompt: str) -> dict[str, Any]:
        return convert_rl_obs_to_vla_obs(obs, prompt)

    def _decode_video_chunk(self, video_pred: torch.Tensor) -> np.ndarray:
        self._video_buffer.append(video_pred.detach())
        ah = self.groot.trained_model.action_head
        try:
            latent = torch.cat(self._video_buffer, dim=2)
            frames = ah.vae.decode(
                latent,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = torch.nan_to_num(frames.float(), nan=0.0)
            frames = ((frames + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            if frames.shape[0] >= self.imagined_frames:
                return frames[-self.imagined_frames :]
            pad = np.repeat(frames[:1], self.imagined_frames - frames.shape[0], axis=0)
            return np.concatenate([frames, pad], axis=0)
        except Exception:
            return np.zeros((self.imagined_frames, 64, 64, 3), dtype=np.uint8)

    def _rl_kwargs(
        self,
        rl_mode: str | None,
        flow_trace: FlowJointTrace | None,
    ) -> dict[str, Any]:
        if rl_mode == "trace":
            return {"rl_mode": "trace"}
        if rl_mode == "log_prob":
            if flow_trace is None:
                raise ValueError("log_prob mode requires flow_trace")
            trace = flow_trace.with_batch_dim()
            return {
                "rl_mode": "log_prob",
                "rl_action_path": trace.action_path.to(self.device),
                "rl_action_eps": trace.action_eps.to(self.device),
                "rl_video_path": trace.video_path.to(self.device),
                "rl_video_eps": trace.video_eps.to(self.device),
            }
        return {}

    def _joint_log_prob_from_aux(self, flow_aux: dict[str, Any]) -> torch.Tensor:
        if "flow_log_prob" in flow_aux:
            return flow_aux["flow_log_prob"]
        if "action_flow_log_prob" in flow_aux:
            return flow_aux["action_flow_log_prob"]
        raise KeyError(
            "flow_aux missing flow_log_prob; ensure rl_mode='trace' during rollout sampling"
        )

    def forward_vla(
        self,
        obs: dict,
        prompt: str,
        *,
        enable_grad: bool = False,
        rl_mode: str | None = None,
        flow_trace: FlowJointTrace | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
        batch = Batch(obs=self._prepare_obs(obs, prompt))
        grad_ctx = torch.enable_grad() if enable_grad else torch.inference_mode()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)
            if rl_mode == "log_prob" and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        with grad_ctx, autocast_ctx:
            result_batch, video_pred, flow_aux = self.groot.lazy_joint_forward_causal(
                batch,
                enable_grad=enable_grad,
                **self._rl_kwargs(rl_mode, flow_trace),
            )
        mean = _flat_action_from_act_dict(
            result_batch.act, self.action_horizon, self.action_dim
        ).to(self.device)
        if rl_mode == "log_prob":
            video = np.zeros((self.imagined_frames, 64, 64, 3), dtype=np.uint8)
        else:
            video = self._decode_video_chunk(video_pred)
        return mean, video, flow_aux

    def _trace_from_aux(self, flow_aux: dict[str, Any]) -> FlowJointTrace:
        action_path = FlowJointTrace._squeeze_batch(flow_aux["action_flow_path"])
        action_eps = FlowJointTrace._squeeze_batch(flow_aux["action_flow_eps"])
        video_path = FlowJointTrace._squeeze_batch(flow_aux["video_flow_path"])
        video_eps = FlowJointTrace._squeeze_batch(flow_aux["video_flow_eps"])
        self._video_latent_numel = int(video_path[0].numel())
        return FlowJointTrace(
            action_path=action_path,
            action_eps=action_eps,
            video_path=video_path,
            video_eps=video_eps,
        )

    @torch.no_grad()
    def sample_step(
        self, obs: dict, prompt: str
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, FlowJointTrace]:
        """Rollout sampling; stores flow trace and per-WM log π from trace mode forward."""
        mean, video, flow_aux = self.forward_vla(obs, prompt, enable_grad=False, rl_mode="trace")
        trace = self._trace_from_aux(flow_aux)
        action = mean.reshape(self.action_horizon, self.action_dim).cpu().numpy().astype(np.float32)
        flow_lp = self._joint_log_prob_from_aux(flow_aux).float().detach()
        return action, video, flow_lp, trace

    def _chain_log_prob(
        self,
        obs: dict,
        prompt: str,
        trace: FlowJointTrace,
        *,
        enable_grad: bool,
    ) -> torch.Tensor:
        """log π from stored path/ε; DiT re-forward only when grad is needed."""
        if not enable_grad:
            trace_lp = self._log_prob_from_trace(trace)
            if self._trace_log_prob_valid(trace_lp):
                return trace_lp

        try:
            _, _, flow_aux = self.forward_vla(
                obs,
                prompt,
                enable_grad=enable_grad,
                rl_mode="log_prob",
                flow_trace=trace,
            )
            lp = self._joint_log_prob_from_aux(flow_aux).float()
            if self._trace_log_prob_valid(lp):
                return lp
        except Exception as exc:
            if enable_grad:
                raise
            print(f"VAMPO WARNING: log_prob forward failed ({exc}); trace fallback", flush=True)

        trace_lp = self._log_prob_from_trace(trace)
        if not self._trace_log_prob_valid(trace_lp):
            ap, ae, vp, ve = trace.to_numpy()
            print(
                "VAMPO WARNING: log_prob degenerate after forward+trace; "
                f"shapes ap={ap.shape} ae={ae.shape} vp={vp.shape} ve={ve.shape}",
                flush=True,
            )
        return trace_lp

    def flow_entropy_per_wm_step(self) -> float:
        """Analytic entropy: action + video latent chains (diagonal Gaussian)."""
        import math

        action_elem = self.action_horizon * self.action_dim
        video_elem = self._video_latent_numel or action_elem * 4
        steps = self.flow_steps + 1
        action_ent = steps * 0.5 * action_elem * math.log(
            2 * math.pi * math.e * self.flow_rl_sigma ** 2
        )
        video_ent = steps * 0.5 * video_elem * math.log(
            2 * math.pi * math.e * self.flow_rl_video_sigma ** 2
        )
        return action_ent + video_ent

    def _log_prob_for_one_trajectory(
        self,
        obs_chunks: np.ndarray,
        traj_len: int,
        prompt: str,
        flow_traces: np.ndarray,
        *,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        row_steps = []
        self._clear_action_head_episode_state()
        step_obs_list = list(obs_chunks)
        trace_list = list(flow_traces)
        for t in range(traj_len):
            raw = trace_list[t]
            if isinstance(raw, FlowJointTrace):
                ft = raw.with_batch_dim()
            elif isinstance(raw, dict):
                ft = FlowJointTrace.from_dict(raw)
            else:
                raise TypeError(f"Unsupported flow trace type: {type(raw)}")
            lp = self._chain_log_prob(step_obs_list[t], prompt, ft, enable_grad=enable_grad)
            row_steps.append(lp)
        self._clear_action_head_episode_state()
        return torch.stack(row_steps)

    def log_prob_from_batch(
        self,
        obs_chunks_batch: np.ndarray,
        actions: torch.Tensor,
        prompts: list[str] | None = None,
        flow_traces_batch: np.ndarray | None = None,
        *,
        enable_grad: bool = True,
        traj_indices: list[int] | None = None,
    ) -> torch.Tensor:
        if flow_traces_batch is None:
            raise ValueError("flow_traces_batch required for flow log_prob_from_batch")

        batch_size, traj_len, _action_flat = actions.shape
        indices = traj_indices if traj_indices is not None else list(range(batch_size))
        step_log_probs = []
        for i in indices:
            prompt = prompts[i] if prompts else ""
            step_log_probs.append(
                self._log_prob_for_one_trajectory(
                    obs_chunks_batch[i],
                    traj_len,
                    prompt,
                    flow_traces_batch[i],
                    enable_grad=enable_grad,
                )
            )
        per_step = torch.stack(step_log_probs, dim=0)
        action_flat = actions.shape[-1]
        return per_step.unsqueeze(-1).expand(-1, -1, action_flat).reshape(
            len(indices), traj_len * action_flat
        )

    def save_rl_checkpoint(self, path: str) -> None:
        payload = {
            "flow_rl_sigma": self.flow_rl_sigma,
            "flow_rl_video_sigma": self.flow_rl_video_sigma,
            "rl_fine_tune_mode": self.rl_fine_tune_mode,
            "trainable": {n: p.cpu() for n, p in self.named_parameters() if p.requires_grad},
        }
        torch.save(payload, path)

    def load_rl_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "flow_rl_sigma" in payload:
            self.flow_rl_sigma = float(payload["flow_rl_sigma"])
            self.groot.trained_model.action_head.flow_rl_sigma = self.flow_rl_sigma
        if "flow_rl_video_sigma" in payload:
            self.flow_rl_video_sigma = float(payload["flow_rl_video_sigma"])
            self.groot.trained_model.action_head.flow_rl_video_sigma = self.flow_rl_video_sigma
        if "flow_rl_video_sigma" in payload:
            self.flow_rl_video_sigma = float(payload["flow_rl_video_sigma"])
            self.groot.trained_model.action_head.flow_rl_video_sigma = self.flow_rl_video_sigma
        for name, tensor in payload.get("trainable", {}).items():
            state = dict(self.named_parameters())
            if name in state:
                state[name].data.copy_(tensor.to(state[name].device, dtype=state[name].dtype))
