"""Closed-loop imagination rollout and init-state loading."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import torch.distributed as dist

logger = logging.getLogger(__name__)

from vampo.imagination.policy.obs_utils import (
    append_imagined_frame,
    build_obs_with_video_history,
    init_frame_buffers,
)
from vampo.imagination.policy.runner import PolicyRunner
from vampo.imagination.reward_debug import debug_reward_enabled, dump_trajectory_video
from vampo.reward.base import BaseRewardModel
from vampo.rl.trajectory import ChunkRecord, Trajectory


class InitStateStore:
    """Load initial observations from manifest + npy under data/init_states/."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            self.entries = data if isinstance(data, list) else data.get("entries", [])
        else:
            self.entries = []

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, index: int) -> tuple[str, dict, str]:
        if not self.entries:
            raise RuntimeError(
                f"No init states under {self.root}; "
                "run: bash scripts/build_init_states_from_droid.sh"
            )
        entry = self.entries[index % len(self.entries)]
        state_id = entry["state_id"]
        prompt = entry.get("prompt", "")
        obs_path = self.root / entry["obs_file"]
        if not obs_path.is_file():
            raise FileNotFoundError(f"Missing observation file: {obs_path}")
        obs = np.load(obs_path, allow_pickle=True).item()
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict obs in {obs_path}")
        return state_id, obs, prompt


class ImaginationRollout:
    """Closed-loop imagined trajectories via configured policy backend."""

    def __init__(
        self,
        policy: PolicyRunner,
        reward_model: BaseRewardModel | None = None,
        max_wm_steps: int = 8,
        primary_camera_key: str | None = None,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.max_wm_steps = max_wm_steps
        self.primary_camera_key = primary_camera_key

    def _pick_camera_key(self, obs: dict) -> str:
        if self.primary_camera_key and self.primary_camera_key in obs:
            return self.primary_camera_key
        for k in obs:
            if "video" in k or "image" in k:
                return k
        raise KeyError("No video key found in observation dict")

    def rollout_one(
        self,
        init_obs: dict,
        prompt: str,
        state_id: str,
        uid: str,
        sample_idx: int = 0,
    ) -> Trajectory:
        from vampo.integrations.verl.parallel_utils import seed_rollout_sample

        seed_rollout_sample(uid, sample_idx)
        traj = Trajectory(init_state_id=state_id, prompt=prompt, uid=uid)
        self.policy.reset_episode()
        obs = dict(init_obs)
        if prompt and "annotation.language.action_text" not in obs:
            obs["annotation.language.action_text"] = prompt
        frame_buffers = init_frame_buffers(obs)
        is_first_wm_step = True

        with self.policy.rollout_mode():
            for _ in range(self.max_wm_steps):
                obs_in = build_obs_with_video_history(
                    obs,
                    frame_buffers,
                    is_first_wm_step=is_first_wm_step,
                    prompt=prompt,
                )
                out = self.policy.infer(obs_in, prompt)
                from vampo.integrations.verl.log_prob_utils import (
                    debug_log_prob_enabled,
                    log_prob_scalar_from_chunk,
                )

                flow_lp = out.info.get("flow_log_prob")
                if flow_lp is None and out.flow_path is not None:
                    flow_lp = log_prob_scalar_from_chunk(
                        ChunkRecord(
                            obs=obs_in,
                            action=out.action,
                            video_frames=out.video_frames,
                            flow_path=out.flow_path,
                            flow_eps=out.flow_eps,
                            video_flow_path=out.video_flow_path,
                            video_flow_eps=out.video_flow_eps,
                        ),
                        action_sigma=float(
                            getattr(
                                getattr(self.policy.backend, "module", None),
                                "flow_rl_sigma",
                                0.05,
                            )
                        ),
                        video_sigma=float(
                            getattr(
                                getattr(self.policy.backend, "module", None),
                                "flow_rl_video_sigma",
                                0.05,
                            )
                        ),
                    )
                if debug_log_prob_enabled() and (not dist.is_initialized() or dist.get_rank() == 0):
                    wm_idx = len(traj.chunks)
                    lp_str = f"{flow_lp:.6e}" if flow_lp is not None else "None"
                    print(
                        f"VAMPO DEBUG log_prob uid={uid[:8]} sample={sample_idx} "
                        f"wm={wm_idx} flow_log_prob={lp_str}",
                        flush=True,
                    )
                traj.chunks.append(
                    ChunkRecord(
                        obs=obs_in,
                        action=out.action,
                        video_frames=out.video_frames,
                        flow_path=out.flow_path,
                        flow_eps=out.flow_eps,
                        video_flow_path=out.video_flow_path,
                        video_flow_eps=out.video_flow_eps,
                        flow_log_prob=float(flow_lp) if flow_lp is not None else None,
                    )
                )
                append_imagined_frame(frame_buffers, out.video_frames)
                obs = obs_in
                is_first_wm_step = False

        if debug_reward_enabled() and (not dist.is_initialized() or dist.get_rank() == 0):
            dump_dir = dump_trajectory_video(traj, state_id=state_id)
            logger.info("VAMPO_DEBUG_REWARD: dumped WM video → %s", dump_dir)

        from vampo.reward.wmpo_sliding import frame_finish_to_wm_step

        if self.reward_model is not None:
            total_frames = int(traj.video.shape[0])
            complete = False
            finish_frame = total_frames - 1
            try:
                result = self.reward_model.predict_success(traj.video, prompt=traj.prompt)
                complete = bool(result.complete)
                finish_frame = int(result.finish_step)
                msg = (
                    f"VAMPO reward [{state_id}] uid={uid[:8]} "
                    f"complete={int(complete)} finish_frame={finish_frame} "
                    f"video_T={total_frames} prompt={traj.prompt[:60]!r}"
                )
                logger.info(msg)
                print(msg, flush=True)
            except Exception as exc:
                logger.error(
                    "Reward model failed for state_id=%s — treating as incomplete: %s",
                    state_id,
                    exc,
                )
            finally:
                if hasattr(self.reward_model, "offload"):
                    self.reward_model.offload()
            frames_per_chunk = max(1, total_frames // max(len(traj.chunks), 1))
            traj.complete = complete
            traj.finish_step = frame_finish_to_wm_step(
                finish_frame,
                frames_per_chunk=frames_per_chunk,
                max_wm_steps=len(traj.chunks),
            )

        return traj

    def rollout_group(
        self,
        init_obs: dict,
        prompt: str,
        state_id: str,
        n_samples: int,
    ) -> list[Trajectory]:
        """Sample n trajectories sharing one GRPO group uid (used by verl rollout)."""
        uid = str(uuid.uuid4())
        return [
            self.rollout_one(init_obs, prompt, state_id, uid, sample_idx=i)
            for i in range(n_samples)
        ]
