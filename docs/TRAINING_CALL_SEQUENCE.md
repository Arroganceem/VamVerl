# Component3 RL · 训练过程代码调用时序

> 对应入口：`scripts/train/train_component3_rl_cluster4.sh`  
> 配置：`configs/vampo_ppo_trainer_cluster4.yaml`  
> 规模假设：`train_batch_size=1` · `n_samples=2` · `max_wm_steps=1` · 4 机 × 1 GPU FSDP

---

## 0. 启动 / 建组

```
train_component3_rl_cluster4.sh
  → python -m verl.trainer.main_dreamzero_ppo
      --config-name vampo_ppo_trainer_cluster4
  → main_hydra() / main(config)
      → _resolve_vamverl_paths(config)
      → _build_runtime_env(config)
      → ray.init(address=RAY_ADDRESS, runtime_env=...)
      → ray.get(main_task.remote(config))
          → Role.ActorRollout = DreamZeroActorRolloutRefWorker
          → DreamZeroRewardManager(reward_fn / val_reward_fn)
          → ResourcePoolManager(nnodes × n_gpus)
          → RayTrainer(...).init_workers()
              → DreamZeroRayWorkerGroup 拉起 4 个 Worker
              → Worker.init_model() ×4
                  → 加载 sharded DiT checkpoint
                  → FSDP sequential_block_wrap
                  → DreamZeroPPOActor + AdamW (optimizer_offload→CPU)
                  → DreamZeroRollout + VideoMAE reward
          → RayTrainer.fit()
```

---

## 1. 单步总览（`RayTrainer.fit` 一个 global_step）

```
[step begin] log_step_banner(phase="rollout")

── A. 凑满 batch（while len(valid) < batch×n_samples = 2）──

DreamZeroInitStateDataset.get_next_batch()
  → DataProto{state_id, init_index, ...}
  → 为每条 prompt 复制 n_samples 份，分配 uid
  → gen_batch.meta_info = {n_samples, recompute_log_prob=True, use_wm, ...}

actor_rollout_wg.generate_sequences(gen_batch)
  → （详见 §2）
  → DataProto{responses, complete, finish_step, old_log_probs,
              rollout_log_prob_scalar, obs_features, flow_traces, ...}

DreamZeroRewardManager.verify(roll_batch)
  → complete → batch["acc"]
  → scores / verify metrics

（可选 filter_accuracy / filter_truncated）
  → 并入 valid_batch
  → print: collected N / 2 rollouts ...

── B. Driver 侧 reward / GRPO ──

log_batch_rewards(phase="driver_verify")

DreamZeroRewardManager.__call__(batch)
  → sparse token reward @ (finish_step × action_token_len - 1)
  → × verifier.reward_coef
  → print: VAMPO [token_reward] verifier=... reward_all=...

apply_kl_penalty(batch, ...)
  → token_level_rewards

compute_advantage(..., adv_estimator="grpo")
  → compute_dreamzero_grpo_outcome_advantage(...)
      → 组内 outcome=(R-μ)/σ
      → 若组内 reward std≈0 → log_prob tie-break
  → print: VAMPO GRPO tie-break / adv=...

log_grpo_summary(...)

── C. 更新 Actor ──

log_step_banner(phase="update_actor")
_maybe_pre_update_cooldown()          # sleep 30s + gc/flush
  → print: pre-update unified-memory cooldown ...

actor_rollout_wg.update_actor(batch)
  → （详见 §3）
  → metrics{actor/pg_loss, entropy_loss, grad_norm, ...}

actor_rollout_wg.compute_entropy(batch)
  → metrics{actor/entropy}

── D. 收尾 ──

compute_data_metrics / logger.log
print: VAMPO [step_done] ...
（可选 save_checkpoint）
global_steps += 1
_maybe_epoch_cooldown()               # sleep 60s
```

---

## 2. Rollout 细节（`generate_sequences`）

```
Driver
  → WorkerGroup.generate_sequences(prompts)
      → DreamZeroActorRolloutRefWorker.generate_sequences
          → sharding_manager.preprocess_data
          → DreamZeroRollout.generate_sequences(prompts)
              │
              ├─ 对每个 init_index:
              │     InitStateStore.get(init_index)
              │       → (state_id, obs, prompt)
              │     ImaginationRollout.rollout_group(obs, prompt, state_id, n_samples=2)
              │       │
              │       ├─ sample_idx=0..1（共享同一 uid）:
              │       │     ImaginationRollout.rollout_one(...)
              │       │       → PolicyRunner.reset_episode / rollout_mode
              │       │       → for wm in 1..max_wm_steps(=1):
              │       │             build_obs_with_video_history(...)
              │       │             PolicyRunner.infer(obs_in, prompt)
              │       │               → DreamZeroInProcessBackend
              │       │                 → DreamZeroPolicyModule forward (WM imagine)
              │       │                   → action + video_frames + flow_path/eps
              │       │             记录 ChunkRecord + flow_log_prob
              │       │             append_imagined_frame(...)
              │       │       → VideoMAERewardModel.predict_success(traj.video)
              │       │             → build_sliding_clips(window=8, min_steps=8)
              │       │             → scan_clips_for_success(threshold≈sidecar)
              │       │             → SuccessResult(complete, finish_frame)
              │       │             → print: VAMPO reward [state] complete=... finish_frame=...
              │       │             → reward_model.offload()
              │       │       → finish_wm = frame_finish_to_wm_step(...)
              │       │       → return Trajectory
              │       → return list[Trajectory]
              │
              └─ trajectories_to_dataproto(...)
                    → DataProto(+ complete / finish_step / flow_traces / ...)

          → _offload_rollout_reward()
          → DreamZeroPPOActor.compute_log_prob(output)     # recompute old π
              → _log_probs_for_batch(..., enable_grad=False)
              → apply_recomputed_log_prob_fields
              → print: VAMPO [recomputed] rollout_log_prob ...
          → log_batch_rewards(phase="worker_rollout")      # rank0
                → print: VAMPO [worker_rollout] success_rate=...
          → reset_episode + _flush_unified_memory("After generate_sequences")
          → return output.to("cpu")
  ← Driver 收到 rollout batch
```

---

## 3. Actor 更新细节（`update_actor` + `compute_entropy`）

```
Driver
  → WorkerGroup.update_actor(batch)
      → DreamZeroActorRolloutRefWorker.update_actor
          → _prepare_for_policy_update()
                → VideoMAE offload
                → reset_episode / 清 KV
                → grad_offload（若仅 grad）
                → flush_unified_memory("Before update_actor")
          → data.to(device)
          → DreamZeroPPOActor.update_policy(..., hooks)
                → optimizer.zero_grad
                → for micro-batch (B=2, micro=1 → 2 次):
                      _ppo_loss_for_micro_batch
                        → VLA train forward 算 logπ
                        → PPO policy-gradient loss + entropy
                      scaled.backward()
                      reset_episode()
                      after_micro_backward
                        → grad → CPU
                        → flush_unified_memory("after micro backward")
                      print: VAMPO update micro i/N pg_loss=... entropy=...
                → before_optimizer_step
                      → load_fsdp_optimizer (Adam 状态短暂上 GPU)
                → clip_grad_norm_vla(...)
                → optimizer.step()
                → after_optimizer_step
                      → optimizer_offload (Adam → CPU)
                      → grad_offload
          → flush_unified_memory("After update_actor")
          → return DataProto(meta_info.metrics)
  ← Driver 收到 actor metrics

Driver
  → WorkerGroup.compute_entropy(batch)
      → DreamZeroPPOActor.compute_entropy
            → 用 flow_entropy 常数值填 mask（几乎无额外 DiT 训练算）
  ← {actor/entropy}
```

---

## 4. Driver ↔ Worker 数据流

```
Driver ──generate_sequences──▶ Worker
  in : state_id, init_index, n_samples, recompute_log_prob=True
  out: responses, obs_features, complete, finish_step,
       old_log_probs, rollout_log_prob_scalar, flow_traces, ...

Driver 本地:
  verify(complete→acc)
    → sparse token reward
    → apply_kl_penalty
    → GRPO advantage (+ tie-break)

Driver ──update_actor──▶ Worker
  in : responses, obs_features, old_log_probs, advantages, finish_step
  out: actor/pg_loss, entropy_loss, grad_norm, micro_batches

Driver ──compute_entropy──▶ Worker
  out: actor/entropy
```

---

## 5. 日志阶段对照

| 日志关键字 | 对应位置 |
|-----------|----------|
| `VAMPO reward [...] complete=...` | §2 VideoMAE `predict_success` |
| `VAMPO [recomputed] rollout_log_prob` | §2 `compute_log_prob` 之后 |
| `VAMPO [worker_rollout]` | §2 rank0 `log_batch_rewards` |
| `gen: Xx seconds` | §1 整段 `generate_sequences` |
| `VAMPO [driver_verify]` | §1 `verify` 后 |
| `VAMPO [token_reward]` | §1 `DreamZeroRewardManager.__call__` |
| `VAMPO GRPO tie-break` / `[grpo]` | §1 `compute_dreamzero_grpo_outcome_advantage` |
| `pre-update unified-memory cooldown` | §1 → §3 之间 |
| `VAMPO === ... update_actor ===` | §3 入口 |
| `update_actor: Xx seconds` | §3（Timer 含随后的 `compute_entropy`） |
| `VAMPO [step_done]` | §1 收尾 |

---

## 6. 关键源文件索引

| 阶段 | 文件 |
|------|------|
| 启动脚本 | `scripts/train/train_component3_rl_cluster4.sh` |
| Hydra / Ray 入口 | `verl/trainer/main_dreamzero_ppo.py` |
| 训练环 | `verl/trainer/ppo/ray_trainer.py` → `fit()` |
| Worker | `verl/workers/dreamzero_worker.py` |
| PPO Actor | `verl/workers/actor/dp_dreamzero.py` |
| Rollout 适配 | `verl/workers/rollout/dreamzero_rollout.py` |
| 想象轨迹 | `verl/workers/rollout/imagination/rollout.py` |
| VideoMAE reward | `verl/utils/reward/videomae_reward.py` · `wmpo_sliding.py` |
| Driver reward | `verl/trainer/ppo/dreamzero_reward_manager.py` |
| GRPO | `verl/trainer/ppo/dreamzero_grpo.py` |
| 进度日志 | `verl/trainer/ppo/dreamzero_progress.py` |
