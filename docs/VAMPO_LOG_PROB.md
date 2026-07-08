# VAMPO · Flow 联合 log prob 计算逻辑

本文说明组件 3（verl GRPO + PPO）中 **log prob 如何定义、何时计算、如何进入 PPO ratio**。  
相关训练总览见 [COMPONENT_RL.md](./COMPONENT_RL.md)。

---

## 1. 一句话

DreamZero 每 WM 步跑 **K 步联合去噪链**（action latent + video latent）。Rollout 时按 `μ + σ·ε` 采样并 **存 path/ε**；PPO 时 **固定 path/ε**，用当前 DiT 重算 `μ`，累加对角高斯 log prob → `ratio = exp(log π_new − log π_old)`。

**不是** 监督里的 MSE(预测噪声, training_target)；**是** `log N(x | μ, σ)`，其中 `x` 为 rollout 已走过的样本。

---

## 2. 何时算 log prob（两处）

| 阶段 | 调用链 | `enable_grad` | 输出 | 计时 |
|------|--------|---------------|------|------|
| **A. old log prob** | rollout trace → `old_log_probs`（默认）；或 fallback `compute_log_prob` | `False` | `old_log_probs [B, T×flat]` | 含在 Driver `gen:` 内 |
| **B. new log prob** | `update_actor` → `_log_probs_for_batch` | `True` | 用于 PPO ratio | Driver `update_actor:` |

```
┌─ RPC: generate_sequences（Worker）─────────────────────────────────┐
│  ImaginationRollout · rl_mode=trace                                │
│    采样 + 存 flow_traces + flow_log_prob → old_log_probs           │
│    （reuse_trace_log_prob=true 时跳过下方重算）                      │
│         │                                                          │
│         ▼  （仅 fallback）                                          │
│  compute_log_prob · rl_mode=log_prob → old_log_probs               │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌─ Ray Driver ────────────────────────────────────────────────────────┐
│  verify → reward → GRPO advantages                                 │
│         │                                                          │
│         ▼                                                          │
│  update_actor · log_prob_from_batch → new_log_probs + backward     │
└────────────────────────────────────────────────────────────────────┘
```

**cluster4 当前规模（`max_wm_steps=8`, `train_batch_size=1`, `n_samples=4`）：**

- 4 条轨迹 × 8 WM 步 = **32 次** trace forward（rollout）
- ~~同 RPC 内再 **32 次** log_prob 重放 → `old_log_probs`~~（**默认已关闭**：`reuse_trace_log_prob: true`，rollout trace 直接写入 `old_log_probs`）
- `update_actor` 再 **32 次** 带梯度重放（`ppo_micro_batch_size=1`，4 条轨迹分 4 个 micro backward）

粗算：每 PPO step ≈ **64 次** DreamZero WM forward（32 rollout + 32 update），较「gen 内二次 old 重放」方案少 **32 次**。

---

## 3. 端到端数据流

```
Init obs + prompt
      │
      ▼
┌─ WM 步 t = 0 … T-1 ─────────────────────────────────────────────┐
│  VLAPolicyModule.sample_step / _chain_log_prob                     │
│    obs_t + flow_trace_t                                            │
│         │                                                          │
│         ▼                                                          │
│  GrootSimPolicy.lazy_joint_forward_causal                          │
│    → WanFlowMatchingActionHead.lazy_joint_video_action             │
│         │                                                          │
│         ├─ rl_mode=trace  （rollout）                              │
│         │     ε ~ N(0,I) 或从 path 重放                            │
│         │     x = μ + σ·ε                                          │
│         │     累加 log N(x|μ,σ) → flow_log_prob                    │
│         │     存 action_path, action_eps, video_path, video_eps    │
│         │                                                          │
│         └─ rl_mode=log_prob （PPO 重放）                           │
│               固定 path[0] 为初始噪声 x_T                           │
│               每步 k：用 DiT 算 μ_k → log N(path[k+1] | μ_k, σ)    │
│               下一步输入：μ_k + σ·eps[k]（同一个 ε）                │
│         │                                                          │
│         ▼                                                          │
│  每 WM 步 1 个标量 log π_t                                         │
└────────────────────────────────────────────────────────────────────┘
      │
      ▼
log_prob_from_batch：对 T 个 WM 步 stack → [B, T]
      │
      ▼
expand 到 action flat 维 → [B, T × action_flat]  （PPO mask 用 finish_step）
      │
      ▼
ratio = exp(new_log_probs - old_log_probs)  →  clip PPO loss
```

---

## 4. Rollout：`rl_mode=trace`

### 4.1 随机性来源

DiT forward 与推理相同，得到 scheduler 的 **均值** `μ_video`、`μ_action`，再注入策略噪声：

```
x^v_{k} = μ^v_k + σ_v · ε^v_k      ε^v_k ~ N(0,I)  （trace 时现场采样）
x^a_{k} = μ^a_k + σ_a · ε^a_k      ε^a_k ~ N(0,I)
```

`σ_a = flow_rl_sigma`，`σ_v = flow_rl_video_sigma`（默认 0.05，可 yaml / 环境变量覆盖）。

### 4.2 记录内容（`FlowJointTrace`）

| 字段 | Shape（单 WM 步） | 含义 |
|------|-------------------|------|
| `action_path` | `[K+1, B, H, D]` | action latent 链，含初始噪声 |
| `action_eps` | `[K, B, H, D]` | 每步 action 噪声 ε |
| `video_path` | `[K+1, B, T, C, H, W]` | video latent 链 |
| `video_eps` | `[K, B, T, C, H, W]` | 每步 video 噪声 ε |

- `K = num_inference_steps`（日志里常见 **DIT Compute Steps 8**，为 skip 后实际 DiT 步数；scheduler 时间步可更多）
- `H×D = action_horizon × action_dim`（默认 8×8=64）

存入 `DataProto.non_tensor_batch["flow_traces"]`，随 batch 回 Driver。

### 4.3 Trace 阶段 log prob 累加

```python
# groot/.../wan_flow_matching_action_tf.py · rl_mode == "trace"
flow_log_prob_total += log N(x^a_T; 0, I)          # 初始 action 噪声
flow_log_prob_total += log N(x^v_T; 0, I)          # 初始 video 噪声
for k in 0..K-1:
    μ = DiT(...) → scheduler.step(...)
    x = μ + σ·ε                                      # 新采样 ε
    flow_log_prob_total += log N(x | μ, σ)           # action + video 各一项
```

---

## 5. PPO 重放：`rl_mode=log_prob`

### 5.1 与 trace 的关键区别

| | trace（rollout） | log_prob（PPO） |
|--|------------------|-----------------|
| 初始噪声 | 随机生成 | **读** `path[0]` |
| 每步 ε | **新采样** | **读** 存好的 `eps[k]` |
| 每步 x（用于 log prob 的 target） | 刚采样的 x | **读** `path[k+1]` |
| DiT 权重 | rollout 时 θ_old | 重放时 θ（old 或 new） |
| 梯度 | 关 | old：关；new：**开** |

### 5.2 重放循环（核心）

```python
# rl_mode == "log_prob"
noise_action = rl_action_path[0]
noise_obs    = rl_video_path[0]
flow_log_prob_total = log N(noise_action; 0,I) + log N(noise_obs; 0,I)

for index, timestep in enumerate(scheduler.timesteps):
    μ_video, μ_action = DiT_forward(...) → scheduler.step(...)

    # target = rollout 时走过的 x，固定不变
    flow_log_prob_total += log N(rl_video_path[index+1]  | μ_video,  σ_v)
    flow_log_prob_total += log N(rl_action_path[index+1] | μ_action, σ_a)

    # 链式输入：用同一 ε 推进
    noisy_input        = μ_video  + σ_v * rl_video_eps[index]
    noisy_input_action = μ_action + σ_a * rl_action_eps[index]
```

高斯 log prob 实现（对角、全元素求和）：

```python
# _flow_rl_gp_log_prob(target, mean, sigma)
# = Σ_i  -0.5 * ( (target_i - mean_i)²/σ² + log(2πσ²) )
```

标准正态初始项：

```python
# _flow_rl_standard_normal_log_prob(sample) = Σ_i -0.5 * (sample_i² + log(2π))
```

### 5.3 Python 调用栈

```
VAMPODPOActor.compute_log_prob / _log_probs_for_batch
  └─ VLAPolicyModule.log_prob_from_batch
       for each trajectory i, wm step t:
         reset_episode()
         FlowJointTrace.from_dict(flow_traces[i][t])
         _chain_log_prob(obs, prompt, trace, enable_grad)
           └─ forward_vla(..., rl_mode="log_prob", flow_trace=trace)
                └─ lazy_joint_forward_causal(**rl_action_path, rl_video_path, ...)
```

`log_prob_from_batch` 将每 WM 步标量 **broadcast** 到 flat action 维（供 verl 的 token-level mask）：

```python
# [B, T] → [B, T, action_flat] → [B, T * action_flat]
per_step.unsqueeze(-1).expand(-1, -1, action_flat).reshape(...)
```

---

## 6. 联合 log prob 公式

单个 **WM 步** 的标量（action + video 全链）：

$$
\log \pi_\theta =
\underbrace{\log \mathcal{N}(x^a_T; 0, I) + \log \mathcal{N}(x^v_T; 0, I)}_{\text{初始 latent}}
+ \sum_{k=0}^{K-1} \Big[
  \log \mathcal{N}(x^a_{k+1} \mid \mu^a_k(\theta), \sigma_a)
  + \log \mathcal{N}(x^v_{k+1} \mid \mu^v_k(\theta), \sigma_v)
\Big]
$$

- $x$、$\varepsilon$ 来自 rollout **固定**；$\mu(\theta)$ 随 DiT 权重变。
- **不更新** $\sigma$、$\varepsilon$；PPO 只改使高 reward 路径下 $\mu$ 更「对齐」已走样本的网络参数。

整条轨迹（T 个 WM 步）在实现上取 **每步一个标量**，再 expand 到 `T × action_token_len` 维做 mask（非逐步独立高斯 policy head）。

---

## 7. 进入 PPO

### 7.1 old / new

```python
# vampo/integrations/verl/actor.py · update_policy
new_log_probs = self._log_probs_for_batch(..., enable_grad=True)
ratio = torch.exp(new_log_probs - old_log_prob)
pg_loss = max(-A·ratio, -A·clip(ratio, 1-ε_low, 1+ε_high))
loss = pg_loss - entropy_coeff * entropy_loss
```

### 7.2 Response mask

```python
response_length = traj_len * action_flat          # cluster4: 8 × 64 = 512
finish = finish_step * action_token_len           # action_token_len = 64
response_mask[t] = (t < finish)                   # VideoMAE 判定 complete 前的 WM 步
```

`finish_step` 来自 VideoMAE sliding-window：首次 `P(success) ≥ threshold` 的 WM 步。

### 7.3 Entropy（VLA 模式）

不用 Gaussian head 的 `log_std`；用 **解析熵** `flow_entropy_per_wm_step()`（action + video 链对角高斯，固定 σ）均匀填到 mask 位置。

---

## 8. 与监督 flow matching 对照

```
监督 SFT                          VAMPO RL
────────                          ────────
真实 demo latent                  rollout: x = μ + σ·ε
      │                                 │
      ▼                                 ▼
DiT → 预测 flow                     存 path / ε
      │                                 │
      ▼                                 ▼
MSE(预测, training_target)        重放: log N(x | μ_new, σ)
                                        │
                                        ▼
                                  PPO + VideoMAE reward
```

同一套 DiT forward 算 $\mu$；**损失形式不同**（MSE vs 策略梯度 on 随机链）。

---

## 9. 配置与代码索引

| 项 | 默认 | 位置 |
|----|------|------|
| `flow_rl_sigma` | 0.05 | `configs/vampo_ppo_trainer.yaml` |
| `flow_rl_video_sigma` | 0.05 | 同上 |
| `action_token_len` | 64 | 同上 |
| `max_wm_steps` | 8 | `data.max_wm_steps`（cluster4 与 base 一致） |
| `n_samples` | 4（cluster4） | `data.n_samples` |
| `ppo_micro_batch_size` | 1（cluster4） | 逐条 trajectory backward |
| 环境变量 | `VAMPO_FLOW_RL_SIGMA` / `VAMPO_FLOW_RL_VIDEO_SIGMA` | 覆盖 yaml |

| 模块 | 文件 |
|------|------|
| 高斯 log prob 工具 | `vampo/integrations/verl/flow_log_prob.py` |
| Trace / 重放入口 | `vampo/integrations/verl/vla_policy.py` |
| PPO actor | `vampo/integrations/verl/actor.py` |
| generate + old_log_prob | `vampo/integrations/verl/worker.py` |
| flow trace → batch | `vampo/integrations/verl/proto_adapter.py` |
| 去噪链 log prob 核心 | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` |
| Driver 训练循环 | `verl/trainer/ppo/ray_trainer.py` |

---

## 10. 小数值例子（单 WM 步 · 单去噪步 · 仅 action）

**Rollout（θ_old）：**

```
μ_old = 0.80
ε   = +1.0        （采样一次，写入 action_eps[0]）
x   = 0.80 + 0.05×1.0 = 0.85   （写入 action_path[1]）
log π_old 含项：log N(0.85 | μ_old, 0.05)
```

**PPO 更新（θ_new，ε 仍为 +1.0，x 仍为 0.85）：**

```
μ_new = 0.84
log π_new 含项：log N(0.85 | 0.84, 0.05)   ← target 是 x，不是 training_target
ratio = exp(log π_new - log π_old)
```

若 advantage > 0（VideoMAE 判 success），梯度增大该路径下的 log π_new。

---

## 11. 常见误区

1. **MISSING/UNEXPECTED VideoMAE LOAD REPORT** 与 log prob **无关**（reward 模型加载日志）。
2. **`reward: 0.0 seconds`** 不是没算 reward；reward 在 rollout 内已写入 `reward_score`，Driver 只做 token 展开。
3. **log prob 不是** 对 `action_pred` 再套一层 `Normal(mean, log_std)`（那是 legacy `GaussianPolicyModule` 路径）。
4. **`language is None, reset current_start_frame`** 在 log prob 重放时出现是 **正常现象**（每 WM 步 `reset_episode()` 清空语言/KV 状态后重跑 forward）。
