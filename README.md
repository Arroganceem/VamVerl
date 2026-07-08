# VamVerl

**VamVerl**（Video-Action Model Policy Optimization）：基于 **verl 分布式 RL 框架**，在 **DreamZero-DROID 视频-动作 VLA** 上实现 **WMPO 式全流程策略优化**。

**InitState 采样 → in-process 想象 rollout → VideoMAE 稀疏 reward → GRPO 组内 advantage → PPO clip 更新**

本仓库在 `vampo/`、`groot/`、`eval_utils/`、`verl/` 内自包含实现。

## 四组件架构

| # | 环节 | 说明 |
|---|------|------|
| ① | **verl 训练编排** | Ray `VAMPO PPO Trainer` · `actor_rollout_ref` hybrid engine · InitState / Rollout / Reward / Actor Worker |
| ② | **想象 Rollout** | `VLAPolicyModule` in-process · 每 WM 步 K=16 flow 联合去噪 · 记录 path/ε/trace → gen 后 `actor.compute_log_prob` |
| ③ | **Reward + GRPO** | 冻结 **VideoMAE** 滑窗判 success · 同 uid `n_samples` 条组内 advantage · log_prob tie-break |
| ④ | **PPO 更新** | 重放 flow 链算联合 log π · clip ratio · 更新 action head 可训子集 → **`policy.pt`** |

## 架构

```text
InitStateStore → ImaginationRollout → VideoMAERewardModel → GRPO (verl) → PPO → policy.pt
     imagination/        imagination/              reward/           integrations/verl/ + verl/
        ▲
        └── in-process DreamZero 基座 + flow log prob（VideoMAE 冻结 reward）
```

| 模块 | 路径 | 说明 |
|------|------|------|
| 想象 rollout | `vampo/imagination/` | `ImaginationRollout` 闭环 |
| VLA 基座 + RL | `groot/` + `integrations/verl/vla_policy.py` | DreamZero-DROID；PPO 更新 action head（full / LoRA） |
| Reward · VideoMAE | `vampo/reward/videomae_reward.py` | 8 帧滑窗 success 分类（冻结） |
| log prob / GRPO | `vampo/integrations/verl/` | `log_prob_utils` · `grpo_advantage` · `worker` fallback |
| verl 训练 | `verl/` + `main_vampo_ppo.py` | 组件 3 · Ray GRPO + PPO |

## 数据与基座

**Stage 1 · DROID init states**

LeRobot DROID → `train/val/rl_init` 划分（**rl_init 与 train disjoint**）→ `data/init_states/`（manifest + `*_obs.npy`，≈2888 条，相机 `exterior_image_1_left`）。

**Stage 2 · DreamZero-DROID**

~14B 联合 video-action WM：`CausalWanAttentionBlock`×40 DiT + Wan VAE + action projector。`lazy_joint_video_action_causal` 每 WM 步 K=16 步联合去噪。

## 安装

```bash
cd /home/robotem/WAM/VamVerl
pip install -e ".[verl,vla]"
```

- **`[verl]`**：组件 3（Ray + Hydra + verl）
- **`[vla]`**：DreamZero in-process rollout

默认基座：`MODEL_PATH=/home/robotem/Models/DreamZero-DROID`  
VideoMAE ckpt：`VIDEOMAE_CKPT=./checkpoints/videomae_droid.pth`  
组件 3 产出 **`policy.pt` 差分 ckpt**（可训权重 + flow σ），续训需基座 + `model.rl_checkpoint`。

## 脚本

| 组件 | 脚本 | 配置 |
|------|------|------|
| 2 · VideoMAE 离线训 | `python -m vampo.reward.train_videomae` | `configs/videomae_droid.yaml` |
| 3 · **verl GRPO + PPO** | **`vampo-train`** | **`vampo_ppo_trainer.yaml`** |
| 3 · 四机 FSDP+LoRA | **`scripts/train/train_component3_rl_cluster4.sh`** | **`vampo_ppo_trainer_cluster4.yaml`** |

```bash
export DROID_DATA_ROOT=/home/robotem/DATA/droid_lerobot
export MODEL_PATH=/home/robotem/Models/DreamZero-DROID
export VIDEOMAE_CKPT=/home/robotem/WAM/VamVerl/checkpoints/videomae_droid.pth

bash scripts/data/build_init_states_from_droid.sh
vampo-train                                  # 单机 FSDP · n_samples=8
# 或
bash scripts/cluster/sync_vamverl_cluster.sh
bash scripts/train/train_component3_rl_cluster4.sh   # 四机 FSDP+LoRA · n_samples=4
```

## 训练前检查

```bash
# 分阶段验证（推荐：无需完整训练 / 不加载 14B）
bash scripts/preflight/verify_staged.sh              # Stage 0–3：代码 + RL mock 链路
bash scripts/preflight/verify_staged.sh --through 4  # + VideoMAE CPU 冒烟
bash scripts/preflight/verify_staged.sh --through 5  # 开训前 strict（等同 preflight --strict-reward）
pytest tests/test_staged_rl_pipeline.py -q

# 传统 preflight
bash scripts/preflight/preflight_rl.sh --skip-reward    # 数据 / init_states / 基座
bash scripts/preflight/preflight_rl.sh --strict-reward  # 含 VideoMAE ckpt 冒烟
```

详见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

| 检查项 | 路径 / 变量 |
|--------|-------------|
| DreamZero 基座 | `MODEL_PATH` |
| DROID 数据 | `DROID_DATA_ROOT` |
| Episode 划分 | `data/splits/` |
| init_states | `bash scripts/data/build_init_states_from_droid.sh` |
| VideoMAE ckpt | `VIDEOMAE_CKPT` / `checkpoints/videomae_droid.pth` |

## 配置 profile

| Profile | 文件 | 拓扑 | n_samples | max_wm_steps | 微调 |
|---------|------|------|-----------|--------------|------|
| 单机正式 | `vampo_ppo_trainer.yaml` | FSDP 1 GPU | 8 | 8 | full（可调 projector-only） |
| 四机 cluster4 | `vampo_ppo_trainer_cluster4.yaml` | **FSDP FULL_SHARD** | 4 | 4 | **LoRA** + projector |

训练日志默认 W&B 项目 `vamverl`（`trainer.logger`）。

## 目录

```
VamVerl/
├── vampo/
│   ├── imagination/           # ImaginationRollout + InitStateStore
│   ├── reward/                # VideoMAE + 离线训练
│   └── integrations/verl/     # worker · actor · grpo · log_prob
├── verl/                      # vendored verl
├── groot/  eval_utils/
└── configs/vampo_ppo_trainer.yaml
```

## 设计原则

1. **WMPO 式闭环**：想象 rollout 与策略同模型（DreamZero），VideoMAE 作冻结 RM，GRPO+PPO on-policy 更新。
2. **log prob 链路**：rollout `rl_mode=trace` 写 `flow_log_prob` → `old_log_probs` / `rollout_log_prob_scalar`；退化时 worker **fallback `compute_log_prob`**。
3. **基座与存盘分离**：`MODEL_PATH` 加载完整 DreamZero；`policy.pt` 只存可训权重 + σ。
4. **数据 disjoint**：RL `rl_init` 与 VideoMAE 训练 episode 不重叠。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 总览与训练循环 |
| [docs/COMPONENT_RL.md](docs/COMPONENT_RL.md) | verl GRPO + PPO 详解 |
| [docs/VAMPO_LOG_PROB.md](docs/VAMPO_LOG_PROB.md) | flow 链 log prob 公式 |
| [docs/COMPONENT_REWARD.md](docs/COMPONENT_REWARD.md) | Reward（VideoMAE 主路径） |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | 分阶段验证（无完整训练） |
