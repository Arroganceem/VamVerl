```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              VamVerl · 数据流与 RL 训练总览                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════  主训练路径  ════════════════════════════════════════════════════════

  ┌──────────────────┐   ┌──────────────────────────┐   ┌─────────────────────────────┐
  │ init_states/     │   │ DreamZero-DROID ckpt       │   │ VideoMAE ckpt（冻结 RM）     │
  │ manifest+*_obs   │   │ /home/robotem/Models/...   │   │ videomae_droid.pth          │
  └────────┬─────────┘   └─────────────┬──────────────┘   └──────────────┬──────────────┘
           │                           │                                  │
           └───────────────────────────┴──────────────────────────────────┘
                                       │
                                       ▼
                         【verl GRPO + PPO · 主训练】
                         init_states + DreamZero 基座 + VideoMAE reward
                                       │
                                       ▼
                         在线 rollout → FlowJointTrace → GRPO → PPO → policy.pt（差分 ckpt）


  ── 推荐执行顺序 ──

  [0] 训练或准备 VideoMAE ckpt（组件 2，与 rl_init disjoint）
  [1] bash scripts/build_init_states_from_droid.sh
  [2] bash scripts/train_component3_rl.sh  （或 train_component3_rl_cluster4.sh）
```

## 训练完整循环

下图从 **数据准备** 到 **verl 每个 training step 的闭环**，对应 `train_component3_rl.sh`。

**默认数值** 来自 `configs/vampo_ppo_trainer.yaml`。

### Phase 0 · 数据规模（一次性）

| 阶段 | 路径 / 产物 | 典型规模 | 格式要点 |
|------|-------------|----------|----------|
| 初始状态 | `data/init_states/` | 从 **`data/splits/rl_init_episodes.json`** 导出 t=0；默认最多 **2888** 条 | 与 hold-out **train** episode disjoint |
| Episode 划分 | `data/splits/` | train ~95% / val ~5% / rl_init（默认=val） | `droid_to_init_states` 或 split 工具写出 |
| 原始 DROID | `data/droid_lerobot/` | LeRobot：`meta/episodes.jsonl` + `videos/.../episode_*.mp4` | RL init 导出用 |
| VLM Reward | `http://192.168.88.41:1234/v1` | **可选**零样本 judge（`reward.backend: lmstudio`） |
| VideoMAE RM | `checkpoints/videomae_droid.pth` | **默认**冻结 success 分类器 |

```
DROID episodes
    ├── train_episodes.json            (~95% hold-out)
    ├── val_episodes.json              (~5%)
    └── rl_init_episodes.json          (= val，或单独 hold-out 10%)
```

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 0 · 一次性准备（训练前）                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  bash scripts/build_init_states_from_droid.sh   # rl_init pool only
        │
        ▼
  init_states/  ·  rl_init episode 起点（默认 ≤2888 条，无 train 泄漏）
  checkpoints/videomae_droid.pth  ·  组件 2 产出（与 rl_init disjoint）


╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 1 · verl 主循环  train_component3_rl.sh                              ║
║  每 global_step：train_batch_size × n_samples 条轨迹 → 1 次 PPO 更新         ║
╚══════════════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────────────────┐
                    │  冻结（每 step · 仍参与 forward）       │
                    │  VLA backbone ❄️                      │
                    │  action_head 内 VAE / text·image enc ❄️ │
                    │  LM Studio VLM（可选 HTTP）❄️          │
                    │  VideoMAE reward model ❄️              │
                    │  init_states 池 · N_init 条          │
                    └──────────────────┬──────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │ ① 采样 · VAMPOInitStateDataset                             │
         │    train_batch_size=1 → 1 个 init_index                    │
         │    复制 n_samples=8 → 同 uid 下 8 条轨迹（GRPO 组）          │
         │    输入 obs dict + prompt                                    │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │ ② Rollout · ImaginationRollout.rollout_one               │
         │    循环 ≤ max_wm_steps 次 WM 步                           │
         │    每 WM 步：DreamZero lazy_joint_forward_causal           │
         │      · action  (8, 8) → flat 64 维                          │
         │      · video   imagined_frames=8 帧/块                     │
         │      · flow 去噪 K=16 步 → trace                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │ ③ Reward · LM Studio VLM（HTTP，任务 prompt + 滑窗视频）      │
         │    → complete ∈ {0,1}，finish_step                          │
         │    稀疏 reward：finish_step×64−1，值 complete×reward_coef   │
         │      reward_coef=5.0 → 成功约 +5，失败 0                     │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │ ④ GRPO · 同 uid 组内标准化 advantage                        │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │ ⑤ PPO · VAMPODPOActor                                       │
         │    ✅ 梯度：见下方「四机显存与微调策略」                        │
         │    ❄️ 无梯度：backbone · VAE · text/image enc · VLM         │
         │    存盘 policy.pt：仅 requires_grad 权重 + flow σ            │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
                                       └──────────▶ global_step+1
```

**cluster4 当前**（`vampo_ppo_trainer_cluster4.yaml`）：`train_batch_size=1` · `n_samples=4` · `max_wm_steps=8` → **4 traj/step · 32 infer/step** · `total_epochs=24576` · Megatron TP=4 · VideoMAE reward · `ppo_micro_batch_size=1`。

| 步骤 | 模型 / 产物 | 是否更新 | 说明 |
|------|-------------|----------|------|
| ③ Reward | LM Studio Qwen3-VL | ❄️ | HTTP 推理，不占训练 GPU |
| ④ GRPO | 无 | — | 只算 advantage |
| ⑤ PPO | 见「四机显存与微调策略」 | ✅ | 默认 `full`+DiT 在四机 GB10 上易 OOM |
| ⑥ 存盘 | `policy.pt` | — | 差分 ckpt |

## 四机显存与微调策略

四机 Ray 集群（各 1× GB10，~122GB 统一内存）为**数据并行**：每台各加载一份 DreamZero + Wan 子模块（T5/CLIP/VAE 冻结但仍占内存）。  
配置 `tune_diffusion_model: true` + `rl_fine_tune_mode: full` 时，Wan DiT 约 **16.5B** 可训参数；AdamW 状态 + 激活远超 122GB，**第一个 PPO backward 极易 OOM**。

推荐按显存从稳到激进选择（改 `configs/vampo_ppo_trainer.yaml` 或 cluster4 覆盖）：

### 方案 A · 只训 Projector（推荐先试）

冻结 Wan DiT，仅更新 action/state 编解码器；可训参数量级 **~10M**，四机 122GB 充裕。

```yaml
# configs/vampo_ppo_trainer.yaml → actor_rollout_ref.model
rl_fine_tune_mode: full
tune_projector: true
tune_diffusion_model: false   # 关键：DiT 冻结
keep_lora_trainable: false
```

| 项目 | 说明 |
|------|------|
| ✅ 训练 | `action_encoder` · `action_decoder` · `state_encoder` |
| ❄️ 冻结 | Wan DiT（~16.5B）· T5 · CLIP · VAE |
| 适用 | 四机集群默认开训；先验证 rollout + reward + PPO 闭环 |

### 方案 B · LoRA 微调 DiT

DiT 主干加 LoRA adapter（config 默认 `lora_rank=4`），只训低秩增量 + projector，显存远小于 16.5B 全量 Adam。

```yaml
rl_fine_tune_mode: lora
keep_lora_trainable: true
tune_projector: true
tune_diffusion_model: true    # LoRA 注入在 DiT 上
```

| 项目 | 说明 |
|------|------|
| ✅ 训练 | LoRA 层 + projector |
| ❄️ 冻结 | DiT 稠密权重 · T5 · CLIP · VAE |
| 适用 | 方案 A 效果不足、且仍无法做全量 DiT 时 |

### 不推荐（当前实现）

`rl_fine_tune_mode: full` + `tune_diffusion_model: true`：四机各 16.5B 可训 + 无跨卡 FSDP 切分 → **实际必 OOM**。  
若必须全量 DiT，需实现跨 4 卡 FSDP / optimizer offload（当前 `VAMPOActorRolloutRefWorker` 未做真实 FSDP）。

| 循环步骤 | 关键数值 | 代码入口 |
|----------|----------|----------|
| ① 采样 | `n_samples=8` · init 池 2888 | `dataset.py` |
| ② Rollout | flow `K=16` | `rollout.py` · `vla_policy.py` |
| ③ Reward | 滑窗 8，`eval_stride=4`，`reward_coef=5` | `vlm_reward.py` · `reward_manager.py` |
| ④ GRPO | 组大小 = `n_samples` | `core_algos.py` · `grpo.py` |
| ⑤ PPO | `σ_a=σ_v=0.05` | `actor.py` · `worker.py` |

## 与 WMPO 对照（简）

| | WMPO | VAMPO |
|---|------|-------|
| 想象环境 | 冻结 OpenSora WM + OpenVLA | **同一模型**联合生成 action+video |
| Reward | 训练 VideoMAE 分类器 | **LM Studio VLM** 零样本 judge |
| RL 改什么 | OpenVLA 全量 | **action head 可训子集**（见「四机显存与微调策略」）+ flow log prob |
| Rollout | WebSocket 或分离 WM | **in-process** `VLAPolicyModule` |

详见 [COMPONENT_RL.md](COMPONENT_RL.md) · [COMPONENT_REWARD.md](COMPONENT_REWARD.md).
