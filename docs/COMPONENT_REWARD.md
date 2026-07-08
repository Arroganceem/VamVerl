# Reward · VideoMAE（主路径）与 LM Studio VLM（可选）

> **正式 RL 训练默认：** `reward.backend: videomae`（`configs/vampo_ppo_trainer.yaml`）  
> **可选零样本路线：** `reward.backend: lmstudio`（下文 § LM Studio）

## VideoMAE success 分类器（主路径）

在线对想象 rollout 视频做 **8 帧滑窗** success 判定，输出 `complete` / `finish_step` → 稀疏 GRPO reward。

**实现：** `vampo/reward/videomae_reward.py` · `VideoMAERewardModel`  
**ckpt：** `checkpoints/videomae_droid.pth`（组件 2 离线训练，与 `rl_init` episode **disjoint**）

### 配置（verl RL）

`configs/vampo_ppo_trainer.yaml` → `actor_rollout_ref.reward`：

```yaml
reward:
  backend: videomae
  hf_model_id: /home/robotem/Models/videomae-base
  videomae_checkpoint: ${oc.env:VIDEOMAE_CKPT,./checkpoints/videomae_droid.pth}
  rm_threshold: 0.82
  window_size: 8
  min_steps: 32
  img_size: 224
```

环境变量：`VIDEOMAE_CKPT` 覆盖 ckpt 路径。

### 逻辑

```text
想象轨迹 video (T,H,W,C)
        │
        ▼
8 帧滑窗 → VideoMAE predict_success（threshold≈0.82）
        │
        ▼
首个 success → complete=1, finish_step=该窗末帧
        │
        └──▶ VAMPORewardManager → 稀疏 token reward（complete × reward_coef @ finish_step）
```

---

## LM Studio VLM（可选路线）

在线对想象/真实视频做 **任务语言条件** 的 success 判定，输出 `complete` / `finish_step`。

**服务地址：** `http://192.168.88.41:1234`（LM Studio OpenAI 兼容 API）  
**实现：** `vampo/reward/vlm_reward.py` · `LMStudioVLMRewardModel`

## 部署 LM Studio（41 节点）

1. 在 LM Studio 加载 [Qwen3-VL-8B](https://lmstudio.ai/models/qwen/qwen3-vl-8b)
2. 开启 Local Server，端口 **1234**
3. 确认 API 可用：

```bash
bash scripts/preflight/preflight_rl.sh --strict-reward
# 或仅测连通: curl http://192.168.88.41:1234/v1/models
```

## 配置（verl RL）

`configs/vampo_ppo_trainer.yaml` → `actor_rollout_ref.reward`：

```yaml
reward:
  backend: lmstudio
  lmstudio_base_url: http://192.168.88.41:1234/v1
  vlm_model: null          # null = 自动选 LM Studio 已加载模型
  window_size: 8
  min_steps: 32
  eval_stride: 4           # 滑窗步长（VLM 较慢，默认 4）
  frames_per_window: 4     # 每窗送 VLM 的采样帧数
  img_size: 384
  temperature: 0.0
  timeout_seconds: 120
```

环境变量（可选）：

| 变量 | 含义 |
|------|------|
| `LMSTUDIO_BASE_URL` | 覆盖 API 根路径 |
| `LMSTUDIO_VLM_MODEL` | 覆盖 model id |

## 逻辑

```text
想象轨迹 video (T,H,W,C) + task prompt
        │
        ▼
8 帧滑窗 (stride=eval_stride)，每窗采样 frames_per_window 帧
        │
        ▼
POST /v1/chat/completions  →  Qwen3-VL-8B  (yes/no)
        │
        ▼
首个 yes → complete=1, finish_step=该窗末帧
        │
        └──▶ VAMPORewardManager → 稀疏 token reward（complete × reward_coef @ finish_step）
```

## 代码入口

| 用途 | 路径 |
|------|------|
| VLM reward | `vampo/reward/vlm_reward.py` |
| 配置工厂 | `vampo/reward/factory.py` |
| rollout 调用 | `vampo/imagination/rollout.py` |
| verl worker 构建 | `vampo/integrations/verl/worker.py` |
