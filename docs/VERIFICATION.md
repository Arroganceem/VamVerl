# 分阶段验证（无需完整 RL 训练）

因 DreamZero ~14B + VideoMAE + Ray 同机易 OOM，**不能依赖一次完整训练**来验证代码。本仓库提供 **Stage 0–5** 分层检查：从静态编译到 VideoMAE 冒烟，逐步覆盖 RL 链路，且默认 **不加载 VLA 权重**。

## 快速命令

```bash
cd /home/robotem/WAM/VamVerl
pip install -e ".[verl,vla,dev]"

# GitHub / CI 默认：Stage 0–3（CPU，无大模型）
bash scripts/preflight/verify_staged.sh

# 含本地 DROID / init_states 检查
bash scripts/preflight/verify_staged.sh --through 1

# 含 VideoMAE CPU 推理（需 ckpt + backbone，仍不加载 DreamZero）
bash scripts/preflight/verify_staged.sh --through 4

# 开训前：等同 preflight --strict-reward
bash scripts/preflight/verify_staged.sh --through 5

# 单元测试（RL mock 链路）
pytest tests/test_staged_rl_pipeline.py -q
```

## 阶段说明

| Stage | 名称 | GPU | 验证内容 |
|-------|------|-----|----------|
| **0** | `static` | 否 | `compileall vampo/` · 核心模块 import |
| **1** | `data` | 否 | 配置 · DreamZero 目录 · DROID · episode split · init_states · 泄漏检查 |
| **2** | `rl-mock` | 否 | mock trajectory → `DataProto` → `VAMPORewardManager` → GRPO tie-break（**不加载 14B**） |
| **3** | `config` | 否 | yaml 必填字段 · `main_hydra` 入口可导入 |
| **4** | `videomae` | 可选 CPU | backbone + ckpt 存在 · `predict_success` 随机视频冒烟 |
| **5** | `preflight-strict` | 可选 | 完整 `preflight_rl.sh --strict-reward` |

**默认 `--through 3`** 适合 **GitHub 上传 / CI**：验证代码结构与 RL 数据流，不要求本机有 DROID 或 VideoMAE 权重。

## 各阶段与训练环节的对应关系

```text
Stage 0–3  ──►  代码可编译、RL 张量链路、GRPO、Hydra 入口
Stage 1    ──►  InitState / 数据划分（训练采样前置）
Stage 4    ──►  Reward 模型（VideoMAE 冻结 RM）
Stage 5    ──►  开训前最后一道门（含 strict VideoMAE）
完整训练   ──►  DreamZero rollout + PPO（需多卡 / cluster，OOM 风险最高）
```

## 环境变量

| 变量 | 用途 |
|------|------|
| `VAMVERL_ROOT` | 仓库根目录 |
| `MODEL_PATH` | DreamZero 基座（Stage 1/5） |
| `DROID_DATA_ROOT` | LeRobot DROID（Stage 1/5） |
| `INIT_STATES_DIR` | `data/init_states` |
| `VIDEOMAE_CKPT` | VideoMAE 权重（Stage 4/5） |
| `VIDEOMAE_BACKBONE` | VideoMAE backbone 本地目录 |
| `VIDEOMAE_VERIFY_DEVICE` | Stage 4 设备，默认 `cpu` |

## JSON 输出（CI）

```bash
bash scripts/preflight/verify_staged.sh --json | jq '.github_ready'
```

## 与 `preflight_rl.sh` 的关系

- `preflight_rl.sh`：训练前 **数据 + 环境 + reward** 一次性检查。
- `verify_staged.sh`：**分层、可部分运行**；Stage 2 的 RL mock 是 preflight 没有的 **纯代码链路验证**。

推荐流程：

1. 开发机 / CI：`verify_staged.sh`（0–3）+ `pytest`
2. 集群数据就绪：`verify_staged.sh --through 1`
3. VideoMAE ckpt 就绪：`verify_staged.sh --through 4`
4. 四机开训前：`verify_staged.sh --through 5` 或 `preflight_rl.sh --strict-reward`

## GitHub 上传清单

上传前建议：

1. 运行 `bash scripts/preflight/verify_staged.sh` 与 `pytest tests/test_staged_rl_pipeline.py`
2. 确认 `.gitignore` 已排除 `checkpoints/*.pth`、`data/` 大文件、`wandb/`、`outputs/`
3. 大权重通过 README 说明从外部路径 / ModelScope 获取，**不要提交进 git**
4. 可选：推送后 GitHub Actions 跑 `.github/workflows/staged-verify.yml`

## 已知限制

- **Stage 2 不覆盖**：Megatron TP、Ray 多机、真实 DreamZero flow 去噪、LM Studio VLM reward。
- **Stage 4 CPU 较慢**：仅验证能 forward，不评估 F1。
- **完整 PPO 一步**仍需 cluster；OOM 时优先用 Stage 2 + 小 `max_wm_steps` / `n_samples` 做 smoke train（见 `vampo_ppo_trainer_cluster4.yaml`）。
