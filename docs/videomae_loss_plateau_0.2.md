# VideoMAE Reward：Loss 计算逻辑与为何卡在 0.2

## 1. Loss 计算逻辑

训练入口：`vampo/reward/train_videomae.py`。

### 1.1 前向与目标

模型是 2 类 `VideoMAEForVideoClassification`：

- 输入：预计算 clip `pixel_values`，形状约 `[B, T, C, H, W]`（`T=window=8`）
- 输出：`logits`，形状 `[B, 2]`
- 标签：`ys`，整型 `0=failure`（视频开头），`1=success`（视频末尾）

每步核心代码：

```python
logits = model(pixel_values=vids).logits
loss = criterion(logits, ys)
```

### 1.2 损失函数

```python
class_weights = _class_weights(cfg, device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

即 **带可选类别权重的交叉熵**（等价于加权负对数似然）：

\[
\mathcal{L}
= -\frac{1}{B}\sum_{i=1}^{B}
  w_{y_i}\,\log\bigl(\mathrm{softmax}(z_i)_{y_i}\bigr)
\]

其中 \(z_i\) 是第 \(i\) 个样本的 logits，\(y_i\in\{0,1\}\)，\(w_c\) 是类别 \(c\) 的权重。

配置项 `train.pos_weight`：

| 取值 | 行为 |
|------|------|
| `null` / `false` | 无权重，`weight=None`，普通 CE |
| 数值（如 `2.0`） | `weight = [1.0, 2.0]`，负类：正类 = 1 : 2 |
| `"auto"`（当前默认） | 读 `videomae_dataset_ready.json`，`weight = [1.0, neg/pos]` |

当前数据近似平衡：

- train `success_clips=157850`，`failure_clips=157672`
- `neg/pos ≈ 1.0` → `weight ≈ [1.0, 1.0]`

因此 **当前 loss ≈ 普通未加权 CE**，不是加权抬高的假象。

### 1.3 日志里各字段含义

| 字段 | 含义 |
|------|------|
| `loss_raw` | 本 rank 当前 batch 的 CE |
| `loss` | DDP 各 rank `all_reduce(AVG)` 后的 step loss |
| `loss_ema` / 进度条上的 `loss=` / `ema=` | 对 `loss` 做 EMA（`loss_ema_beta=0.99`），用于 smoothed 显示，不参与反传 |
| `pos` / `batch_pos` | 当前 batch（DDP 平均）中正类比例 |

反传路径：`loss.backward()` → 可选 `grad_clip=1.0` → `optimizer.step()`。EMA 只用于日志。

---

## 2. Loss=0.2 在数学上意味什么

对未加权 CE，单样本 \(-\log p_y\)，batch 均值约 0.2：

\[
\mathbb{E}\bigl[-\log p_y\bigr] \approx 0.2
\quad\Rightarrow\quad
\text{平均正确类概率 }
p_y \approx e^{-0.2} \approx 0.82
\]

也就是说：模型对正确类别平均置信度约 **82%**。  
要 intuitive 再降到 0.05，需要平均 \(p_y \approx 0.95\)，几乎所有样本都高置信判对。

随机猜两类：\(\mathrm{CE}\approx \log 2 \approx 0.693\)。  
从 0.69 降到 0.2，说明已经学到不少；**卡在 0.2 不一定是 bug**。

历史对照：旧标签/不平衡数据时 train EMA 卡在约 0.31 且 val F1≈0.03；新数据后 EMA≈0.2 且 step1000 val F1≈0.78 / step2000≈0.79，说明分类有效，CE 没必要压到 0。

---

## 3. 为何继续压不下去（主要原因）

### 3.1 标签构造：不是“真失败 episode”，是“同段成功视频的头 vs 尾”

数据定义（`videomae_dataset_ready.json`）：

```text
label=1: video end (finish + pos_near)
label=0: video start, same count as success
```

且：

```json
"episode_success": 6314,
"episode_failure": 0
```

即：

- **全部训练 episode 都是 success episode**
- `failure` clip = **同一段成功过程的开头**
- `success` clip = **同一段成功过程的末尾（finish + pos_near）**

模型学的是：**成功轨迹「开头 vs 结尾」**，不是「真实失败任务 vs 成功任务」。

DROID 场景里开头/结尾经常共享：

- 同一相机、同一桌面/物体布局
- 机械臂姿态可能只差一点点
- `pos_near` 会把「接近结束但仍像中段」的帧标成 success

这些构成 **不可约误差（Bayes error）**：部分样本客观上难分到自信满分，CE 自然有地板。

### 3.2 类别权重不是天花板原因

当前 `pos_weight: auto` 且正负 clip 数接近，权重约 `[1,1]`。  
地板不是「正类被放大权重」造成的。

### 3.3 优化面是次要因素

| 配置 | 值 | 影响 |
|------|-----|------|
| backbone `lr` | `1e-5` | 全量微调但步子小，后期变平 |
| `head_lr` | `1e-4` | 分类头略大 |
| `max_epochs` | `1`（当前 yaml） | 只跑一轮，易看起来“卡住” |
| `steps_per_epoch` | `3000` | 相对全量 windows（~31.5万）是子集遍历 |
| `grad_clip` | `1.0` | 防爆，不是平台期主因 |
| cooldown | 每 30 step 睡 50s | 拖慢墙钟时间，不改变 loss 地板 |

优化器慢会让曲线更平，但 **0.2 的量级更像任务难度，不是单纯学习率太小**。

### 3.4 指标：该看 Val F1，不是硬棒 CE→0

Component2 用途是 reward 分类；验收以 **val F1 / best_thresh** 为准。  
CE=0.2 + F1≈0.78–0.79 已经说明判别力够用；继续死磕 train loss 到极低，容易过拟合头尾视觉捷径，泛化到真失败场景反而变差。

---

## 4. 结论

1. **Loss = 加权交叉熵**，当前数据下权重≈1，即普通 CE。
2. **0.2 ≈ 平均正确类概率 0.82**，不是异常爆炸或写错公式。
3. **压不下去的主因**：标签是成功视频「开头 vs 末尾」，`episode_failure=0`，存在视觉不可分样本，存在 CE 地板。
4. **不要用 train loss 是否到 0 验收**；用 val F1（及后续 RL 侧 reward 是否有用）验收。

---

## 5. 若仍想让 CE 再降（可选方向）

按收益 / 改动成本排序：

1. **引入真实 failure episode**（任务未完成 / 中断），让 label=0 对应真失败而非仅开头，再重建 clip 数据。
2. **减弱过近的 pos_near**（例如只保留 video end 一帧为 success，video start 一帧为 failure），减少边界模糊。
3. 更长训练：`max_epochs` / `max_steps` 加大，必要时略提 `lr`（慎用，防崩）。
4. 观察 train/val CE：若两者同卡在 0.2，是数据地板；若 train≪val，是过拟合再谈正则。

不要仅靠关闭 `pos_weight` 来“降 loss”：当前权重已近 1，几乎无贡献，且与“位置辅助损失”无关。
