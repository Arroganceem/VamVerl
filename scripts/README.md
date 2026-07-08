# scripts/

按用途分类的运维与训练脚本（从仓库根目录执行）。

| 目录 | 用途 | 脚本 |
|------|------|------|
| `cluster/` | 四机 NFS、代码同步 | `mount_nfs_cluster4.sh` · `setup_nfs_exports_41.sh` · `sync_vamverl_cluster.sh` |
| `data/` | DROID / init_states / Component2 clip | `build_init_states_from_droid.sh` · `bootstrap_init_states.sh` · `extract_droid_lerobot_zips.sh` · `prep_component2_data_local.sh` |
| `checkpoint/` | DreamZero → FSDP 分片 ckpt | `convert_dreamzero_fsdp_checkpoint.py` · `convert_dreamzero_fsdp_cluster4.sh` |
| `train/` | 分布式训练启停 | `train_component2_reward_cluster4.sh` · `train_component3_rl_cluster4.sh` |
| `preflight/` | 开训前检查与分阶段验证 | `preflight_rl.sh` · `preflight_component2_train.sh` · `verify_staged.sh` |
| `dev/` | 调试与一次性 setup | `dump_wm_rollout.py` · `ensure_libero_config.py` |
| `ray/` | Ray 集群启停 | `start_ray_head.sh` · `start_ray_worker.sh` |

典型四机 RL 流程：

```bash
bash scripts/cluster/mount_nfs_cluster4.sh all
bash scripts/cluster/sync_vamverl_cluster.sh
bash scripts/ray/start_ray_head.sh
bash scripts/train/train_component3_rl_cluster4.sh
```
