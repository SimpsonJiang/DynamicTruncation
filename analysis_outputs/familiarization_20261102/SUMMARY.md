# 项目熟悉记录（2026-11-02）

范围：当前工作区源码、配置、启动脚本与已有运行日志。HEAD 为 `7c499a0a6de330e50a24f7419c521adf88c260d1`；存在未提交修改，本记录不代表该 commit 的纯净状态。未运行训练、评估或测试，未修改源码和历史报告。

## 结构与训练数据流

- `train.py:20`：Hydra 配置入口，创建日志器、Buffer、环境和 Dreamer；训练正常返回后保存 `latest.pt`。
- `trainer.py:100`：在线采集、训练频率、周期评估；采集后根据当前观测选择动作，并存入 replay。`episode` 是固定环境槽 ID，跨 episode 的 RSSM 重置依靠 `is_first`。
- `buffer.py:28`：采样 `[B,T+1,...]`，首行提供 cached latent；后 T 行观测与前 T 行动作配对，更新后回写 latent。
- `dreamer.py:389`：编码与 RSSM posterior、世界模型损失、想象 actor/critic 损失、replay `repval` 联合反向传播。
- `rssm.py`：确定性状态与离散随机状态；特征为二者拼接。`networks.py` 和 `distributions.py` 提供网络、分布及 return 归一化。
- `eval_video.py:159`：加载运行目录配置和 checkpoint，生成策略视频；不支持 IsaacLab。

## 算法约定

`rep_loss=dreamer` 使用重构，`r2dreamer` 使用冗余约束投影；另有 InfoNCE、DreamerPro 分支。默认模型配置为 size12M、5 个 critic heads、H=15、开启截断；当前实验脚本覆盖为 dreamer。

每次更新随机选一对 heads 的标量均值用于想象 bootstrap 和 advantage，全部 heads 用于不确定性。各 head 共享 return target，使用各自 slow target，损失在 heads 间平均。想象轨迹及 targets 不反传世界模型，replay `repval` 保留特征梯度；frozen 网络与对应在线网络共享参数存储。

`truncation.py` 支持 wasser、ucb、sigma、cramer、sqrt_cramer。输出 detached float32 `[B*T,H,1]`，排除最后 bootstrap state。W1/Cramer 比较完整回报分布，sigma 比较标量值分歧，UCB 为均值加 beta 乘样本标准差。

阈值使用第一步分数的 95% 分位数及滞后 EMA（0.999），至少保留 8 步，首次越阈后不恢复；边界保留 bootstrap。掩码在整个 batch/time 上按均值归一化。完整 H+1 想象先生成，再应用截断，因此不能由有效 horizon 下降直接推断 rollout 计算量下降。

## 当前实验脚本

| 文件 | 当前配置 |
| --- | --- |
| `run_uv.sh` | uv；wasser，H15，5 heads，开启截断；finger_spin/hopper_hop/quadruped_run；seeds 1/42/100；GPU 6/7，每卡最多 3 作业 |
| `uv_run.sh` | uv；wasser/sigma，H15，5 heads，开启截断；同三项任务；seeds 200/300；GPU 2/3，每卡最多 3 作业 |
| `run.sh` | Conda；ucb/sigma，H15，5 heads，开启截断；walker_walk；seeds 1/42/100；GPU 4/5，每卡最多 2 作业 |

这些是文件内容，并非进程运行状态证明。实验实际参数以保存的 `.hydra/config.yaml` 为准。

## IDE 所指运行目录的日志快照

目录：`logdir/new_VE_wasser_H15/dmc_finger_spin_proprio_s1_1102`。

保存配置确认：dreamer、5 heads、wasser、H15、开启截断、seed 1、预算 510000、action repeat 2、train ratio 512。

读取 `metrics.jsonl` 时：最新训练指标 step 17120，有效 horizon 14.248046875，保留率 0.9498698115，ensemble std 0.0197349470，学习率 0.00004；最近已写入评估为 step 10016、score 10.9000005722。控制台尾部为 `Evaluating the policy...`，当时未找到 `latest.pt`。这些观测不能证明进程正在运行、卡死或训练完成，也不构成最终实验效果判断。

IDE 中的 `figures/dreamerv3_vs_new_ve_baseline_h15.pdf` 当前磁盘路径不存在，未分析该图。

## 验证范围

当前三个测试文件包含 41 个 test 方法，覆盖 ensemble、分数、梯度、截断边界、checkpoint roundtrip 和向量日志。历史 `analysis_outputs/review_20261102/unittest_after_fix.log` 记录 41 tests、24.601s、OK；本次未重跑，不将其视为当前工作区验证结果。真实 replay、CUDA 编译和完整训练续跑不在本次验证范围内。
