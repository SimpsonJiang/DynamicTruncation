# 仓库分析：架构、实验语义与验证

日期：2026-11-02（Asia/Shanghai）。基准 HEAD：`a10481d60fa226a6238eba72a21f66ff1c2ff9b6`，分析对象为当前工作区，包含修改过的 `buffer.py`、`run_uv.sh` 以及未跟踪的 `uv_run.sh`、`figures/`。本轮未修改算法、测试或启动脚本，未启动训练/评估环境；执行了一次 CPU 单元测试。

## 训练数据流

1. `train.py:21` 通过 Hydra 组合环境与模型配置，初始化日志、Buffer、环境和 Dreamer，调用 `OnlineTrainer.begin`。
2. `trainer.py:100` 管理多环境在线采集，按环境侧步数计数；更新频率由 `batch_size * batch_length / train_ratio * action_repeat` 决定（`trainer.py:19`）。评估调用 `agent.act(..., eval=True)`，actor 取 mode。
3. `trainer.py:156` 先从当前观测选择下一动作，再将动作和 posterior latent 存入 replay。`buffer.py:28` 采样 T+1 行，首行提供 cached latent，动作移位后配对后 T 行观测；`.clone()` 是当前未提交修改。
4. `dreamer.py:350` 采样数据、更新 slow targets，在 autocast 下计算联合梯度，经 GradScaler、AGC、LaProp、warmup scheduler 更新，并将 posterior latent 回写 replay。
5. `dreamer.py:405` 用 encoder/RSSM 得到 posterior，计算 dynamics/representation KL、representation variant、reward 和 continuation 损失。
6. `dreamer.py:477` 从所有 B*T 个 posterior starts 构建完整想象序列，进行 actor/value 学习；`dreamer.py:560` 的 replay value loss 保留通向 encoder/RSSM 的特征梯度。

关键形状：replay `[B,T,...]`；stochastic latent `[B,T,S,C]`；deterministic latent `[B,T,D]`；集成预测 `[K,B*T,H+1,1]`；截断分数 `[B*T,H,1]`。这里 K 指 head 数，C 指离散状态类别数，避免与其他文档的 K 混淆。

## 模型与扩展位置

| 关注点 | 入口 | 修改时应保持的契约 |
| --- | --- | --- |
| 新 uncertainty metric | `truncation.py:8`、`dreamer.py:40` | detached float32、每轨迹每深度、排除 bootstrap；同步配置校验与测试 |
| 截断策略 | `truncation.py:77`、`dreamer.py:508` | 滞后 EMA、首次越阈、最短前缀、边界 bootstrap、loss 归一化 |
| critic ensemble | `dreamer.py:95`、`:494`、`:594` | 独立在线 heads、逐 head slow target、共同 target、head 均值 loss |
| representation loss | `dreamer.py:118`、`:418` | dreamer decoder；r2dreamer Barlow 风格投影；infonce 对比；dreamerpro 增强与 EMA/prototypes |
| replay / collector | `buffer.py`、`trainer.py:156` | 动作时间索引、T+1 上下文、latent 回写、跨 episode flags |
| 环境与观测 | `envs/__init__.py:55`、`configs/env/` | reset、reward、is_first/is_last/is_terminal、action repeat |
| 向量日志 | `trainer.py:183`、`tools.py:198` | 新增 metric 名需接入 vector 分支，保持 JSON 数组与逐深度 TensorBoard tags |

`Dreamer.clone_and_freeze` 创建共享参数存储的冻结视图，`.to()` 后重新建立共享；它们不是慢速目标网络。`_slow_value` 才是独立参数的目标 heads。

## 动态截断的实际含义

`_imagine(start,H+1)` 完成后才计算分数和掩码。当前方法改变 return 递推及 imagined actor/value loss 的有效前缀，没有实现逐轨迹提前结束计算。

默认第一步分数分位数为 0.95；首轮直接使用当前阈值，之后使用前一轮 EMA 值再更新（decay=0.999）。掩码是小于等于阈值且有限的累乘，再强制保留最少 8 步；强制前缀内越阈也会阻断后续后缀。`mask_to_last` 保留边界状态的 bootstrap。`normalized_mask` 以全体位置的保留率归一化，不是分别对每条轨迹归一化。

每次更新对所有 batch/time 共享一次随机 head pair，用于想象 bootstrap 和 advantage；分数使用所有 heads。W1/Cramer 比较分布 CDF，使用 `symexp_twohot` 的原始回报 bins 间距；sigma 使用标量预测样本标准差；UCB 是均值加 beta 倍该标准差。因此 UCB 不等价于纯分歧分数。

关闭动态截断时仍计算所选分数并记录曲线。单头仅在关闭截断时允许；W1/sigma/Cramer 分数为零，UCB 为单头预测。

## 配置与当前实验

YAML 默认：`env=dmc_vision`、`model=size12M`、`rep_loss=r2dreamer`、heads=5、H=15、动态截断开启。

工作区 `run_uv.sh`：DMC proprio、dreamer、heads=5、H=15、wasser、截断关闭；`uv_run.sh`：dreamer、截断开启，遍历 wasser/sigma 和 horizon 数组。二者都显式设置 CUDA 模型与 replay 存储设备。

用户当前查看的 `logdir/new_VE_baseline_H15/dmc_finger_spin_proprio_s1_1102/.hydra/config.yaml` 确认为五头、H15、seed=1、截断关闭的 baseline。读取到的 step=12128 日志行中，学习率为 `4e-5`，ensemble std 为 `0.028880974277853966`；控制台两者显示为 `0.0` 是 `tools.py:189` 的一位小数格式。该观察仅是已有运行的快照，不能证明收敛、训练完成或方法增益。

## 评估、产物与复现边界

`OnlineTrainer.eval` 在训练期间评估；`eval_video.py:159` 为独立 checkpoint 视频评估入口，读取 `.hydra/config.yaml` 和 `latest.pt`，默认 seed 加 10000，关闭模型 compile，不支持 IsaacLab。生成视频不等价于 benchmark 完整评测。

`train.py:54` 仅在 `begin` 返回后保存 agent/optimizer state。未见周期 checkpoint 或完整恢复训练入口；scheduler、scaler、RNG、replay、collector 和 step 没有组成完整恢复协议。

实验比较应固定 observation modality、模型规模、环境预算、action repeat、train ratio、种子和表示学习分支；注明同时修改 buffer 或启动器的情况。没有进行外部论文/上游版本一致性认证。

## 本轮验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

Python 3.11.17；运行 41 个测试，耗时 23.646 秒，退出码 1，`FAILED (errors=12)`。其中一个方法的 10 个子测试及另外两个方法均因访问 `agent.config["beta"]` 报 AttributeError；实现持有 `agent.ucb_beta`。详见 [RISKS](RISKS.md) 和 [测试日志](unittest.log)。没有修改测试后重跑，也没有将其他测试通过解释为 CUDA/环境集成验证。

测试已覆盖多头独立性、loss/梯度、随机 pair、目标网络、分数性质、EMA、截断边界、checkpoint state roundtrip 和日志。真实 replay 在更新测试中被 mock，当前 buffer 修改没有端到端回归证据。

## 交付

长期有效的算法、数据和开发约定已整理到根目录 `AGENTS.md`。本次报告使用独立日期目录，保留原 `analysis_outputs/SUMMARY.md`、`RISKS.md`、`status.json` 及其他历史分析。

## 后续修复与复测（2026-11-02）

按用户要求，将 `tests/test_value_ensemble.py` 中 10 处 `agent.config["beta"]` 读写改为 `agent.ucb_beta`，训练实现未改动。使用上述相同 CPU 命令重跑三个测试文件：41 个测试全部通过，耗时 24.601 秒，退出码 0。日志见 [unittest_after_fix.log](unittest_after_fix.log)。前文失败结果及原测试日志保留为修复前记录。
