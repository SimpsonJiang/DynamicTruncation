# Baseline 超参数审查

审查日期：2026-10-26，Asia/Shanghai。源码 revision：`2ff4ce7892d40f6d90aff2661d699a819a274d17`。工作区已有的 `run.sh` 修改仅新增 `imag_hori=15`，本次保留。未修改训练代码、配置或已有实验日志；本目录为新审查记录，保留上层原有分析。

结论：现有 `logdir/baseline` 的 9 个运行正确选择了本仓库的单 head Dreamer baseline。核心超参数与本仓库初始版本相符，没有发现误开 VE、动态截断、错选表征分支或种子间超参数不一致。此结论不等于与官方 DreamerV3 DMC preset 严格一致，也不等于训练性能已复现。

## 范围与证据

- 逐一检查 baseline 和 VE 的全部 18 份 `.hydra/config.yaml`，以及 baseline 的 9 份 `.hydra/overrides.yaml` 所代表的启动设置。
- 追踪 `run.sh → train.py → trainer.py → buffer.py → dreamer.py → rssm.py/networks.py/distributions.py → optim/`；检查 DMC 的环境构建、action repeat、time limit、flags、训练与独立评估环境。
- 读取 18 份 `metrics.jsonl` 的标量记录，以及 baseline 9 份 `latest.pt` 的 optimizer 元数据。checkpoint 使用 `weights_only=True, mmap=True, map_location=cpu`，没有恢复运行。
- 比较本地初始 revision `10f4afc` 的配置，以及 ensemble 引入前 revision `80ee48e` 的 Dreamer。
- 在线核对 R2-Dreamer 与 DreamerV3 官方仓库 main。main 未锁定历史 revision，只作当前参考，不据此认证原论文实验版本。

## 实际 baseline 参数

9 份 baseline 配置之间只有任务、seed 和 logdir 不同。

| 项目 | 实际设置 | 核对结论 |
| --- | --- | --- |
| 算法 | `model.rep_loss=dreamer` | 使用 decoder 重建分支；不是 R2/InfoNCE/Pro |
| Value | `value_ensemble_size=1` | 单 head bootstrap，无随机双 head 选择 |
| 动态截断 | `dynamic_trunc.enabled=false` | 全长想象；阈值配置不参与训练 |
| 模型容量 | `size12M`：deter 2048，hidden/units 256，stoch 32，classes 16 | 与原始 size12M 配置一致；名称不代表 proprio 实际恰好 12M 参数 |
| 数据 | `env=dmc_proprio`；encoder/decoder 仅 MLP keys | 图像仍采集，但不进入视觉编码/重建分支 |
| Batch | 16 × 64 | 每次更新 1024 个 replay 训练位置，另取每序列一个上下文位置 |
| Replay 容量 | 500000 条记录 | 包含 reset 行；不是累计环境步数 |
| 收集环境 | 16 | 与 batch size 数值相同，但含义不同 |
| 预算 | 510000 个环境步 | `step` 累计 action repeat；最终 eval 在 500000 |
| Repeat / time limit | 2 / 1000 | 每 episode 最多 500 次 agent 决策 |
| Train ratio | 512 | 每 4 个累计环境步安排一次更新；稳定阶段每 iteration 通常 8 次 |
| 初始化训练 | pretrain 0 | 首次训练时不额外预训练；`Every` 首次返回 1，随后按步差调度 |
| 想象长度 | 15 | 16 个 feature 位置，包括起点；15 个 return/loss 位置 |
| 折扣 | `horizon=333` | gamma = 1 − 1/333 ≈ 0.996997；不是 imag_horizon |
| Lambda / entropy | 0.95 / 0.0003 | 与原始配置一致 |
| KL | free nats 1，dyn 1，rep 0.1 | 两条 KL 的 detach 方向正确；unimix 差异见 RISKS |
| 其他 loss scales | recon/rew/con/policy/value 均 1，repval 0.3 | 重建权重按各观察 key 展开；repval 保留对 encoder/RSSM 的梯度 |
| 优化器 | LaProp，lr 4e-5，betas 0.9/0.999，eps 1e-20 | checkpoint 与配置一致，不是 Adam |
| Warmup / AGC | 1000 次 update；0.3，pmin 0.001 | warmup 单位为梯度更新；所有 trainable 模块共用 optimizer |
| Slow value | 每次 update 混入当前网络 0.02 | 独立 EMA；其余 frozen 网络共享当前参数存储 |
| Return normalization | 5%/95% 分位数，EMA rate 0.01，scale 下限 1 | 代码固定默认值，与当前官方配置的这些数值一致 |
| Reward/value | 255 bins 的 symexp two-hot；outscale 0 | 初始输出头为零，属于明确配置 |
| Actor | mean 经 tanh，std 0.1～1.0，outscale 0.01 | 采样不保证在 [-1,1]；见 RISKS 的下游核对限制 |
| Eval | 每 10000 环境步，10 个独立环境各一个 episode | eval_score 是这 10 个回合的平均值；动作取 mode，latent 仍随机采样 |
| 精度 | FP16 autocast + GradScaler；compile=True | 小模型测试不覆盖完整 CUDA 编译执行 |

baseline 与 VE 的保存配置仅有 `value_ensemble_size: 1→5` 和 `dynamic_trunc.enabled: false→true` 两项算法差异（另有 logdir）。因此当前 VE 分数代表两项改动的联合效果；不能单独归因给 ensemble。

此前另一个目录中 `dmc_hopper_hop_horizon_15_seed1` 的配置也做了抽查：共有的算法超参数与当前 baseline 相同，差别为新加的 ensemble/truncation 配置、设备与日志路径。未审计那个目录运行时源码版本。

## 验证结果

1. 现有 `tests/test_value_ensemble.py` 的 20 项小模型 CPU 测试全部通过，运行耗时 4.618 秒。覆盖单 head bootstrap、零 disagreement、梯度边界、slow target、loss 聚合及 update/checkpoint roundtrip，也覆盖多 head 和动态截断相关行为。
2. 数值对齐：将 `git show 80ee48e:dreamer.py` 在内存加载为参考类，以与测试相同的小模型、相同初始参数与随机种子，对比当前单 head、关闭截断的 `_cal_grad`。CPU FP32、compile disabled 下，全部 loss 最大绝对差为 **0**，全部 trainable 参数梯度最大绝对差为 **0**。不代表 GPU FP16/compile 的逐位等价。
3. 18 份 metrics 日志没有发现非有限的数值标量；9 份 baseline checkpoint 的所有 79 个 optimizer 参数状态均已更新，学习率、betas、eps 与配置一致，实际 optimizer step 在 126915～126921。日志中的 update counter 统计调用次数，FP16 GradScaler 可能跳过非有限梯度更新；没有逐次 skip 日志，无法由稀疏日志重建具体跳过时刻。
4. baseline 核心 9 个 Python 文件通过 AST 解析。没有启动训练、环境或策略评估，没有编写或修改测试。

## 如何界定“正确”

相对本仓库原始 baseline：核心 YAML 与 `10f4afc` 一致；全局配置仅将 buffer storage device 改为引用运行设备，对本次 cuda:0 运行解析结果无变化；新增的 ensemble/truncation 开关已由脚本正确覆盖。当前单 head 梯度路径与 `80ee48e` 对齐。

相对官方 DreamerV3 当前 main 的 `dmc_proprio` preset：不能称为完全一致。官方当前 preset 使用 size1m、repeat 1、train_ratio 1024、steps 1.1e6；本实验使用 size12M、repeat 2、train_ratio 512、steps 5.1e5。两者拥有许多相同的核心优化参数，但预算、模型和环境协议不同。当前本地代码还使用 FP16、uniform SliceSampler，以及不同的 continuation target 表达。这些需要作为复现条件记录，不能根据名称直接等同。

当前 R2-Dreamer 上游 main 相比本地历史实现已有新变化，例如 BF16、replay online/update_latents 配置。因此本报告优先使用本地固定 revision 判断是否出现回归，不把最新 main 当成历史实验真值。

## 后续建议

- 修正或删除无效的 `run.sh:12` 的 `imag_hori` 变量。若要用它控制训练，需要同时传入 `model.imag_horizon`，并写入生成的 worker 环境；只改声明不会生效。本次实际长度仍是 YAML 的 15。
- baseline 启动应始终显式指定 `rep_loss=dreamer`、`value_ensemble_size=1`、`dynamic_trunc.enabled=false`。仅指定 `rep_loss=dreamer` 会沿用当前 YAML 的 5 heads 与动态截断。
- 若目标是严格对齐官方 DreamerV3，先固定官方 revision 和实验协议，再单独验证 KL unimix、采样 estimator、replay 与精度差异。不要为了改变分数直接调整现有 baseline。
- 若要解释 VE 收益，补充“5 heads + 不截断”的消融；这是建议，本轮未运行。

## 外部参考

- [R2-Dreamer 默认模型配置](https://github.com/NM512/r2dreamer/blob/main/configs/model/_base_.yaml)
- [R2-Dreamer DMC proprio 配置](https://github.com/NM512/r2dreamer/blob/main/configs/env/dmc_proprio.yaml)
- [R2-Dreamer 当前 agent](https://github.com/NM512/r2dreamer/blob/main/dreamer.py)
- [R2-Dreamer 当前 RSSM](https://github.com/NM512/r2dreamer/blob/main/rssm.py)
- [官方 DreamerV3 配置](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)
- [官方 DreamerV3 agent](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py)
- [官方 DreamerV3 RSSM](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)

按项目指引设置代理后，shell 获取远端源码时代理连接被拒绝；官方参考通过网页工具读取。本地固定 revision 对比及测试不依赖网络。
