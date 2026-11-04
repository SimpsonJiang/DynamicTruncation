# 风险与证据边界

日期：2026-11-02。以下区分运行时已复现的问题、源码可确定的行为，以及尚未验证的影响。本次仅记录，不修改实现或测试。


后续状态（2026-11-02）：首项测试接口问题已按用户要求修复，10 处引用改为 `agent.ucb_beta`。三个测试文件共 41 个测试复测全部通过；详见 [复测日志](unittest_after_fix.log)。下文记录的是修复前证据，其他风险与验证边界仍适用。

## 已修复：测试访问过期接口

- 证据：`tests/test_value_ensemble.py:189,209,306` 使用 `agent.config["beta"]`；`Dreamer.__init__` 没有保存 `self.config`，而在 `dreamer.py:43` 保存 `self.ucb_beta`。
- CPU unittest 共 41 个测试，12 条 AttributeError；10 条来自同一个方法的 metric/horizon 子测试，另外两条来自两个 UCB 测试方法。
- 结论：当前测试套件不能标为通过。这是测试与模型接口不匹配，不足以推断 UCB 数学实现错误。
- 后续：核对这些断言应使用的配置来源或 `agent.ucb_beta`，修复测试接口后重跑；本次没有自动执行修复。

## 已确认：截断不提前结束 rollout

- 证据：`dreamer.py:484` 生成 H+1 步，`:503` 才计算分数，`:508` 才应用截断；`_imagine` 的循环不读取阈值或掩码（`:617`）。
- 结论：effective_horizon 是学习前缀长度，不是实际计算步数；关闭截断仍计算 uncertainty。
- 未验证：实际 wall time、GPU 峰值内存及不同 metric 的开销。若目标包含计算加速，需额外设计与测量，不能依据有效长度直接宣称提速。

## 已确认：checkpoint 不提供完整续训协议

- 证据：`train.py:52` 先完成整个训练，`:54` 才保存；入口未加载 checkpoint。保存 agent 与 optimizer 不包含完整 replay、RNG、collector/step、scheduler/scaler 恢复协议。
- 影响：中途异常不会由该路径产生新的 checkpoint，已有 checkpoint 不能直接当作可无缝续训的证明。
- 测试边界：`tests/test_value_ensemble.py:749` 的 roundtrip 只覆盖模型和优化器状态，且 replay 被 mock。

## 验证缺口：真实 replay、GPU 与环境

- `buffer.py:41` 的动作移位 `.clone()` 是工作区已有修改；未更改。训练 collector 保存 outgoing action，采样后移位为 incoming action，阅读注释时以执行顺序为准。
- `trainer.py:116` 的 episode ID 固定为环境槽，允许序列跨 episode；`rssm.py:158` 用 `is_first` 重置，return 依赖 `is_last/is_terminal`。
- 单元测试未使用真实 TorchRL 环形 buffer 验证 wraparound、索引回写、别名及跨 episode 动作对齐；没有运行 CUDA compile 或真实多进程环境。
- 后续修改采样/存储时，应以带可辨认时间编号的小序列检查输出，覆盖环形边界与 reset，避免凭静态索引推断完全正确。

## 静态事实，算法影响待验证

| 事实 | 本地证据 | 尚未验证/建议核对 |
| --- | --- | --- |
| sampling/entropy 使用 unimix，KL 使用原始 logits | `rssm.py:219,222`；`distributions.py:17,266` | 混合前后 KL/梯度差异及其训练影响；不能据此直接宣称算法失效 |
| bounded_normal 仅 tanh 均值，仍为普通 Normal | `distributions.py:217`；`envs/wrappers.py:42` | 采样可越界；需检查具体环境下游处理与 RSSM 输入归一化是否保持一致 |
| `rssm.initial='learned'` 未切换初始化路径 | `rssm.py:87,133` | 当前 initial 始终为零张量；配置不能解释为存在可学习初态 |
| proprio 仍生成、传输、保存 image | `envs/dmc.py:72,85`；`trainer.py:149,165`；`buffer.py:26` | 只关闭 encoder CNN 不会停渲染；显存/吞吐成本需测量，不能在未验证时删除视频数据路径 |

## 日志与实验解释风险

- `tools.py:189` 仅输出一位小数，`metrics.jsonl` 保留数值与分数向量。当前 baseline 的 step=12128 行中 lr=4e-5、std≈0.02888，而控制台均显示 0.0；不能据此断定没有学习或 heads 完全相同。
- `value_ensemble_std` 使用总体标准差（`unbiased=False`），`sigma`/UCB 使用样本标准差；两个指标也有不同的时间聚合范围，不能直接互换。
- `run_uv.sh` 当前关闭截断，而 YAML 默认开启；同名方法目录不能代替保存的 Hydra 配置。日期命名只精确到天，同日重复相同任务/seed 可复用目录；使用独立 logdir 避免混入日志。
- 历史报告引用较早 revision，其外部论文对比和旧行号没有在本轮重新验证。本文仅依据当前本地代码和一次测试运行，不主张训练收益、baseline parity 或论文结论。
