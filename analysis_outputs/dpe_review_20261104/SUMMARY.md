# Diverse Projection Ensembles 与本仓库不确定度设计

日期：2026-11-04（Asia/Shanghai）。静态代码分析与论文阅读；没有运行训练、评估或回归测试。
工作区 HEAD：`a4ec04b23284157a42bbac0a3f33349c64890fca`，存在用户未提交修改。

## 论文依据

阅读 ICLR 2024 正式版本的相关正文、理论与附录：
[Diverse Projection Ensembles for Distributional Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2024/file/cfea00ddf5a5f514bd3368d639d1933d-Paper-Conference.pdf)。

The paper combines categorical and quantile representations, estimates local uncertainty through average pairwise W1, and learns a propagated exploration bonus. Shared bootstrap targets motivate distinguishing local disagreement from total value error. The formal bounds require contraction assumptions and account for a projected fixed point; disagreement is an estimator, not a guaranteed error bound. Appendix A.4 discusses persistent projection errors. Appendix B.2 uses another distributional ensemble for the bonus: the reported four models should not be read as four extrinsic-value members. Relevant locations: Sections 4.2, 5.1, 6.3 and Appendices A.4, B.2–B.3.

以下为当前代码事实及本仓库适配建议，不是论文已经验证的 Dreamer 结论。

## 当前实现与关键差别

| 内容 | 当前证据 | 对设计的含义 |
|---|---|---|
| 同构成员 | `dreamer.py:94-101`，每个成员独立构造 MLPHead，复用同一 critic 配置 | 共享 RSSM feature，不是只有最后一层独立；也没有独立世界模型 |
| 相同表示 | `configs/model/_base_.yaml:179-191`，255 bins 的 symexp_twohot | 成员没有 categorical/quantile 表示差异 |
| 非均匀支持 | `distributions.py:242-251` | symexp 构造原始回报坐标的 bins；不能把这些 bins 误称为在 symlog 坐标上计算距离 |
| 共同标量目标 | `dreamer.py:498-500,522-540,594-608` | 随机两头均值用于共同 lambda-return；各成员另有自己的 slow-value 标量正则目标 |
| replay 同样耦合 | `dreamer.py:566-580` | 使用全体标量均值和 imagination bootstrap 构造 replay 目标，repval 梯度会进入共享表示 |
| 分布距离 | `truncation.py:47-85` | 相同支持上按 bin 宽度积分 CDF 差，softmax/CDF/积分为 float64，输出 float32 |
| 截断 | `dreamer.py:484,503-514`；`truncation.py:111-145` | 完整想象后，以局部分数和首步分位数 EMA 筛选前缀；不是计算上的提前停止 |

目前的输出确实是离散分布，但训练路径是“标量 lambda-return → two-hot 标签”，并非对完整下一时刻 return distribution 做 Bellman 推送。该分布可能反映随机训练目标、近似和正则效应；不能直接宣称它是经过校准的真实完整回报分布，更不能把分布宽度直接解释成环境随机性。

## 最重要的设计判断

1. **先检验分歧是否能预测错误。** 共享 feature、奖励模型、转移模型与共同目标，允许所有 heads 一起出错。这是结构允许的盲区，尚未证明当前实验已经发生。
2. **分开研究表示多样性与距离选择。** wasser/cramer/sqrt_cramer 只改变评分函数，不改变成员表示、数据或训练目标。若成员共同偏差很强，更换评分不能创造缺失的信息。
3. **区分 critic 误差与 dynamics 误差。** 当前分数直接测量的是 value-distribution disagreement；把它用作 rollout 信任度，需要独立验证。异构 critics 也不能保证识别单一 RSSM 的共同错误。
4. **当前 UCB 分数不适合作为纯分歧对照。** `truncation.py:28-34` 返回 mean + beta*std。即使所有 heads 一致，高 value 仍可能对应高 score。优先把 sigma、W1、Cramer 作为分歧诊断，UCB 单列为不同的决策启发式。

## 可检验的改进方向

### A. 优先进行冻结模型诊断

复用 `physics_probe/pipeline.py:349-440` 的同动作真实续跑机制。当前代码已经计算 uncertainty 和截断边界，但 row 没有存每步 uncertainty，而且 `next_reward_abs` 比较的是解码恢复分支与真实分支，并非 Dreamer reward head 与真实 reward。新增诊断时需要明确这两个差别。

建议记录逐起点、逐深度的原始 W1/Cramer/sqrt-Cramer/sigma，head 均值、逐对距离、阈值、保留标记，以及以下分开的误差：

- 同一想象动作序列下，Dreamer reward head 预测和真实环境奖励之差；先核对 reward 与 feature/action 的时间对齐。
- 1/5/15 步折扣累计奖励预测误差。这是有限前缀误差，不是完整 value 误差。
- 若研究 value 误差，另用真实闭环策略的多次 continuation 估计回报，并记录有限 horizon、bootstrap 和 Monte Carlo 方差。
- decoder 物理状态误差及恢复后的动力学误差，作为辅助指标；它们混合了世界模型与 decoder 误差。

各任务、checkpoint、深度分别评估 Spearman 相关和高误差识别；不要仅计算混合全部深度的相关，否则二者随深度共同增长即可产生虚假支持。检查 risk–coverage：保留同等比例样本时，分歧排序是否比仅按深度筛选更能降低误差。

数据按 episode 分割；阈值在校准集确定，测试集不调参。统计不确定性按 episode/model seed 聚类，不能把同一轨迹的每个点当作独立样本。RSSM 随机采样应做重复或固定随机性对照。已有 stage2 恢复报告没有使用 decoder/世界模型，不能充当上述相关性证据。

### B. 异构表示作为第二步

保留现有五头 baseline，增加预算匹配的四头对照：4 categorical、4 quantile、2 categorical + 2 quantile。这里 2+2 是本仓库候选方案，不是对原论文模型计数的复述。控制 shared feature、训练数据、总参数量或报告参数差、更新次数、reward scale、seeds 与截断规则。

先在冻结 RSSM 的辅助 critics 上比较预测与分歧质量，再考虑进入 actor bootstrap 和 repval。这样可避免把 critic 表示变化与共享 feature 学习变化混为一谈。

两条路线需分别命名：

- 轻量异构回归：categorical 与 quantile 拟合同一批标量 lambda-return 样本，用来测试表示归纳偏置。单个确定性标签上 quantiles 可收缩到同一点，不能假定换头自然产生正确的 return distribution。
- 完整分布式目标：保留下一状态的分布支持和概率，构造分布目标并设计 multi-step/lambda、continuation 和 slow regularization。这会改变训练算法，应作为独立实验，不能声称只是替换一个输出层。

异构支持下必须重写距离接口：当前实现取第一个成员的 bins.diff()，只适用于共享同一有序支持。可在所有 atoms 的并集上精确积分离散 CDF；采样统一 quantile 网格可作为近似，但要报告积分误差。不同成员的 logits 不能逐元素直接比较。

### C. 校准表示偏差，再考虑时间聚合

至少分别记录 categorical 组内、quantile 组内和跨组距离。构造已知分布的受控拟合问题，测量充分拟合后仍有多少表示差异。使用真实 posterior 状态的校准分布作参考，但不能自动把 posterior 上的所有分歧都减掉：其中也可能有真实的不确定性。

当前首步 95% 分位数 EMA 提供一个参考尺度，但不是按回报范围、状态区域条件化的误差校准。比较新旧分数时除原始阈值实验外，还需要匹配有效 horizon/保留率的对照。

可作为后续启发式实验的前缀预算：对非负且校准后的局部分数 u，定义

`C_h = sum_{t=0}^{h-1} gamma**t * u_t`。

让累计分数超过预算后停止保留，并在需要时加入 continuation 权重。这个公式不是论文定理对 Dreamer 的直接推论：当前 u 可能已经包含远期价值分歧，累加可能重复计数。先验证它相对于单点越阈是否改善误差–保留长度权衡，再考虑学习额外的不确定度 critic。

还应把“当前一步想象不可信”和“截断边界 bootstrap value 不可信”分开记录；截短轨迹不自动解决边界价值误差。

## 建议顺序与验收依据

冻结模型校准诊断 → 异构辅助 critics → 表示偏差校准 → 在线截断实验 → 可选时间传播。
最低限度需要无截断、固定 horizon、原始动态截断和候选动态截断对照；匹配平均 horizon 的固定截断能帮助判断收益是否只是缩短轨迹。

成功标准应是更好的误差排序、同等保留率下更低的真实预测误差，以及跨 seeds 的控制性能收益。单纯距离增大、阈值增大或截断变早都不足以支持方法改进。

本轮仅新增分析文档；用户现有修改和历史分析未改动。
