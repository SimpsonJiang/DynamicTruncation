# DreamerV3 baseline 静态核对点

复核日期：2026-10-16。对象：本地 `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`。

以下区分“源码可确定的行为/差异”与“尚未验证的影响”。没有运行训练、环境、autograd instrumentation 或数值对齐测试，任何性能影响均不能视为确认的缺陷。未修改项目代码。本报告更新了原有六项检查，特别修正了将 cached latent、imagination root mask 和未使用参数简单视作疑似问题的判断。

## 优先核对：采样分布与 KL 的 unimix 不一致

- 本地证据：[OneHotDist](../distributions.py#L16) 将 softmax 概率与 1% 均匀分布混合，用于状态采样与 entropy；[RSSM.kl_loss](../rssm.py#L222) 却把原始 `post_logit/prior_logit` 直接交给 [distributions.kl](../distributions.py#L266)，后者未执行 unimix。
- 官方对照：[RSSM.loss / _dist](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py) 的 KL 经过 `_dist`，其中传入 `unimix`。
- 可以确定：本地 KL 对应未混合的 categorical，与采样/entropy 对应的分布不同；形状和 stop-gradient 方向正确并不能消除这一差异。
- 未验证：是否为作者有意简化、对稳定性/分数的实际影响。后续最小验证可固定 logits，对比混合前后 KL 及梯度，尤其检查极尖锐分布；无需先跑训练。

## 需明确的采样梯度 estimator

- 本地证据：[OneHotDist.rsample](../distributions.py#L31) 使用 `F.gumbel_softmax(..., tau=1, hard=True)`，前向为 one-hot，反向使用带 Gumbel 噪声的 softmax relaxation。
- 可以确定：不能仅描述为“categorical sample 然后直接透传概率梯度”；二者前向 categorical 行为可以一致，反向 estimator 不同。baseline world-model 的 posterior recurrent/reconstruction 路径经过这一梯度。
- 未验证：尚未追查本轮官方依赖包内 OneHot 的反向实现，因此不将其断言为相对某一固定官方版本的已确认偏差。imagination 本身 `no_grad`，不受这一反向差异直接影响。

## 连续动作的采样范围

- 本地证据：[bounded_normal](../distributions.py#L217) 仅对均值 tanh，返回普通 `Independent(Normal)`；采样值可超出 `[-1,1]`。同文件的 `Bound` wrapper 未被该工厂使用。[NormalizeActions.step](../envs/wrappers.py#L42) 只做线性映射，没有 clip。
- [Deter.forward](../rssm.py#L44) 在 RSSM 输入处将动作按绝对值归一到范围内；这并不能证明环境收到的动作与 RSSM 输入动作一致。
- 未验证：具体 benchmark 下游是否裁剪，以及是否影响环境/模型对齐。后续应观察实际动作分布并查对应环境实现，而不是根据函数名推断 bounded sampling。evaluation 取 Normal mode，范围问题主要涉及训练时采样。

## continuation 和 imagination weight：比较时必须成对看

- 本地证据：[continue target](../dreamer.py#L431) 为 `1-is_terminal`；[imagination](../dreamer.py#L454) 令 `gamma=1-1/horizon`，使用 `cumprod(gamma*imag_cont)`，lambda return 也显式乘 gamma。
- 官方对照：[agent.py 的 con loss 和 imag_loss](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py) 配合 [configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml) 的 `contdisc` 选项；不能只比较其中一行。
- 本地根动作 weight 是 `gamma*cont_0`。**不能据此直接认定多乘了一步 continuation**：官方也可以包含 root continuation，以抑制 terminal roots；本地所有 posterior states 都进入 imagination，这一做法也出现在官方 starts 路径中。
- 未验证：在固定 reference、明确 contdisc 模式下，target、return 和 loss weight 三者的数值等价性。后续比较时应同时固定 continuation 预测含义和 gamma 的位置；不要先“修复”为首项 1。

## replay 采样协议与论文不完全相同

- 本地证据：[Buffer](../buffer.py#L14) 仅配置 TorchRL ReplayBuffer + SliceSampler；未发现将新收集的非重叠序列优先填入 minibatch 的显式 online queue。
- 论文对照：[Nature Experience replay](https://www.nature.com/articles/s41586-025-08744-2) 描述 uniform replay 与 online queue 结合。
- 可以确定：本地代码未实现论文所描述的这一显式混合路径。未验证 TorchRL 具体版本的采样分布以及作者所用复现实验设置，不能据此断言性能不合格。
- 其他比较边界：本地默认 size12M；论文正文报告默认 size200M。本地各 benchmark 的预算、train_ratio 和 action_repeat 也必须单独核对；小模型命令不代表论文的完整实验协议。README 的约 5×/1.6× 加速属于仓库声明，本轮未测量。

## cached latent 与跨 episode 序列是有意设计的迹象

- [Buffer.sample](../buffer.py#L39) 从第 0 行 cached posterior 初始化，训练第 1…T 行；[Dreamer.update](../dreamer.py#L345) 再将新 posterior 写回对应 replay 行。因此中途切片的上下文可能来自旧参数；回写的 state 也由本次 optimizer step 之前的 forward 产生。
- [Nature Experience replay](https://www.nature.com/articles/s41586-025-08744-2) 同样描述缓存与回写 latent。这是需要理解的近似，而非本项目独有的可疑操作。
- [trainer.py](../trainer.py#L116) 的 `episode_ids` 实际为恒定环境槽 ID。采样序列允许跨 episode；[RSSM.obs_step](../rssm.py#L161) 依赖 `is_first` 重置，[lambda return](../dreamer.py#L560) 依赖 `is_last/is_terminal`，reset reward 由 [ParallelEnv](../envs/parallel.py#L55) 置零。
- 风险边界：新环境必须保留上述 flags/reward 契约。尚未做 TorchRL 环形存储和所有环境适配器的运行时审计。

## 配置名称与真实行为

- [rssm.initial 配置](../configs/model/_base_.yaml#L66) 写 `learned`，但 [RSSM.initial](../rssm.py#L133) 始终创建零状态；`self._initial` 被保存但没有用于选择初始化路径。可以确定：配置标签不能解释为实际存在 learnable initial parameter。
- `norm` 并非在每个模块里都有动态控制效果。例如 [MLP](../networks.py#L325) 与 RSSM 多处固定使用 RMSNorm，`norm=False` 不会统一关闭所有归一化。[ConvEncoder](../networks.py#L192) 则根据实际输入和下采样层数推导空间大小，没有读取配置的 `minres`。只应按具体构造函数判断；本轮未逐项审计全部配置。
- `_lambda_return` 的 `value` 参数只参与 shape assertion，实际递推使用 `boot`。官方当前同名函数也有这一现象，因此本身不是本地算错 return 的证据。阅读时需要显式区分参数名与实际使用的数据。

## checkpoint 尚不是完整恢复协议

- [train.main](../train.py#L54) 仅在 `begin()` 返回后保存 `latest.pt`，包含 agent 和 optimizer state；未发现定期保存和入口加载逻辑。
- 未发现显式保存 scheduler、GradScaler、环境/collector 状态、replay 或训练 step 的完整恢复路径；部分模型 buffer 会随 agent state_dict 保存，这不等于完整实验状态。
- 可以确定：当前入口不提供完整断点续训流程。对长期实验是工程限制；本轮未验证手动恢复方案，也不修改现有行为。

## 外部参照的证据限制

论文已查阅 arXiv v2（2024）及 Nature（2025）；官方源码为本轮网页可读取的 main，未锁定原始实验 commit。不能把这里的结构对照当成严格版本 parity 认证。

本轮按 AGENTS.md 设置代理后尝试通过 shell 获取官方 revision，但沙箱网络访问返回 `Operation not permitted`，因此改用网页工具阅读主来源。此处未请求额外权限，也没有因网络问题运行任何替代训练。
