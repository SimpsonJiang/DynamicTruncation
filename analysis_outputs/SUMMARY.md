# 项目架构与 DreamerV3 baseline 阅读指南

本轮复核日期：2026-10-16（Asia/Shanghai）。分析对象为本地 revision `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`。仅静态阅读项目代码和配置、查阅论文及官方源码，没有启动环境、训练或评估，没有修改项目源代码。保留原有的英文张量流审计，置于本文后半部分。

## 1. 先确定正在读哪一个算法

这是 R2-Dreamer 的实现仓库，也提供共享训练框架下的 Dreamer、InfoNCE、DreamerPro 分支。默认配置为 `env=dmc_vision`、`model=size12M`、`model.rep_loss=r2dreamer`。

Dreamer baseline 的配置示例（仅说明配置，本轮未执行）：

```bash
python3 train.py env=dmc_vision env.task=dmc_walker_walk model=size12M model.rep_loss=dreamer logdir=./logdir/dreamerv3
```

算法分支位于 [dreamer.py](../dreamer.py#L78) 和 [Dreamer._cal_grad](../dreamer.py#L378)：

| `model.rep_loss` | 表征约束 | 新增模块 |
| --- | --- | --- |
| `dreamer` | 重建观察，包括图像 MSE / 向量 symlog MSE | `MultiDecoder` |
| `r2dreamer` | posterior feature 投影与 encoder embedding 的相关矩阵约束；目标 embedding detach | `Projector`，无 decoder |
| `infonce` | 投影 feature 与 detached embedding 的对比交叉熵 | `Projector`，无 decoder |
| `dreamerpro` | 增强视图、EMA encoder、prototype/Sinkhorn 目标 | 投影、原型、EMA 模块 |

四条分支共用 encoder、RSSM、reward/continue、actor/value、imagination、replay value learning 和优化流程。先理解 `dreamer`，再回来看另外三条表征分支，阅读成本最低。

注意两个同名概念：`model.rep_loss` 是算法选择字符串；`losses["rep"]` 是所有分支都有的 representation KL。局部变量 `rep_loss` 在 `kl_loss` 调用处指后者。

## 2. 项目地图与入口

| 文件/目录 | 责任 | 优先阅读位置 |
| --- | --- | --- |
| `train.py` | Hydra 入口、随机种子、logger、buffer/env/agent/trainer 构建；训练结束保存 | `main`，第 21 行 |
| `trainer.py` | 环境交互、策略推理、入库、更新频率、定期评估 | `begin` 第 100 行；`eval` 第 27 行 |
| `buffer.py` | TorchRL 序列采样、一个上下文状态、action 移位、latent 回写 | `sample` 第 28 行 |
| `dreamer.py` | 统一 agent：世界模型、行为学习、全部 loss 和一次更新 | `_cal_grad` 第 349 行；`act` 第 245 行 |
| `rssm.py` | 分块 GRU 确定状态、离散 prior/posterior、observe/imagine、KL | `obs_step` 第 158 行；`Deter` 第 10 行 |
| `networks.py` | 观察 encoder/decoder、MLP 分布头、BlockLinear、return EMA | `MultiEncoder` 第 99 行；`MLPHead` 第 339 行 |
| `distributions.py` | one-hot 采样、two-hot 回归、MSE、连续动作分布、KL | `OneHotDist` 第 16 行；`TwoHot` 第 67 行 |
| `optim/` | LaProp 和 adaptive gradient clipping（AGC） | `laprop.py`、`agc.py` |
| `envs/` | 各 benchmark、统一 flags/action 接口、进程并行；IsaacLab 独立路径 | `make_envs`、`ParallelEnv.step`、`wrappers.py` |
| `configs/` | 全局 batch/buffer/trainer；模型容量/共同超参数；环境协议 | `configs.yaml` → `model/size12M.yaml` + `model/_base_.yaml` + `env/dmc_vision.yaml` |
| `runs/` | 多任务、多 seed 的启动脚本，脚本中多数 `METHOD` 默认是 R2-Dreamer | 修改算法时需核对脚本实际传参 |
| `tools.py` | 日志、权重初始化、调度计数、种子及辅助函数 | `Every` 第 286 行解释 update ratio |

推理入口为 `Dreamer.act()`；评估嵌在 `OnlineTrainer.eval()` 中。未发现独立的 checkpoint 驱动评估 CLI，也未发现训练入口的恢复逻辑。`tools.py` 存在 optimizer 加载 helper，不等于训练主程序支持 resume。

## 3. 架构：有观察时推断状态，无观察时预测状态

以下是从本地实现归纳的数学对应关系，`e_t` 为 encoder 输出，`f_t` 为 actor/value/head 的共同输入：

```text
e_t = Encoder(x_t)
h_t = Deter(h_(t-1), z_(t-1), a_(t-1))
q_t = Posterior(h_t, e_t)     # 真实数据：用观察修正状态
p_t = Prior(h_t)              # 想象数据：没有观察，用先验采样
z_t ~ q_t 或 p_t
f_t = concat(flatten(z_t), h_t)

Decoder(f_t) -> x_hat_t       # 仅 dreamer 分支；训练表征
Reward(f_t)  -> reward distribution
Cont(f_t)    -> continuation probability
Actor(f_t)   -> action distribution
Value(f_t)   -> return distribution
```

论文中 `q(z_t | h_t, x_t)` 的 encoder 概念，对应本项目 `MultiEncoder` **加上** RSSM 内部的 `_obs_net`，不能把 `MultiEncoder` 的连续 embedding 误认为已经采样出的 `z_t`。

`h_t` 是确定性递归记忆，`z_t` 是多组离散 categorical one-hot 状态。`Deter` 使用八个块的 GRU 风格门控，同时通过输入投影混合全局信息，并非八个完全互不通信的模型。

默认视觉配置的主要尺寸：

| 张量 | 形状/数值 | 对应实现 |
| --- | --- | --- |
| replay 图像 | `[16,64,64,64,3]`，即 `[B,T,H,W,C]` | batch 16、序列 64、图像 64×64 |
| 图像 embedding | `[B,T,1024]` | Conv+MaxPool 四级，通道 32→48→64→64，空间 64→32→16→8→4 |
| posterior `stoch` | `[B,T,32,16]` | 32 个 categorical group，每组 16 类；不是 32×32 |
| posterior `deter` | `[B,T,2048]` | `size12M.deter` |
| `feat` | `[B,T,2560]` | `32*16+2048` |
| imagined feature | `[1024,16,2560]` | `N=B*T` 个起点，包含起点及 15 步后续状态 |
| imagined return | `[1024,15,1]` | 每个起点 15 个 action/value 训练位置 |
| reward/value logits | 最后一维 255 | 对指数间隔 bins 的 two-hot 回归 |

`preprocess` 将图像除以 255；`ConvEncoder` 再减 0.5，decoder 则 sigmoid 输出 `[0,1]` 以对齐重建目标。向量观察通过配置选择 MLP 并使用 symlog。环境依旧可能返回未被 encoder 选择的字段；选择由 `mlp_keys/cnn_keys` 正则控制。

## 4. 一次更新与梯度去向

`Dreamer.update()` 采样并预处理，更新 slow value，然后在 AMP 下进入 `_cal_grad()`。baseline 的总目标为本地 loss 标量的加权和：

```text
L = L_recon + L_reward + L_continue
  + 1.0 L_dyn + 0.1 L_rep
  + 1.0 L_policy + 1.0 L_value + 0.3 L_repval
```

重建按 observation key 单独命名并加权。世界模型两个 KL 的方向都是 `KL(q || p)`，区别是 stop-gradient 的位置：`L_dyn=max(1,KL(sg(q)||p))`，`L_rep=max(1,KL(q||sg(p)))`。低于 free-nat 阈值时，该项的局部梯度为零。实际 KL 未使用 unimix，见风险报告。

| 分支 | 输入是否 detach | 更新哪些部分 |
| --- | --- | --- |
| 重建/reward/continue/KL | posterior 计算图保留；KL 按方向 detach | 各 head、RSSM、encoder；局部 KL 分支按 stop-gradient 分工 |
| imagination 生成、reward/value target、lambda return | 起点 detach；rollout 和 return 为 `no_grad` | 不直接产生梯度 |
| policy loss | imagined feature/action/advantage detach | actor；`log π(a|f)*adv + entropy` 的 score-function 目标 |
| imagined value loss | imagined feature、return、slow value target detach | value |
| replay `repval` | posterior feature 保持连接；return/slow target detach | value + RSSM + encoder |

**这里 actor 没有通过 imagined dynamics 反向传播。** 连续与离散动作均走当前 actor 的 log-probability 目标，不应套用其他 Dreamer 版本的 dynamics-gradient 讲解。

`return_ema` 跟踪 return 的 5%/95% 分位数，advantage 除以至少为 1 的分位差。`imag_horizon=15` 控制 rollout 长度；`horizon=333` 定义 `gamma=1-1/333`；两者不是同一个量。lambda 为 0.95。

所有 trainable 模块使用一个 LaProp optimizer、一次 backward，外层再做 GradScaler unscale、AGC、step、warmup scheduler。`_frozen_*` 与当前模块共享 parameter storage，提供不累积参数梯度的计算视图；它们不是落后若干轮的 target network。真正独立缓慢更新的是 `_slow_value`，每次更新按 0.02 混合当前 value。

## 5. 需要先理解的数据时间轴

采集阶段存储的行是 `(x_t, r_t, a_t, z_t, h_t)`：reward 来自到达当前观察的上一个动作，存储的 action 却是当前刚选出的下一动作。`trainer.py` 第 159 行注释不足以表达实际赋值，应读第 156/161 行。

`Buffer.sample()` 采样 65 行：第 0 行 cached latent 为上下文，后 64 行为训练观察；action 列左移一行。因此 RSSM 真正收到的是 `(previous state, a_(t-1), x_t)`，这个对齐是刻意设计。

`episode` 是恒定的环境槽 ID，序列可以跨真实 episode；`is_first` 重置 RSSM，`is_last` 截断 replay lambda trace，`is_terminal` 取消 bootstrap。时间限制终止可以 `is_last=True` 且 `is_terminal=False`，不能把两个 flag 合并。新环境接入时这组约定比 observation 字段名更关键。

默认 `train_ratio=512` 表示每收集一个决策步，训练 512 个 replay 时间位置；不是每步做 512 次更新。由于 `B*T=1024`、action repeat 为 2，稳定阶段约每 4 个底层环境步触发一次梯度更新；16 个环境一起前进时可在一次循环中补做多次 update。初始化/预热附近由 `Once/Every` 另行控制，不应据此推断总更新数精确值。

## 6. 对照论文与官方实现时的版本边界

已查阅 [arXiv v2（2024）](https://arxiv.org/html/2301.04104v2)、[Nature（2025）论文](https://www.nature.com/articles/s41586-025-08744-2)，以及官方 [RSSM](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py)、[agent](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py)、[配置](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)。外部源码为本轮网页读取的 main，不是作者原始实验 commit，也未完成逐版本历史审计。

根据 block GRU、RMSNorm、SiLU、symexp-binned two-hot、LaProp、latent replay 回写及 repval 的组合，**本地 baseline 更适合按后期 DreamerV3 实现阅读**；这是结构对应判断，不是对移植来源或数值等价性的证明。Nature 论文也明确列出这一组架构与优化技术。

本地 `symexp_twohot` 在 `symexp(linspace(-20,20))` 产生的原始数值 bins 上插值，再计算分布加权均值；没有先 symlog target 再均匀插值，也没有对加权均值再 symexp。不要仅凭名称将其解释成早期实现的 symlog 空间 two-hot。

整体范式一致不代表逐项一致。已记录 KL/unimix、continuation/weight 约定、replay online queue 缺失等比较点；采样 estimator 与训练协议亦需单独核对。见 [RISKS.md](RISKS.md)。README 的速度声明未在本轮验证。

## 7. 后续带读顺序与每轮目标

1. **闭环地图**：`train.main` → `OnlineTrainer.begin` → `Dreamer.act`。讲清楚环境状态、agent 记忆和 replay 分别存什么。
2. **一个 transition 的时间轴**：`ParallelEnv.step` → `Buffer.sample` → `RSSM.obs_step`。手画连续三步，解释 action 移位与三个 flags。
3. **RSSM 本体**：先看 `obs_step/img_step/get_feat`，再展开 `Deter.forward`、`BlockLinear`、`OneHotDist`。区分 deterministic memory、posterior、prior。
4. **世界模型目标**：`_cal_grad` 第 365 行起到 reward/continue。逐项追输入、目标、stop-gradient；最后展开 encoder/decoder。
5. **想象与行为学习**：`_imagine` → `_lambda_return` → policy/value losses。手算一个短 horizon，确认 `r_(t+1)` 与 bootstrap 的下标。
6. **replay value 与优化**：`repval` → `update` → frozen view/slow value → AMP/AGC/LaProp。画清哪些 loss 可以训练 encoder。
7. **返回项目贡献**：只替换 `_cal_grad` 的表征分支，比较 R2/InfoNCE/DreamerPro 如何改变训练信号与计算开销。

每轮按“目的 → 公式 → 调用关系 → 张量 shape → 梯度 → 下标例子”展开即可，无需先顺序通读 721 行的 `dreamer.py`。

保守的扩展位置：改序列模型看 `rssm.py`；改观察编码/重建看 `networks.py`；改表征目标看 `_cal_grad` 分支及 loss scales；改行为学习看 imagination/policy/value/repval；接新环境看 `envs` 的 action/flag 契约。本轮仅标注位置，不提出代码改动。

---

# DreamerV3 baseline tensor-flow audit（保留的详细张量审计）

## Scope and evidence

- Repository revision: `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`.
- Analysis is static and read-only with respect to project code. No training, evaluation, or environment was executed.
- The repository default is `model.rep_loss: r2dreamer`; its DreamerV3 baseline is selected with `model.rep_loss=dreamer` ([README.md](../README.md#L28), [configs/model/_base_.yaml](../configs/model/_base_.yaml#L9)). The encoder/RSSM/reward/continue/actor/value path is shared; the `dreamer` branch adds the observation decoder and reconstruction loss ([dreamer.py](../dreamer.py#L78), [dreamer.py](../dreamer.py#L378)).

Shape symbols follow [docs/tensor_shapes.md](../docs/tensor_shapes.md): replay batch `B`, replay length `T`, stochastic groups `S`, categories `K`, deterministic width `D`, action width `A`, embedding width `E`, and feature width `F=S*K+D`. Let `N=B*T`, configured imagination horizon `H`, and returned imagination sequence length `L=H+1`.

For the default `dmc_vision + size12M` configuration: `B=16`, `T=64`, `S=32`, `K=16`, `D=2048`, `F=2560`, `E=1024` for a 64x64 image, `H=15`, `L=16`, and `N=1024` ([configs/configs.yaml](../configs/configs.yaml#L17), [configs/model/_base_.yaml](../configs/model/_base_.yaml#L3), [configs/model/_base_.yaml](../configs/model/_base_.yaml#L54), [configs/model/size12M.yaml](../configs/model/size12M.yaml#L4), [networks.py](../networks.py#L197), [networks.py](../networks.py#L218)). `A` is environment-dependent.

## End-to-end layered call chain

```text
train.main
  -> Buffer(config.buffer) + make_envs(...) + Dreamer(...)
  -> OnlineTrainer.begin(agent)
       -> envs.step(previous action, done)
       -> Dreamer.act(observation, recurrent state)
            -> preprocess -> frozen encoder
            -> frozen RSSM.obs_step(previous posterior, previous action, observation embedding)
            -> frozen actor(feature) -> action
       -> store {observation/reward/flags, action, posterior stoch/deter}
       -> Dreamer.update(replay_buffer)
            -> Buffer.sample: sample T+1, split cached state[0] + data[1:T+1],
                              shift action to original [0:T]
            -> Dreamer._cal_grad
                 [World model]
                 -> encoder(data) -> embedding
                 -> RSSM.observe -> posterior stoch/deter/logits
                 -> RSSM.prior(posterior deter) -> aligned prior logits
                 -> KL(dynamic, representation)
                 -> decoder reconstruction (only rep_loss=dreamer)
                 -> reward(feature), continue(feature)

                 [Latent imagination]
                 -> flatten and detach all B*T posterior states
                 -> _imagine(H+1) with frozen RSSM + frozen actor
                 -> frozen reward/continue/value/slow-value targets
                 -> _lambda_return -> imagined return and detached advantage
                 -> current actor(imagined feature) -> policy loss
                 -> current value(imagined feature) -> imagination value loss

                 [Replay value representation learning]
                 -> first imagined return per posterior state as bootstrap
                 -> _lambda_return over replay rewards/flags
                 -> current value(attached posterior feature) -> repval loss

                 -> weighted sum of all losses -> one backward pass/one optimizer
            -> Buffer.update sampled positions with detached fresh posterior states
```

Construction and top-level dispatch are at [train.py](../train.py#L21), [train.py](../train.py#L39), [train.py](../train.py#L45), and [train.py](../train.py#L51). Collection/update dispatch is at [trainer.py](../trainer.py#L100), [trainer.py](../trainer.py#L147), [trainer.py](../trainer.py#L156), [trainer.py](../trainer.py#L161), and [trainer.py](../trainer.py#L176). The complete optimization body is [Dreamer._cal_grad](../dreamer.py#L349).

## Replay and posterior time alignment

The collection loop stores each row after policy inference. Therefore storage row `j` contains current observation `x_j`, reward `r_j` caused by the previous action, newly selected outgoing action `a_j`, and posterior state `s_j` inferred from `x_j` ([trainer.py](../trainer.py#L147), [trainer.py](../trainer.py#L156), [trainer.py](../trainer.py#L161)). Reset rows carry `is_first` and zero reward in the parallel environment path ([envs/parallel.py](../envs/parallel.py#L55), [envs/parallel.py](../envs/parallel.py#L59)).

`Buffer.sample()` samples `T+1` consecutive storage rows `q ... q+T`, then constructs:

```text
initial       = cached latent from storage row q
data.obs[t]   = storage observation q+t+1
data.reward[t]= storage reward      q+t+1
data.flags[t] = storage flags       q+t+1
data.action[t]= storage action      q+t       (explicit one-step shift)
                                      t = 0 ... T-1
```

This follows directly from [buffer.py](../buffer.py#L20), [buffer.py](../buffer.py#L32), [buffer.py](../buffer.py#L39), [buffer.py](../buffer.py#L40), and [buffer.py](../buffer.py#L41). Thus the apparent collection-time pairing “observation plus outgoing action” is deliberately rewritten into the RSSM pairing “previous action plus current observation.”

At replay index `t`, `RSSM.observe()` calls:

```text
h_t = Deter(z_{t-1}, h_{t-1}, data.action[t])
q(z_t | h_t, embed(data.obs[t]))
p(z_t | h_t)
```

`is_first[t]` zeros the incoming stochastic state, deterministic state, and previous action before the deterministic transition ([rssm.py](../rssm.py#L140), [rssm.py](../rssm.py#L148), [rssm.py](../rssm.py#L158), [rssm.py](../rssm.py#L161)). The prior is not a second rollout: `prior(post_deter)` evaluates `p(z_t|h_t)` at every posterior deterministic state, so `prior_logit[:,t]` is aligned with `post_logit[:,t]` for the two KL directions ([dreamer.py](../dreamer.py#L369), [dreamer.py](../dreamer.py#L371), [rssm.py](../rssm.py#L222)).

## Imagination and lambda-return time alignment

Every posterior state is flattened into `N=B*T` independent start states and detached ([dreamer.py](../dreamer.py#L438)). `_imagine(start, H+1)` records the current feature and its action before applying `img_step`; consequently returned index `k` is state/feature `f_k` and outgoing action `a_k`, while the next state is produced after both have been appended ([dreamer.py](../dreamer.py#L533), [dreamer.py](../dreamer.py#L539), [dreamer.py](../dreamer.py#L547)). With `H=15`, returned arrays have indices `0...15`; actor/value losses consume only `0...14`. The transition after action 15 is computed but its resulting state is not returned or consumed.

Reward and continuation are state predictions. Return `G_k` for action `a_k` uses the reward and liveness at the next imagined state, i.e. array index `k+1`. For general input arrays, the implementation is:

```text
live_j  = gamma * (1 - terminal_j)
trace_j = lambda * (1 - last_j)
G_k = reward_{k+1}
    + (1 - trace_{k+1}) * live_{k+1} * boot_{k+1}
    + trace_{k+1} * live_{k+1} * G_{k+1}
```

The recursion is initialized with the last `boot` value and returns one fewer time step ([dreamer.py](../dreamer.py#L554)). For imagination, `last=0`, `terminal=1-imag_cont`, and `boot=imag_value`, producing `[N,H,1]` returns ([dreamer.py](../dreamer.py#L448), [dreamer.py](../dreamer.py#L457), [dreamer.py](../dreamer.py#L459)). The formal `value` argument is checked for shape but is not otherwise read by `_lambda_return`.

For replay return learning, the first imagined return from each of the `B*T` starts is reshaped to `[B,T,1]` and used as `boot`; replay `reward/is_last/is_terminal` then yield `[B,T-1,1]` targets ([dreamer.py](../dreamer.py#L497), [dreamer.py](../dreamer.py#L504), [dreamer.py](../dreamer.py#L509)). A true terminal makes `live=0`; a time-limit truncation (`is_last=1`, `is_terminal=0`) cuts the lambda trace while retaining a one-step value bootstrap. Current terminal rows are excluded from `repval` by `weight=1-is_last` ([dreamer.py](../dreamer.py#L508), [dreamer.py](../dreamer.py#L514)).

## Key tensor table

“Gradient” describes whether the tensor lies on a path to trainable parameters in the single final backward pass, not whether the tensor itself is a leaf parameter.

| Name | Source / exact location | Shape | Meaning / time index | Main consumers | Gradient participation |
|---|---|---:|---|---|---|
| sampled storage slice | `SliceSampler`, [buffer.py](../buffer.py#L14) | `[B,T+1,...]` after view | Consecutive context plus training rows | split into `initial` and `data` | No; replay data |
| `initial=(stoch,deter)` | storage row 0, [buffer.py](../buffer.py#L39) | `[B,S,K]`, `[B,D]` | Cached posterior immediately before `data[:,0]` | `RSSM.observe` | No; stored tensors, later refreshed only when sampled as data |
| `data` | slice rows 1...T, [buffer.py](../buffer.py#L40) | TensorDict batch `[B,T]` | Current observations, rewards and flags | encoder, RSSM, reward/continue targets, replay return | No |
| `data["action"]` | original rows 0...T-1, [buffer.py](../buffer.py#L41) | `[B,T,A]` | Previous action entering each current observation/state | `RSSM.observe` | No |
| `embed` | `MultiEncoder(data)`, [dreamer.py](../dreamer.py#L367), [networks.py](../networks.py#L138) | `[B,T,E]` | Current observation embedding | posterior network; decoder-independent auxiliary loss | Yes, to encoder through posterior/world-model losses; baseline reconstruction also trains encoder through posterior |
| `post_deter` | deterministic transition in `obs_step`, [rssm.py](../rssm.py#L167) | `[B,T,D]` | `h_t` after previous action, before observing `z_t` | posterior logits, prior logits, feature | Yes, through RSSM and recurrently to earlier posterior/encoder states |
| `post_logit` | observation-conditioned `_obs_net`, [rssm.py](../rssm.py#L171) | `[B,T,S,K]` | logits for `q(z_t|h_t,x_t)` | posterior sample and representation KL | Yes; detached only on dynamic-KL left side |
| `post_stoch` | straight-through Gumbel sample, [rssm.py](../rssm.py#L175), [distributions.py](../distributions.py#L31) | `[B,T,S,K]` | categorical posterior state `z_t` | next RSSM step, feature, decoder, imagination starts | Yes in world-model/replay-value paths; imagination start copy is detached |
| `prior_logit` | `_img_net(post_deter)`, [dreamer.py](../dreamer.py#L371), [rssm.py](../rssm.py#L189) | `[B,T,S,K]` | aligned `p(z_t|h_t)` logits; sampled prior from this call is discarded | dynamic KL, entropy metric | Yes through dynamic KL; detached for representation KL |
| `dyn_loss` / `rep_loss` | [rssm.py](../rssm.py#L222) | `[B,T]` | KL with posterior detached / prior detached respectively, then free-nat clip | total loss | Yes on opposite KL branches; entries below `kl_free` have zero local gradient from the clip |
| `feat` | concat flattened posterior stochastic plus deterministic, [dreamer.py](../dreamer.py#L377), [rssm.py](../rssm.py#L211) | `[B,T,F]` | posterior feature at replay state `t` | decoder, reward, continue, start states, replay value | Yes in world-model and `repval` branches |
| reconstruction distributions | `MultiDecoder(post_*)`, [dreamer.py](../dreamer.py#L378), [networks.py](../networks.py#L172) | per observation key `[B,T,*obs_shape]` | Dreamer baseline observation prediction at the same state/time | negative log probability reconstruction losses | Yes to decoder and posterior world model; exists only for `rep_loss=dreamer` |
| reward prediction | `self.reward(feat)`, [dreamer.py](../dreamer.py#L430) | logits `[B,T,255]`; log-prob `[B,T]` | predicts replay `reward_t` at posterior state `t` using symexp-binned two-hot | reward loss | Yes to reward head and posterior world model |
| continuation target/prediction | `1-is_terminal`, `self.cont(feat)`, [dreamer.py](../dreamer.py#L431) | target `[B,T,1]`; Bernoulli batch `[B,T]` | probability episode continues after reaching this state | continue loss; frozen copy later supplies imagined continuation | Yes in replay world-model loss |
| `start` | flattened detached posterior, [dreamer.py](../dreamer.py#L438) | `[N,S,K]`, `[N,D]` | each replay posterior becomes an independent imagination root | `_imagine` | No |
| `imag_feat`, `imag_action` | frozen actor/RSSM rollout, [dreamer.py](../dreamer.py#L533) | `[N,L,F]`, `[N,L,A]` | `feat_k` and outgoing `action_k` at same imagined index | frozen targets; current actor/value | No; `_imagine` is `no_grad` and outputs are detached |
| `imag_reward` | frozen reward mode, [dreamer.py](../dreamer.py#L448) | `[N,L,1]` | predicted reward attached to imagined state; return for action `k` uses item `k+1` | lambda return, metrics | No; fixed target path |
| `imag_cont` | frozen continue mean, [dreamer.py](../dreamer.py#L450) | `[N,L,1]` | predicted liveness at imagined state | return discount and loss weighting | No; fixed target path |
| `imag_value`, `imag_slow_value` | frozen value copies, [dreamer.py](../dreamer.py#L452) | each `[N,L,1]` | current-value snapshot and EMA slow-value prediction | return bootstrap / value regularizing target | No; fixed targets |
| `weight` (imagination) | cumulative `imag_cont*gamma`, [dreamer.py](../dreamer.py#L454) | `[N,L,1]` | prefix survival/discount weight; loss uses first `H` entries | actor and imagination-value losses | No; detached explicitly at loss |
| imagined `ret` | `_lambda_return`, [dreamer.py](../dreamer.py#L459) | `[N,H,1]` | `G_k` for actions/states `k=0...H-1`, using next-state reward/continue/value | advantage, value target, replay bootstrap | No; function is `no_grad` |
| `adv` | `(ret - imag_value[:,:-1])/ret_scale`, [dreamer.py](../dreamer.py#L462) | `[N,H,1]` | percentile-range-normalized advantage; EMA offset is not subtracted here | actor loss | No at actor loss (`adv.detach()`); scale/return already detached |
| `policy`, `logpi`, `entropy` | current actor on detached imagined features, [dreamer.py](../dreamer.py#L466) | log-prob/entropy `[N,H,1]` after slice | likelihood and entropy of frozen-policy sampled actions 0...H-1 | `policy` loss | Yes, only to current actor parameters |
| `imag_value_dist` | current value on detached imagined features, [dreamer.py](../dreamer.py#L472) | logits `[N,L,255]`; used log-prob `[N,H,1]` | fits detached lambda return plus detached slow-value target | imagination `value` loss | Yes, only to current value parameters |
| replay `boot` | first imagined return per replay start, [dreamer.py](../dreamer.py#L504) | `[B,T,1]` | horizon-H model-based bootstrap for every replay posterior | replay lambda return | No |
| replay `ret` | `_lambda_return` on replay data, [dreamer.py](../dreamer.py#L509) | `[B,T-1,1]` | target for posterior features 0...T-2 using reward/flags at 1...T-1 | `repval` | No |
| `value_dist` (replay) | current value on attached posterior feature, [dreamer.py](../dreamer.py#L513) | logits `[B,T,255]`; used `[B,T-1]` | predicts detached replay return and detached slow-value target | `repval` loss | Yes to current value and, intentionally, through `feat` to RSSM/encoder |
| `total_loss` | scaled sum of named losses, [dreamer.py](../dreamer.py#L525) | scalar | world model + representation + actor + value + replay-value objective | AMP backward and one LaProp optimizer | Yes; all trainable modules are updated together |

## Actor and critic objectives

The actor loss is exactly:

```text
mean(weight_k * -(log pi_current(a_k | imag_feat_k) * stopgrad(adv_k)
                  + act_entropy * entropy(pi_current_k)))
```

for `k=0...H-1` ([dreamer.py](../dreamer.py#L466), [dreamer.py](../dreamer.py#L470)). Since imagination actions come from the frozen/shared actor under `no_grad`, this is a score-function objective; it does not differentiate through actions, dynamics, reward, continue, or return.

The imagination critic loss fits both the lambda-return target and the slow value prediction with two-hot log-probabilities ([dreamer.py](../dreamer.py#L472)). The replay `repval` objective has the same two-target form, but its feature input remains attached so it is explicitly also a representation/world-model training signal ([dreamer.py](../dreamer.py#L497), [dreamer.py](../dreamer.py#L512)).

All current modules are collected into one parameter mapping and optimized with one LaProp optimizer ([dreamer.py](../dreamer.py#L69), [dreamer.py](../dreamer.py#L131), [dreamer.py](../dreamer.py#L144)). Frozen target modules are deep copies whose parameter data are rebound to the current parameters and marked `requires_grad=False`; this preserves gradient flow with respect to inputs when not under `no_grad`, while preventing target-parameter gradients ([dreamer.py](../dreamer.py#L180)). In the imagination path, the additional `no_grad`/detach boundaries remove input gradients as well.

## Entrypoints and configuration map

- Training executable: Hydra-decorated `train.main` ([train.py](../train.py#L21)); there is no separate offline trainer.
- Evaluation entrypoint: `OnlineTrainer.eval`, called periodically from `begin` ([trainer.py](../trainer.py#L27), [trainer.py](../trainer.py#L125)).
- Online inference entrypoint: `Dreamer.act` ([dreamer.py](../dreamer.py#L245)).
- Model family switch and common losses: [configs/model/_base_.yaml](../configs/model/_base_.yaml#L1).
- Model capacity presets: `configs/model/size{12,25,50,100,200,400}M.yaml`.
- Environment suites: `configs/env/*.yaml`; runner examples: `runs/*.sh`.
- Global batch/buffer/trainer wiring: [configs/configs.yaml](../configs/configs.yaml#L1).

Potential concerns discovered during static inspection are intentionally separated into [RISKS.md](RISKS.md).
