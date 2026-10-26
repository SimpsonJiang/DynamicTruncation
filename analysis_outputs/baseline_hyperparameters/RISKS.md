# Baseline 审查发现及证据边界

2026-10-26；对象 `2ff4ce7892d40f6d90aff2661d699a819a274d17`。以下区分已确认的源码行为与尚未确认的性能影响。没有修改实验实现。

## 1. 当前启动脚本里的想象长度变量无效

- 位置：`run.sh:12`、`:78`、`:118`。
- 已确认：声明了 `imag_hori=15`，但 python 命令没有 `model.imag_horizon=...`，生成 worker 的变量列表也没有该变量。
- 后果：改这个变量不会改变训练长度；当前长度由 `configs/model/_base_.yaml:3` 的 15 决定，所以现有 baseline 的长度值本身正确。
- 建议：未来修复时同时处理 Hydra override 和 worker 变量传递；本次不修改用户工作区。

## 2. `model.rep_loss=dreamer` 单独使用不足以选择 baseline

- 位置：`configs/model/_base_.yaml:18`、`:23`，`run.sh:86`。
- 已确认：全局默认有 5 heads 且启用动态截断；当前脚本与全部已保存 baseline 都显式设置为 1/false，因此这 9 个实验没有受到影响。
- 风险：用 README 早期简化命令或手动只切换 rep_loss，会运行修改过的 Dreamer。

## 3. KL 未应用所配置的 unimix

- 位置：`rssm.py:219`、`:222`；`distributions.py:16`、`:266`。
- 已确认：状态分布和 entropy 调用 OneHotDist，应用 1% 均匀混合；KL 函数直接用原始 logits，未应用该混合。官方 DreamerV3 RSSM 通过 `_dist` 计算 KL，传入 unimix。
- 数值例：小模型中 q logits `[12,0,-12,-24]`、p 为其相反数、free=0 时，本地 raw-logits KL 为 35.99985，而混合后采样分布之间的 KL 为 5.92405。例子仅证明两个计算不等价，不是实际训练 batch 的统计。
- 来源：本地初始 `10f4afc` 和当前 R2-Dreamer 上游 RSSM 都保留此行为；不是 ensemble 修改引入的回归。
- 未确认：是否为作者有意简化、对本次任务分数和稳定性的影响。对齐官方时应优先核对，不能把它直接归因为低分原因。

## 4. `rssm.initial=learned` 不对应可学习初始状态

- 位置：`configs/model/_base_.yaml:74`；`rssm.py:87`、`:133`、`:158`。
- 已确认：`_initial` 只赋值，没有参与初始化分支；initial 返回零，episode reset 也先清零再执行确定性转移。
- 来源：从原始本地及 R2-Dreamer 上游继承。
- 影响边界：配置名称会误导阅读；不能据此认定零初始化错误，也不能声称当前官方 DreamerV3 必须使用 learned initial。

## 5. 部分配置名不是有效切换开关

- `norm`：`networks.MLP` 和 RSSM 固定构建 RMSNorm；把配置改成 false 不会统一关闭归一化。当前 true 与实际 RMSNorm 行为一致。
- `_lambda_return` 的 `value` 只参与形状断言，递推使用 `boot`。这是继承行为，本身不证明 return 错误；本次没有改动。

## 6. 严格官方对齐仍存在协议与实现差异

- 当前官方 DMC proprio preset 为 size1m/repeat1/train_ratio1024/steps1.1e6；本地为 size12M/repeat2/train_ratio512/steps5.1e5。这是选用的实验协议不同，不是本地脚本覆盖失败。
- 本地 replay 使用 TorchRL SliceSampler，未实现显式 online queue；当前上游 R2-Dreamer 已提供 online 配置，不能跨版本混为一谈。
- 本地 continuation target 为 `1-is_terminal`，在想象 weight/return 中乘 gamma；官方当前默认 contdisc=true，在 target 中也表达 gamma。是否等价需要固定版本并成对核对，不能只根据一行表达式判错。
- 本地 one-hot rsample 使用 hard Gumbel-softmax；与 categorical sample 后 straight-through 的梯度估计不应混称。未逐项核对官方依赖库实现。
- 本地 bounded_normal 仅限制均值，采样本身可越界；NormalizeActions 不 clip，RSSM 输入处则限制动作幅度。DMC/MuJoCo 下游动作处理没有在本轮运行验证，不判定为已确认的训练错误。
- 当前本地使用 FP16+GradScaler，上游当前已改为 BF16；小模型 CPU 测试不覆盖 CUDA 编译和 AMP。日志标量有限也无法排除未记录的瞬时非有限梯度。
- Eval 使用独立环境，但与训练调用同一 seed+env index 构建器；actor 取 mode，latent 仍采样。若要求官方 eval 协议或独立 seed 集合，需要另行固定。

## 7. 现有证据不能证明分数差异的原因

- baseline 与 VE 同时改变 heads 和截断；比较支持联合效果，不支持独立归因。
- seed 相同不保证两种架构的初始化和后续随机数流逐步一致，head 数量改变会消耗不同数量的随机数。
- 没有运行完整训练、checkpoint eval、DMC 环境动作核对，也没有固定官方历史实验 commit。未修改任何源代码来追分。
