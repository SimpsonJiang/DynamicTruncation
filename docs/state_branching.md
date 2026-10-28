# 从 imagined latent 恢复 Finger 仿真并采集数据

首版支持 `dmc_finger_spin` 的 proprio 观察及 `model.rep_loss=dreamer`。
StateDecoder 预测物理状态，独立分支环境继续真实仿真；分支 replay 按序列比例参与正常 Dreamer 更新。
功能默认关闭，已有的 value ensemble 和动态截断仍由各自配置控制。

## 运行配置

```bash
# 正常采集基线
MUJOCO_GL=egl python train.py +experiment=finger_baseline logdir=logdir/finger_baseline

# 仅训练 detached StateDecoder
MUJOCO_GL=egl python train.py +experiment=finger_state_decoder logdir=logdir/finger_state_decoder

# 从正常 replay 的真实物理快照恢复后采集
MUJOCO_GL=egl python train.py +experiment=finger_replay_branch logdir=logdir/finger_replay_branch

# 从 imagined feature 重建的状态恢复后采集
MUJOCO_GL=egl python train.py +experiment=finger_imagined_branch logdir=logdir/finger_imagined_branch
```

也可显式设置 `model.state_decoder.enabled=true branch_collection.mode=imagined_state`，
同时指定 `env=dmc_proprio env.task=dmc_finger_spin model.rep_loss=dreamer`。
`imagined_state` 要求 decoder 和 `model.dynamic_trunc.enabled` 均开启，否则在创建环境之前报错。
`replay_state` 不要求启用 decoder。`mode=off` 时可单独训练 decoder。

## 数据与梯度

`state_decoder.py` 将通用逻辑放在抽象基类 `StateDecoder` 中，Finger 的状态处理由
`FingerStateDecoder(StateDecoder)` 实现。`make_state_decoder()` 根据环境 state spec 的
`task` 字段选择子类；不会通过状态维度猜测任务。配置 `shape: null` 时由子类的
`encoded_size` 决定输出维度，显式填写 shape 则必须与子类一致。

`StateDecoder(feat.detach())` 复用 `networks.MLPHead`，使用 `identity` 输出。
默认两层隐藏层，每层为 Linear → RMSNorm → SiLU，再接 7 维线性输出；
网络配置沿用其他 head 的格式，损失权重为 `model.loss_scales.state_recon`（默认 1.0）。
当前共用 MLP 始终包含 RMSNorm，与其他 head 一致；`norm` 字段不会关闭它。
两个限位关节按 joint range 标准化，
转子角使用 sin/cos，三个关节速度使用 symlog，共 7 个输出；反变换得到 6 维 qpos/qvel。
该任务没有 actuator activation。重建 MSE 只更新 StateDecoder，共用现有 optimizer、AMP 和 AGC。
单独反传重建损失不会产生 encoder/RSSM 梯度；在固定输入和随机结果下，增加该损失不改变它们的梯度。
共享 AMP scaler 仍可能因任意损失的非有限梯度而跳过整次 optimizer step，这不属于梯度回传。

监督字段 `_physics_state` 不属于 observation space，并从 encoder/原 observation decoder 显式排除。
完整 `_integration_state` 仅在真实快照分支需要时采集，使用 float64，绕过普通观察的 float32 转换。
元数据 `_source/_branch_id/_origin_step/_imag_depth` 分别标识来源、分支、真实起点采集步数和想象深度。
`_source=0` 为正常数据，`1` 为分支数据。

## 起点选择、恢复与采集

想象模式直接使用训练 `Dreamer.update()` 中产生的 imagination：选择实际截断 mask 第一个 False 的位置 k，
把该位置的 `imag_feat[:, k]` 交给 detached StateDecoder，随后恢复环境并采集。
不是前一步 k−1，也不是后一步 k+1；不再另外从 replay 重新想象固定步数。
截断沿用 `model.VE_metric`（wasser 或 ucb）、该次训练使用的 EMA 阈值及 `dynamic_trunc.min_horizon`。
现有判定是 score **大于**阈值才截断，等于阈值仍保留。
若在强制保留前缀内提前越界，实际分支位置为前缀结束处；该处 score 可能已回落。
达到最大 `imag_horizon` 而没有实际截断的轨迹不产生候选；由非有限 score 引起的截断也不用于恢复。

仅接收来源为正常 replay、且 posterior 起点非 is_last/非 is_terminal 的轨迹，避免分支数据递归分叉。
每次更新后，从有效截断位置均匀抽取至多 `max_attempts` 个，复制 feature 和原始步数、实际深度、score、阈值。
候选池只保留最新一次更新的结果（包括空结果），到达原定采集频率及 warmup 后消费；无候选则跳过本轮。
这不会为每条截断轨迹立即创建环境，仍受每轮分支数、尝试次数及总环境步数预算限制。
真实快照对照模式 `replay_state` 仍从正常 replay 序列的非结束位置随机选择真实快照，深度记为 0。
候选抽样和分支交互使用独立、持续的 CPU/CUDA 随机流，不推进主训练的随机流，也不额外更新截断阈值 EMA。

独立 worker 正常 reset 后写入预测 qpos/qvel，控制输入和 warmstart 归零，再 forward 刷新派生量。
真实快照模式写入 integration state，并在 forward 后恢复 integration 输入以保留 warmstart；
两种模式都以新的 episode 计数器开始。当前实现仅支持固定模型的 Finger Spin，不是通用任务快照系统。

拒绝非有限状态、关节越界、退化角度向量、超速、严重穿透或 MuJoCo warning。
第一步真实交互同时承担稳定性检查，通过后纳入片段；后续每一步继续检查。
物理异常导致整个暂存片段丢弃并重建 worker。阈值是工程初值，不代表重建状态准确性保证。

策略从恢复后的实际 observation 和全新 memory 开始 posterior 推断；不沿用 imagined latent。
初始记录 `is_first=true, reward=0`，每行保存依据当前观察选出的下一动作。
长度或预算截断只设置 `is_last=true`，不伪造 terminal，结束行动作清零。

## 参数与预算

| 参数（branch_collection 下） | 默认值 | 语义 |
| --- | --- | --- |
| `warmup_steps` | 20000 | 正常交互达到此环境步数且已发生梯度更新后，才允许分支采集 |
| `every_steps` | 2000 | 两轮之间新增的正常环境步数，分支步不会触发下一轮 |
| `branches_per_round` | 4 | 每轮最多成功采集的分支数 |
| `max_attempts` | 16 | 最新训练更新保留的候选数及每轮尝试次数上限，包含解码/恢复被拒绝 |
| `horizon` | 64 | 每条分支最多执行的策略决策次数 |
| `replay_capacity` | 100000 | 分支 buffer 记录容量，包含 reset 行 |
| `replay_fraction` | 0.25 | 每个训练 batch 的分支序列比例，不是 loss 权重 |
| `max_velocity` | 200 | rad/s，所有关节速度的绝对值上限 |
| `max_penetration` | 0.001 | 米，允许的最大接触穿透 |
| `min_angle_norm` | 1e-6 | sin/cos 预测向量的最小范数 |

`branch_collection.imag_depth` 已移除；`_imag_depth` 元数据及 `branch/imag_depth` 日志现在记录实际截断深度。
`branch/cutoff_score` 和 `branch/cutoff_threshold` 为成功分支对应截断位置的平均分数与阈值；
`branch/rejected/no_truncation` 表示本轮候选池为空或已耗尽。若 `min_horizon >= imag_horizon`，不会产生想象分支。

分支 buffer 的存储及训练设备完全继承 `buffer.storage_device`、`buffer.device`。
当前默认配置及启动脚本将 replay 存在 GPU；若改成 `buffer.storage_device=cpu`，两个 buffer 都跟随。
分支 buffer 是额外容量，不能由混合比例推算显存占用；即使 proprio 模式，环境记录仍含图像。

默认 batch=16 时抽取 12 条正常序列和 4 条分支序列，分支不足时回退到完整正常 batch。
序列保持 T+1 上下文、动作移位、latent 缓存回写；索引按 buffer 分流，跨片段靠 is_first/is_last 隔离。

总训练预算为正常步加分支步，均计入 action repeat。一次完整分支为 64×2=128 环境步。
失败后被丢弃的仿真步也计费；restore/reset 和 latent imagination 不计费。
发生仿真异常的控制步按已尝试的控制调用计费，不承诺该调用完整推进了内部物理积分。
临近预算上限时分支缩短或跳过；正常并行环境保留原有的一次向量 step 粒度。
训练更新比例及评估使用总步数。评估环境始终从正常 reset 开始，branch 指标与普通 episode 分开。

## 验证

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

新增测试覆盖开关、梯度隔离、元数据排除、快照精度与恢复、周期角、混合采样和回写、失败计费及预算截断，
以及实际 VE 截断位置、强制前缀、无截断跳过、候选池隔离与训练输出对齐。
闭环单测使用有效状态的可控 decoder 和零控制来验证数据管线，不证明未经训练的 decoder 能产生有效起点。
训练效果应通过正常采集 / 真实状态分支 / 想象状态分支的同总交互预算、多 seed 对照验证。

本次动作对齐测试还复现并修复了 CPU replay 原有的重叠原地复制问题：
动作移位前现在先 clone，避免首个动作被重复传播到整段序列。
已有 checkpoint 可继续用于原有关闭功能的模型；启用 decoder 的 checkpoint 需传入环境的 `get_state_spec()` 构造模型。
训练入口仍未提供完整断点续训，replay 和分支 worker 的运行状态不会随 agent checkpoint 恢复。

StateDecoder 从早期手写 Sequential 切换为 MLPHead 后，增加了 RMSNorm 并使用项目统一初始化。
早期 decoder 权重的结构和键名不兼容当前 decoder；旧参数 `model.state_decoder.loss_scale`
需改为 `model.loss_scales.state_recon`。

## 扩展任务的 StateDecoder

新增任务需继承 `StateDecoder`，定义 `encoded_size`，实现 `setup_state_spec()`、
`encode()` 和 `decode()`，再将任务名及子类加入 `STATE_DECODERS`。
任务相关的归一化常数应注册为 buffer；可覆盖 `error_metrics()` 输出分组指标，
覆盖 `prediction_rejection_reason()` 添加编码表示的合法性检查（先调用基类的有限值检查）。
通用 `forward()` 和 `loss()` 负责输入/标签 detach、MLPHead 和编码空间 MSE。

基类扩展不意味着其他环境已支持恢复：当前端到端功能仍限定 Finger Spin。
接入其他任务时，还需补齐环境的 state spec、恢复及物理约束实现，调整
`validate_branch_config()` 的任务限制，并验证完整采集流程。
