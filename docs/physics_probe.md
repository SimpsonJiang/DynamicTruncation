# 冻结 Dreamer feature 的物理状态解码实验

本工具用于验证 feature 是否能恢复可继续仿真的物理状态。它不修改 Dreamer 训练代码，
不把物理标签送入 encoder，也不把恢复后的分支采样写回训练 replay。

已约定的范围：`VE_wasser_H15`，proprio `finger_spin/hopper_hop/quadruped_run/walker_walk`，
模型 seeds `200/300`，每个任务与 seed 独立训练一个 feature decoder（共 8 个）。
`observations` 输入用于可选对照，不与 feature decoder 混合训练。

## 环境与设备

在仓库根目录使用已安装 DMC extra 的 Python 3.11：

```bash
.venv/bin/python -m physics_probe --help
```

只允许物理 GPU **6 或 7**。CUDA 命令必须显式限制可见设备；编号会重新映射：

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe ... --device cuda:0
```

也支持 `CUDA_VISIBLE_DEVICES=7`，以及显式的 `6,7`/`7,6` 和对应逻辑编号。
未设置掩码、包含其他 GPU、逻辑设备越界都会拒绝。小规模接口测试支持 `--device cpu`。
工具不自动选 GPU、不启动 tmux、不运行整批实验。环境为纯 proprio，沿用 DMC step 和
仓库动作归一化/action repeat/time limit，但省略图像渲染。

## 按阶段运行

所有输出必须是新路径，避免覆盖实验。下面的 `3 episodes × 12 steps、2 epochs`
是**冒烟测试预算**，只能验证代码是否连通，不代表正式数据规模或科学验收。
正式实验必须自行指定 episode 数、每段最大长度、训练轮数和评估起点数量。

### 1. 建立固定模型清单

```bash
.venv/bin/python -m physics_probe manifest \
  --root logdir/VE_wasser_H15 --output logdir/physics_probe/manifest.json
```

检查保存配置中的任务、seed、proprio、5 heads、开启截断、wasser、H15、dreamer。
每个任务/seed 必须恰好找到一个 `latest.pt`；缺失或有歧义时报错，不替用户选模型。
清单保存 checkpoint/config SHA256，后续会校验。该命令只检查文件和配置，
可用下面的命令另外验证严格加载：

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe checkpoint-check \
  --manifest logdir/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --device cuda:0 --output logdir/physics_probe/finger_spin_s200_load.json
```

其余 entry 为 `finger_spin_s300`、`hopper_hop_s200`、`hopper_hop_s300`、
`quadruped_run_s200`、`quadruped_run_s300`、`walker_walk_s200`、`walker_walk_s300`。

### 2. 真实快照恢复验证

```bash
MUJOCO_GL=egl .venv/bin/python -m physics_probe restore-check \
  --manifest logdir/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --warmup 12 --steps 5 --atol 1e-9 \
  --output logdir/physics_probe/finger_spin_s200_restore.json
```

在真实环境经过随机动作 warmup 后保存状态，将其恢复到另一环境，执行相同动作。
报告 `full` 和 `core` 两组；`--atol` **只用于完整快照恢复的工程验证**。
完整恢复不通过时返回失败。核心状态恢复的误差只记录，`core_accepted: null`，
不默认判定精度足以支持研究。应覆盖不同采样种子和运动阶段后再确定科学容差。

完整恢复使用 `mjSTATE_INTEGRATION`、任务 RNG/标量上下文、内外 episode 计数和
缓存传感器。模型二进制 hash 不同则拒绝恢复。DMC legacy stepping 的加速度相关传感器
可能对应上一个子步，完整恢复保留其缓存；解码恢复重新计算传感器，不借用真实传感器。

核心恢复使用预测 `qpos/qvel/act`、已知时间/上一动作的 `ctrl`、reset-default 的其余
用户输入以及零 warmstart。如果外力、mocap、约束开关、用户数据或插件状态不同于
reset 默认值，工具拒绝该简化策略。完整快照不等于任意任务的完整环境；本实现只支持
约定的四个固定模型任务，不支持随机地形或任意 model 修改。

### 3. 从冻结模型采集配对数据

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe collect \
  --manifest logdir/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --episodes 3 --max-steps 12 --device cuda:0 \
  --output logdir/physics_probe/finger_spin_s200_data
```

数据 seed 与模型 seed 分开；各 episode 使用 `seed + episode_id`。
模型严格加载、关闭 compile、eval、所有参数冻结；使用随机 actor 和随机 posterior latent，
不使用旧 replay cached latent。每个 episode 从初始 RSSM 状态顺序运行。

一行数据严格对应：

```text
snapshot/core[t], observation[t], posterior_feature[t], stoch/deter[t]
action[t] -> 下一观测及 reward[t]、done[t]
```

snapshot 在当前观测时刻保存，action 在同一 feature 上采样。`max-steps` 可截短采集，
此时 `complete_episode=false`；它不是环境终止标签，也不能用于 episode return 比较。
一个 imagined step 等于一个策略动作步，包含配置中的 action repeat。

`dataset.json` 记录模型身份、数据 seed、预算、版本、schema 和每个 episode 的 hash。
`episode_*.pt` 分片保存 CPU tensors：float64 物理快照、feature、独立的 observation
向量、动作和真正的 episode ID。没有给环境的 observation space 增加监督字段。
当前实现保存每个时刻的完整快照，数据量可能较大；正式预算需考虑磁盘与主机内存。

### 4. 训练独立 decoder

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe fit \
  --dataset logdir/physics_probe/finger_spin_s200_data \
  --output logdir/physics_probe/finger_spin_s200_decoder \
  --source features --seed 17 --epochs 2 --hidden 32 --batch-size 8 --device cuda:0
```

正式训练可调整网络；默认两层 hidden=256 的 SiLU MLP，Adam lr=1e-3。
以上小网络和两轮训练仅用于接口验证。使用 `--source observations` 和独立输出目录
可训练观测对照；使用同一数据集和同一训练 seed 会得到相同的 episode 划分。

按 episode 随机划分 train/validation/test；通常为 60/20/20，小数据至少各有一个 episode。
只用训练数据拟合标准化统计，按 validation loss 保存最好模型，测试集不参与选择。
`decoder.pt` 保存模型、标准化参数、分割、预算及数据清单 hash。
训练目前把选中 episode 的 feature/label 拼接到内存，不适合直接处理无限大的数据集。

目标处理：

- 学习所有绝对 `qpos/qvel/act`；**没有自动消去全局位置或航向**。
- 无限制 hinge 用 sin/cos 编码，解码为主值角；不恢复整圈计数。
- 四元数采用符号不变的方向损失，恢复时归一化；退化四元数会被拒绝。
- 其他量用训练集统计标准化；恒定分量使用训练均值。
- 位置、速度、activation 分组平均，四元数方向另计，不让维度大的组自动主导。

这是一版可解释的基线。某些任务的绝对位置不能由 proprio 唯一确定，可能限制该目标的
可预测性；是否引入相对坐标需另行决定。不能以该基线失败直接判定 idea 不可行。

### 5. 测试 posterior 解码与真实续跑

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe evaluate \
  --dataset logdir/physics_probe/finger_spin_s200_data \
  --checkpoint logdir/physics_probe/finger_spin_s200_decoder/decoder.pt \
  --output logdir/physics_probe/finger_spin_s200_posterior.json \
  --device cuda:0 --stride 6 --horizon 3 --max-starts 2
```

只使用保存的 test episodes。每个起点分别恢复真实快照和预测核心状态，执行相同的已记录
动作，报告原始/周期校正的位置误差、速度、方向、activation、观测、奖励和 done 差异。
保存逐起点/逐深度数据及聚合、异常比例；异常不能算作成功样本。
`max-starts` 按固定 episode/时间顺序限制评估量，较小时可能只覆盖首个 test episode，
正式评估应设置足够大并检查实际覆盖范围。

### 6. 想象 feature 的恢复诊断

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe imagine-check \
  --dataset logdir/physics_probe/finger_spin_s200_data \
  --checkpoint logdir/physics_probe/finger_spin_s200_decoder/decoder.pt \
  --output logdir/physics_probe/finger_spin_s200_imagination.json \
  --device cuda:0 --stride 6 --horizon 15 --max-starts 2
```

从测试集 posterior latent 起步，用冻结 actor/RSSM 生成 H+1 个 feature，并让真实模拟器
执行相同想象动作。depth=0 是 posterior 起点。各深度同时检查解码误差、解码状态恢复后
的当前观测与下一步动力学；这不是将样本接入训练的第二阶段算法。

阈值固定使用 checkpoint 已保存的 EMA，不更新它；报告该阈值下的掩码边界，区分
真正提前截断和完整 H 步。这个策略不等价于在线训练中持续更新 EMA，报告会明确标注。
提前遇到真实 episode 结束则停止，不自动 reset 接着比较。

## 判读结果与限制

`accepted: null` 表示尚未指定科学验收阈值。命令成功仅说明完成了诊断。
低训练 loss、无异常、单个 seed 结果、几个短轨迹都不能证明想象状态恢复可靠。
各任务、各模型 seed 分别报告；真实状态恢复、posterior 解码、想象误差不可混为一谈。
imagined-feature 测试测到的是世界模型与 decoder 的组合误差，无法独自归因。
状态扰动敏感性扫描、坐标规范化及第二阶段 replay 接入尚不包含在首版。

CPU 回归检查：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p test_physics_probe.py -v
```

未安装 DMC extra 时，真实模拟器测试会明确 skip，不能视作已验证恢复。
