# 冻结 Dreamer feature 的物理状态解码实验

本工具用于验证 feature 是否能恢复可继续仿真的物理状态。它不修改 Dreamer 训练代码，
不把物理标签送入 encoder，也不把恢复后的分支采样写回训练 replay。

首轮范围：`VE_wasser_H15`，proprio `finger_spin/hopper_hop/quadruped_run/walker_walk`，
模型 seeds `200/300`，每个任务与 seed 独立训练一个 feature decoder（共 8 个）。
`observations` 输入用于可选对照，不与 feature decoder 混合训练。

2026-11-05 重跑组：`VE_wasser_symlog_f64_H15`，proprio
`finger_spin/hopper_hop/quadruped_run` × seeds `1/42/100`，共9个checkpoint、18个decoder。
每模型100个episode、100轮训练、相同数据/训练seed及60/20/20划分；结果单独保存，
不覆盖首轮。该组没有walker；目录名中的f64不用于推断世界模型参数精度，加载时以实际张量为准。

从 2026-11-04 起，新训练的重建模块统一使用 **float64**：decoder 参数、标准化统计、
标签、loss、参数梯度、Adam 动量以及解码输出均为 float64，恢复状态继续使用 float64。
冻结 Dreamer 的计算精度不改变；其 feature 只在进入 decoder 时转换为 float64，
这种转换不会补回 feature 原先精度之外的信息。不启用全局 `torch.set_default_dtype`。

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
  --root logdir/VE_wasser_H15 --output logdir/dyn_recon/physics_probe/manifest.json
```

检查保存配置中的任务、seed、proprio、5 heads、开启截断、wasser、H15、dreamer。
每个任务/seed 必须恰好找到一个 `latest.pt`；缺失或有歧义时报错，不替用户选模型。
可用 `--tasks`、`--seeds`、`--distance-space` 和 `--expected-count` 明确选择其他已确认组。
例如这次9个symlog checkpoint的清单：

```bash
.venv/bin/python -m physics_probe manifest \
  --root logdir/VE_wasser_symlog_f64_H15 \
  --tasks finger_spin hopper_hop quadruped_run --seeds 1 42 100 \
  --distance-space symlog --expected-count 9 \
  --output logdir/dyn_recon/symlog_manifest.json
```

清单保存 checkpoint/config SHA256，后续会校验。该命令只检查文件和配置，
可用下面的命令另外验证严格加载：

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe checkpoint-check \
  --manifest logdir/dyn_recon/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --device cuda:0 --output logdir/dyn_recon/physics_probe/finger_spin_s200_load.json
```

其余 entry 为 `finger_spin_s300`、`hopper_hop_s200`、`hopper_hop_s300`、
`quadruped_run_s200`、`quadruped_run_s300`、`walker_walk_s200`、`walker_walk_s300`。

### 2. 真实快照恢复验证

```bash
MUJOCO_GL=egl .venv/bin/python -m physics_probe restore-check \
  --manifest logdir/dyn_recon/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --warmup 12 --steps 5 --atol 1e-9 \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_restore.json
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

### 2a. 批量验证真实核心状态恢复

`physics_probe.restore_experiment` 不加载世界模型或 decoder，只读取清单中的环境配置。
先保存**未经过任何恢复的原始参考轨迹**，再在相同动作上比较三个恢复分支：

- `full`：完整快照恢复，作为恢复接口的对照。
- `core64`：真实 float64 `qpos/qvel/act`，其余字段使用上面的简化规则。
- `core32`：核心状态先量化到 float32 再恢复，用于隔离数值精度影响，不代表 decoder 的误差。

在共享服务器上，先明确一个可用 CPU 核 `PROBE_CPU`，使用单核、低优先级运行。
下面是本次确认的诊断规模：四任务、5 个环境 seed、6 个起点、2 种动作保持时长，
共 240 个起点，每种恢复方式最多续跑 50 个策略步。各任务实际的物理时间由其控制周期决定。

```bash
taskset -c "${PROBE_CPU:?请先指定一个可用CPU核}" nice -n 19 env \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m physics_probe.restore_experiment \
  --manifest logdir/dyn_recon/physics_probe_dev/manifest.json \
  --output logdir/dyn_recon/physics_probe_stage2_repeat \
  --seeds 41000 41001 41002 41003 41004 \
  --points 0 10 50 150 300 450 --holds 1 10 --horizon 50 --full-atol 1e-9
```

保存完整参考数据 `reference_*.pt`、逐起点/逐深度的 `metrics.jsonl`、每任务的
`summary.json` 和总 `status.json`。到达真实 episode 终点即停止，不重置后继续比较。
`full_atol` 仅检验完整恢复对照；`core_accepted` 仍为空，不擅自设定科学验收标准。
恢复瞬间的触觉、IMU 等缓存观测与后续动力学应分别判读。同一轨迹上的多个起点相关，
报告中的 p95 是描述性统计，不是独立样本置信区间。随机动作覆盖不能代替训练策略覆盖。

### 3. 从冻结模型采集配对数据

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe collect \
  --manifest logdir/dyn_recon/physics_probe/manifest.json --entry finger_spin_s200 \
  --seed 12345 --episodes 3 --max-steps 12 --device cuda:0 \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_data
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
  --dataset logdir/dyn_recon/physics_probe/finger_spin_s200_data \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_decoder \
  --source features --seed 17 --epochs 2 --hidden 32 --batch-size 8 --device cuda:0
```

正式训练可调整网络；默认两层 hidden=256 的 SiLU MLP，Adam lr=1e-3。
以上小网络和两轮训练仅用于接口验证。使用 `--source observations` 和独立输出目录
可训练观测对照；使用同一数据集和同一训练 seed 会得到相同的 episode 划分。

按 episode 随机划分 train/validation/test；通常为 60/20/20，小数据至少各有一个 episode。
只用训练数据拟合标准化统计，按 validation loss 保存最好模型，测试集不参与选择。
`decoder.pt` 保存模型、标准化参数、分割、预算及数据清单 hash。
新 checkpoint 同时记录 `dtype=float64`，训练报告记录 decoder/label dtype；加载时先按
保存精度构造模型，避免 `load_state_dict` 隐式降精度。历史 float32 checkpoint 仍以
float32 加载和评估，不把它冒充为 float64 训练结果。精度声明与实际张量不符时拒绝加载。
新训练要求数据分片的原始 `cores` 为 float64，拒绝使用已量化为 float32 的标签。
此次切换只改变计算精度，保留既有 loss 权重、标准化下限和近常量分量阈值；并不保证
所有微小变化都能被学会，也不保证 float64 的解码准确度必然优于 float32。
训练目前把选中 episode 的 feature/label 拼接到内存，不适合直接处理无限大的数据集。

目标处理：

- 学习所有绝对 `qpos/qvel/act`；**没有自动消去全局位置或航向**。
- 无限制 hinge 用 sin/cos 编码，解码为主值角；不恢复整圈计数。
- 四元数采用符号不变的方向损失，恢复时归一化；退化四元数会被拒绝。
- 其他量用训练集统计标准化；恒定分量使用训练均值。
- 位置、速度、activation 分组平均，四元数方向另计，不让维度大的组自动主导。

这是一版可解释的基线。某些任务的绝对位置不能由 proprio 唯一确定，可能限制该目标的
可预测性；是否引入相对坐标需另行决定。不能以该基线失败直接判定 idea 不可行。

### 4a. 第三步的独立预测评估

`prediction-check` 只评估已保存的 test episodes，不启动任何物理恢复或续跑：

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe prediction-check \
  --dataset logdir/dyn_recon/physics_probe/finger_spin_s200_data \
  --checkpoint logdir/dyn_recon/physics_probe/finger_spin_s200_decoder/decoder.pt \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_prediction.json --device cuda:0
```

报告测试 loss、最佳验证轮次、原始逐坐标误差、周期校正后的非四元数 qpos 误差、
四元数方向误差、qvel/act 误差，以及逐 episode 的统计。四元数单独计算方向误差，
不把它们以零误差占位混进非四元数 qpos 的平均值。原始坐标误差仍包含角度绕圈和
四元数符号等表示差异，不能独自用来判断物理恢复质量。

### 4b. 已确认的第三步完整队列

2026-11-04 确认：8 个模型各采集100个episode，每个最多500个策略步；各训练100轮，
同时训练 feature 与 observation 对照，共16个独立decoder。网络和超参数沿用默认：
两层256单元、batch=256、Adam lr=1e-3，float64。数据seeds为50000–50099，
decoder训练及划分seed为17；每模型60/20/20个episode，两个输入对照共享完全相同的划分。

队列顺序执行，不自动进入实验④⑤。每个阶段使用独立子进程，结束后释放内存。
CPU必须绑定到单核并降低优先级；GPU仍限定6/7，以下限制PyTorch显存分配为2GiB
（不包括CUDA上下文本身的占用）。

```bash
taskset -c "${PROBE_CPU:?请先指定一个可用CPU核}" nice -n 19 env \
  CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m physics_probe.stage3 \
  --manifest logdir/dyn_recon/physics_probe_dev/manifest.json \
  --output logdir/dyn_recon/physics_probe_stage3_20261104 \
  --episodes 100 --epochs 100 --max-steps 500 --observations \
  --device cuda:0 --gpu-memory-gib 2 --data-seed 50000 --train-seed 17
```

输出根目录包含 `protocol.json`、固定模型清单、原子更新的 `status.json` 和逐阶段日志。
各模型目录保存 `data/`、`features/`、`observations/` 和两个 `test_*.json`；
每轮训练进度写入 `history.jsonl`。全部完成后自动生成 `summary.csv`、`summary.json`
和 `REPORT.md`。异常会停止队列并记录失败，不自动反复重试。

`--resume --output ...` 可跳过已完成且主要结果文件hash未变化的阶段；未完成输出和日志
不会被覆盖或自动重训，需先检查。它不是从一次decoder训练中间恢复优化器的接口。

运行期间可为尚未启动的模型增加辅助队列：

```bash
taskset -c "${PROBE_CPU:?请指定独立CPU核}" nice -n 19 env \
  CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m physics_probe.parallel_entry \
  --output logdir/dyn_recon/physics_probe_stage3_20261104 --entry walker_walk_s300
```

原队列与辅助队列通过 `.locks/` 下的逐阶段文件锁互斥；持锁后先校验已完成产物的模型身份、
数据身份和预算，匹配时复用，不重复执行。未完成目录不会被覆盖。
辅助队列使用独立 `parallel/<entry>/status.json` 和日志，原队列仍是根 `status.json`
与最终报告的唯一写入者。根状态可能尚未推进到已由辅助队列完成的模型，应结合
辅助状态或实际产物查看全局进度，不能简单累加两份状态中的阶段数。
本轮经用户授权最多4个GPU6 worker，每个2GiB PyTorch显存、一个独立CPU核、nice=19；
实际启动安排记录于输出目录的 `parallel_execution.json`。GPU利用率高时并行不保证线性加速。

对新9模型清单，主队列使用 `--expected-models 9`，输出路径改为
`logdir/dyn_recon/physics_probe_stage3_symlog_f64_20261105`。
辅助队列可用 `--entries model_id_1 model_id_2` 在同一worker上顺序处理多个模型，
不随模型数增加并发。此次仍最多4个worker，GPU6/7各最多2个；每worker独立CPU核、
nice=19、2GiB PyTorch显存上限。其余训练超参数与首轮相同。

### 5. 测试 posterior 解码与真实续跑

针对 `VE_wasser_symlog_f64_H15` 的9模型，使用专用入口 `physics_probe.stage4`。
它只读取该组第三步的已保存 decoder 和 test episodes；不训练、不加载 Dreamer 到GPU，
也不进行想象 feature 测试。重建路径保持 float64，原始观测仍为采集时的 float32。

2026-11-05 首轮运行预算：每个模型20个测试episode，各取
0/100/200/300/400/450步，feature 与 observation 对照各续跑最多50个策略步。
9模型合计1080个起点、2160条解码恢复轨迹，另有每起点的完整快照和真实核心状态对照。

```bash
env CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 .venv/bin/python -u -m physics_probe.stage4 \
  --source-root logdir/dyn_recon/physics_probe_stage3_symlog_f64_20261105 \
  --output logdir/dyn_recon/physics_probe_stage4_symlog_f64_20261105 \
  --points 0 100 200 300 400 450 --horizon 50 --sources features observations \
  --cpus "${PROBE_CPU:?请先检查CPU负载并指定一个可用核}"
```

入口最多允许两个独立CPU worker，默认不隐式选择CPU；负载高时可使用上面的单核配置。
本次按用户允许的并行方式运行两个单核 worker（CPU 69、89），nice=19、每worker单线程。
GPU6允许使用，但本次每个episode只推理6个起点，实际执行选择CPU，无GPU分配。
它在输出目录创建源码快照，再从快照运行各模型；不覆盖已有输出，不自动重试失败任务。
所有分支执行同一段记录动作。完整快照参考先核对采集时的后续状态、观测、奖励和终止标志，
核对失败的起点会明确列出，队列不会把它算作已通过的模型。

输出 `status.json`、逐模型日志、各模型 `report.json` / `metrics.jsonl`，
全部完成后生成 `REPORT.md`、`summary.csv` 和 `summary.json`。
分别报告第0/1/5/15/50步的位置、速度、四元数方向、观测、奖励和累计奖励误差，
包括均值/p95/最大值、异常数与有效起点覆盖。第0步额外比较解码状态与真实核心恢复的观测，
辅助分离传感器缓存效应；此差值本身也不是科学验收阈值。
后续深度只包含有有效指标的分支，必须结合异常和缺失数判读，不能把完成运行视为重建成功。

本轮第一步接续图片仅保留第四步输出目录 `first_step_images_shared_view/` 下的3张任务汇总图，
共享相机取景范围覆盖三个分支；原参考相机图片和单独帧已按用户要求删除。
每模型使用测试 episode 60 / 环境 seed 50060，在策略步100恢复，执行同一条记录动作后只拍步101。
左/中/右为 feature、observation、真实核心状态恢复；每张图包含该任务3个seed。
`README.md` / `index.html` 仅引用这3张图；渲染入口也只保存这3张汇总图。
各图的观测和速度误差与第四步存档核对一致；取景变化不修改物理坐标。
`manifest.json` 保存动作、相机、数据选择、指标与源码hash，`render_script.py` 是对应源码副本。
图像由 MuJoCo 在 GPU6 上实际渲染；单帧不能显示完整速度或触觉/IMU信息。

下面的通用单decoder入口仍可用于小规模诊断：

```bash
CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl .venv/bin/python -m physics_probe evaluate \
  --dataset logdir/dyn_recon/physics_probe/finger_spin_s200_data \
  --checkpoint logdir/dyn_recon/physics_probe/finger_spin_s200_decoder/decoder.pt \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_posterior.json \
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
  --dataset logdir/dyn_recon/physics_probe/finger_spin_s200_data \
  --checkpoint logdir/dyn_recon/physics_probe/finger_spin_s200_decoder/decoder.pt \
  --output logdir/dyn_recon/physics_probe/finger_spin_s200_imagination.json \
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

## 2026-11-05：VE_wasser_H15 的12个vision模型

用户确认使用 `VE_wasser_H15` 下的 vision checkpoint：四任务
finger_spin/hopper_hop/quadruped_run/walker_walk × seeds 0/100/200。
每模型采集100个episode（最多500策略步），按episode 60/20/20划分，训练100轮。
两个输入来源是冻结 vision Dreamer 的 posterior feature，以及同一时刻的单帧RGB图像。
图像对照不输入proprio；保存的proprio仅用于物理诊断。两种decoder共24个，均使用float64。
原Dreamer维持float32，64×64 RGB按原生 `preprocess` 除以255后进入encoder，绝不展平图像送CNN。

image→state对照为四层stride=2、kernel=3、padding=1的卷积，通道16/32/64/128，
每层SiLU，再接两个256单元的SiLU全连接层和状态输出层。batch=256、Adam lr=1e-3；
像素输入固定归一化为 `pixel/255-0.5`，目标归一化仅根据训练集计算，几何loss沿用feature decoder。
像素在数据及CPU批次中保持uint8，训练/推理时只将当前批次转float64。
单帧CNN没有RSSM历史，因此对照的时间信息量不同，尤其不能假设单帧能唯一确定速度。

步骤④沿用每测试episode的0/100/200/300/400/450步，共1440个起点、2880条解码恢复轨迹，
另有1440条真实核心状态恢复及1440条完整快照参考，每条最多续跑50策略步。
`observation_rmse`仍表示proprio诊断；另报`image_rmse_01`（RGB/255）和`image_max_abs_01`。
完整快照的核心状态、proprio、奖励、终止匹配是参考校验；像素复现误差单独报告。
预检12模型全部严格加载且采集器动作与原生actor逐值一致，状态恢复误差为0；
部分hopper/quadruped渲染像素相差1灰度级，记录在预检中，不能表述成图像逐像素完全一致。

资源：最初为GPU6上2个worker；用户随后授权扩容，现新增4个worker，总上限6个。
每worker独立单核、PyTorch单线程、nice=19，PyTorch显存上限2GiB。
EGL显式绑定 `MUJOCO_EGL_DEVICE_ID=6`，CUDA也限定6。队列使用源码快照，避免后续工作区编辑影响待运行模型。
扩容记录见总队列目录的 `parallel_expansion.json`，新增worker状态和逐阶段日志位于
`additional/extra1` 至 `additional/extra4`。它们调用原源码快照的stage锁，
独立保存状态，不占用原辅助队列的 `parallel/<entry>` 目录，防止后续排队任务目录冲突。
新增CPU为119/109/117/115，原CPU88/89继续运行；新增任务的结果仍写入同一个stage3目录。
查看总进度时应结合主队列、原辅助队列、新增worker状态及各模型实际产物，避免漏算或重复计数。

路径：

- 预检：`logdir/dyn_recon/physics_probe_vision_20261105_preflight_v2/`。
- 总队列：`logdir/dyn_recon/physics_probe_vision_20261105/`，查看`status.json`及各阶段日志。
- 采集/训练：`logdir/dyn_recon/physics_probe_stage3_vision_20261105/`；主队列与辅助队列通过锁避免重复训练。
- 续跑/汇总：`logdir/dyn_recon/physics_probe_stage4_vision_20261105/`。
- 图片：续跑目录下`first_step_images/`，仅4张任务汇总PNG，分别包含三个seed；
  从同一测试episode第100步恢复，仅保存执行一个相同动作后的第101步。

`physics_probe.vision_campaign` 自动串联采集/训练、续跑、渲染。任一阶段失败会记录错误并停止后续阶段，
不会覆盖已有数据或自动重试；没有启用想象feature测试，也没有将恢复采样接入Dreamer训练。
正式启动前的 `vision_smoke` 只运行3个短episode和2轮训练，用于集成验证，不作为科学结果。
