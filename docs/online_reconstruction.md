# 在线物理 decoder 与安全重建采样

两个开关默认关闭，原有 checkpoint 和不启用功能的路径保持兼容。首版覆盖
`finger_spin`、`hopper_hop`、`quadruped_run`、`walker_walk` 的 proprio/vision；
重建分支损失支持 `model.rep_loss=dreamer` 和 `r2dreamer`。

## 分阶段启用

`uv_run.sh` 和 `run.sh` 顶部通过一个 `dyn_rec` 参数选择模式，默认 `off`。
与其他超参数一样直接修改脚本顶部赋值：

```bash
dyn_rec=on # on / decoder_only / off
```

| 模式 | 在线 decoder | 截断重建采样 |
|---|---|---|
| `on` | 开启 | 开启 |
| `decoder_only` | 开启 | 关闭 |
| `off` | 关闭 | 关闭 |

`on` 要求 `enable_dny_trunc=true`；三个模式均不改变 `grad_to_wm=false` 的默认值。
修改后执行 `bash uv_run.sh`。`METHOD_TEMPLATE` 显式包含 `_dyn_rec{dyn_rec}`，
按所选模式生成实验目录名。模式传入每个 tmux worker，并映射为底层
`model.physics_decoder.enabled`、`reconstruction.enabled` 两个布尔配置。
临时文件使用 `/mnt/phx2/tmp`。
启动会按脚本当前 tasks、seeds 和 GPU_MAX_JOBS 批量运行，启动前核对这些设置。
开启重建仍需达到默认100,000 env steps并通过安全门槛，才会调度重建分支。

先只训练 decoder，在现有训练命令后添加：

```bash
model.physics_decoder.enabled=true
```

`grad_to_wm=false` 为默认：从当前 minibatch 的 posterior feature detach 后，
用独立 Adam 更新 decoder。需要辅助监督世界模型时显式添加
`model.physics_decoder.grad_to_wm=true`；此时联合 backward，两个优化器使用同一个
AMP GradScaler，分别 unscale、裁剪和 step。

完整接入使用：

```bash
model.physics_decoder.enabled=true reconstruction.enabled=true
```

开启后仍须先通过本次运行的普通交互标定/验证。不会额外创建环境，不会自动启动实验。

## 数据、精度与 checkpoint

`envs/online_physics.py` 在 Dtype wrapper 之外，以独立 RPC 返回同一时刻的
`qpos + qvel + act`，存入 replay 的 `physics/core`，保留原始 float64。
它不属于 observation space，encoder 和 actor 无法把标签作为观测输入。
`physics/context` 只用于候选恢复；图片仍是原来的 uint8 数据。

Decoder 为两层 256 单元 SiLU MLP，Adam lr=1e-3，loss scale=1，梯度范数上限100。
参数、输入提升后的计算、标签、损失、EMA统计和 Adam 一、二阶矩均为 float64；
世界模型继续使用原来的 AMP。EMA decay=.99，尺度下限1e-3，只从训练 minibatch 更新。
周期 hinge 使用 sin/cos，四元数使用符号不变的方向损失。推理不更新统计量。
初始化使用隔离 RNG；候选池使用独立 NumPy Generator。

`latest.pt` 包含 decoder 参数、EMA统计、schema 摘要及独立 optimizer 状态；
完整 schema 同时写入 checkpoint 的 `physics_schema` 和 `.hydra/config.yaml`。加载时拒绝 schema/float64 不匹配。
这仍然是原仓库的模型/优化器保存机制，**不是完整在线训练断点恢复**：
环境、replay、候选池和调度器没有跨进程恢复接口。

## 穿模门槛

同一批普通环境槽的前50,000 env steps 标定，随后50,000 env steps验证，
两段均计入原有预算。对每个几何对，每积分子步检查正穿透深度；
阈值=max(1.25×正穿透深度99.9%分位数, 特征尺寸×1e-3)。
这里的特征尺寸为两几何各自最大 size 分量的较小者；平面取移动几何的尺寸。
没有正接触样本的几何对使用下限。

检查现有接触及被碰撞过滤屏蔽的关键几何对，排除同刚体、焊接刚体、直接相邻肢体。
当前四任务使用原始几何体；遇到未支持的 mesh/hfield/SDF 关键对时禁止重建。
诊断使用独立 MjData，执行位置/碰撞计算与有符号距离查询，不推进主环境，
不改主环境 warmstart。保留 dm_control 的 step2 → step → step1 顺序及任务回调。

验证以策略步计数：任意子步越阈即一次误拒，误拒率必须≤0.1%。
失败时继续普通采样并记录原因，不放宽阈值。阈值冻结后不使用重建数据调整。
`reconstruction_guard.json` 保存几何对、版本/模型/积分设置签名、阈值与验证状态。

恢复前检查 float64、有限数值、四元数、上下文兼容性和穿透；
恢复失败直接普通 reset，不循环重试。续跑失败立即结束分支，整段丢弃。
这是异常穿透门槛，不保证所有视觉网格零交叠，也不把全部负接触距离当成错误。

## 分支与 replay

- `is_branch` 只标分支根；`is_recon` 标整个片段。
- 根 latent 复用 replay 的 `stoch/deter`，由 `is_branch` 标记并禁止 latent 回写覆盖。
  采样时仅为当前 batch 构造 `branch_stoch/branch_deter`，保持 RSSM 接口不变。
- RSSM 在窗口任意位置遇到根时直接读取 seed，不对根执行 obs_step；普通 is_first 仍清零。
- 首动作使用候选动作；真实下一观测到达后，当前策略 posterior 闭环采样。
- 根的虚构入边、占位奖励、观测/KL/物理标签监督被屏蔽；根之后的真实转移仍训练。
- `reserve/commit/discard` 保持二维按槽布局。未完成/失败位置不可采样；提交检查位置代际。
- 采样检查整个 T+1 窗口，拒绝未提交数据及环形时间跳跃。样本不足时保留更新额度。
- 到达分支长度仅设置 is_last；不伪造 is_terminal。普通重置/分支跳跃不传播回报。

上下文从真实起点的时间、计步、任务属性和已知字段，加上想象动作链构造。
不读取截断时刻的未来真实快照。分支长度受来源 episode 剩余时间限制。

## 候选与交互混合

只有普通来源、实际发生截断、边界处 full_adv>0 的候选可以入池。
保留已有 normalized `imag_adv_full` 日志；内部 raw full_adv 不另写同名重复日志。

| 参数 | 默认值 |
|---|---:|
| 候选容量 / Top-K | 1024 / 32 |
| 候选有效期 / 刷新周期 | 128 / 8 次更新 |
| 单次去重后最大新增 / 每来源最多保留 | 64 / 2 |
| 标准化 deter 余弦相似度阈值 | 0.98 |
| 首动作 RMS 差阈值 | 0.1 |
| 重建比例上限 / 同时重建槽上限 | 10% / 2 |
| 分支最大策略步 / 同槽连续分支上限 | 64 / 2 |
| 额度上限 | 128 策略步 |

先过期淘汰和近重复合并，保留较高 Adv；再从排名前K的候选中均匀随机、无放回选取。
相似度只是候选多样性启发式，不是恢复质量保证。

门槛通过且达到100,000 env steps后，从零额度开始：每9个普通策略步挣1个重建步。
只在槽空闲时调度，不中断普通 episode。启动前预占额度，提前结束只退未执行部分。
额度不足64时可启动较短分支；无候选、额度或并发位置时普通 reset。
失败/丢弃分支已经执行的动作仍计入交互步数；纯恢复查询不计步。

不提高重建数据的 replay 权重。日志包含交互比例、训练有效步占比、拒绝原因、
穿透深度、整段丢弃率、候选过期/合并、候选相似度、预算额度和调度回退。
重建片段分数单列为 `reconstruction_episode/*`，不混入普通 `episode/*`。
训练结束输出 `reconstruction_summary.json`。

## 验证入口

```bash
TMPDIR=/mnt/phx2/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -m unittest discover -s tests -p test_online_physics.py -v

# 仅手动运行的小型 GPU 接线测试；先核对 GPU6/7 的实时资源。
TMPDIR=/mnt/phx2/tmp CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. .venv/bin/python tests/smoke_online_reconstruction.py
```

接线测试用真实核心状态作为恢复成功路径的 oracle fixture；生产代码始终调用 learned decoder。
测试通过不等于重建精度或探索收益通过，正式精度、阈值误拒率与收益仍需实际训练评估。

replay 仅保留原有一份 `stoch/deter`，普通位置是可更新缓存，分支根是不可更新种子。
相比首版重复存储，默认 size12M、50万行、float32 latent 时减少约4.77 GiB常驻显存。
当前 batch 的分支种子张量仍有小量临时开销，float64 标签/上下文保持不变。
修改仅对新启动进程生效，不会释放正在运行的旧进程已分配的存储。


## 模块接口与兼容性

- `physics_state.py` 是在线/离线共用的状态编码与 MLP 底座，不依赖 probe、环境或训练器。
  `physics_probe.decoder.StateCodec` 仍保留原导入路径；旧 probe 导出和主模型 state_dict 的参数键保持兼容。
- 在线 decoder 使用自己的 `export()/from_export()`，格式标记为 `online_physics_mlp`，
  同时保存 EMA 和归一化超参数；离线 probe 继续使用其原有格式。
- `EnvStepResult` 为进程 RPC 返回值命名，三种采集路径共用 `ParallelEnv._pack_steps()`。
  CPU 结果保持在 CPU，仅 CUDA 目标启用 pinned memory。
- `PhysicsContext` 集中维护紧凑 context vector 的编解码和想象时间推进；replay 的张量布局不变。
- Buffer 通过 `enable_transactions()`、`has_pending()`、`retention_frames()` 提供重建所需查询。
  Controller 不直接读取其底层 storage。
- Dreamer 提供 `configure_reconstruction()`、`take_reconstruction_candidates()`、
  `candidate_distance_statistics()`、`decode_physics()` 和 `act_with_seeds()`；
  latent 拆解由 RSSM 的 `state_from_feat()` 负责。
- GeometryMonitor 构造不修改环境。`install_hook()/uninstall_hook()` 明确管理 hook 的唯一所有权，
  `install_thresholds()` 只安装阈值。重复 setup 不额外 reset、不消耗新的环境随机数。
- 训练器负责保存诊断文件；Controller 只返回状态/指标并管理采集流程。
  视频日志遇到 `ReplayNotReady` 会跳过本次视频，训练更新额度仍按原规则保留。
