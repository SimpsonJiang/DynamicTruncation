# 截断重建代码复查（2026-11-17）

结论：模块划分可以保留，但配置校验、可选编译路径、拒绝诊断和分支生命周期仍有局部技术债。本轮确认两个可运行的小型反例、一处明确的诊断信息缺口，以及一处状态管理与测试覆盖风险；没有发现新的候选去重规则、分支动作时间对齐或 replay 根种子保护错误。

审查基于 HEAD `d0741691f71c024207bde9aa34aefdb586e9b572` 加本轮开始时已有的工作区修改。生产代码未改；只新增此目录的审查记录。

## F1：Python 入口缺少重建依赖校验，允许永远没有候选的配置（P2，已复现）

- `train.py:42-44` 只检查重建依赖 physics decoder。
- `dreamer.py:413-418` 的 `configure_reconstruction()` 检查 decoder 与 representation 类型，没有检查动态截断是否开启。
- `dreamer.py:769` 只在 `collect_candidates and truncation_active` 时产生候选。
- `uv_run.sh:50-55`、`run.sh:48-53` 有相关检查，但直接使用 Hydra/Python 入口可以绕过。

用现有小型 CPU Dreamer fixture，关闭动态截断、安装 physics head、调用 `configure_reconstruction()`，配置被接受；随后显式调用 `_cal_grad(..., truncation_active=False, collect_candidates=True)`，结果没有 `_candidates`。正常 controller 仍会开启事务 replay、标定和验证等流程，但没有候选可恢复。

建议把必需关系放在 Python 的统一入口校验。启动脚本可提前提示，但不能成为唯一保证。`min_horizon >= imag_horizon` 等导致不可能截断的组合也值得在开启重建时给出清楚提示；本轮没有为这些额外组合运行反例。

## F2：固定形状修复尚未覆盖所有支持路径（P2，局部编译图断裂已复现）

- R2-Dreamer 的 `dreamer.py:635-636` 仍使用 `x1[valid]`、`x2[valid]`，行数取决于 batch 中根的数量。
- 可选联合物理监督的 `physics_decoder.py:94-104` 包含有限值检查产生的张量条件跳转，以及两次数量可变的布尔索引；`dreamer.py:879-882` 会把该调用放入 `_cal_grad` 的编译范围。

本轮单核 CPU `torch._dynamo.explain` 结果：

| 检查对象 | 图数 | 图断裂数 | 原因 |
|---|---:|---:|---|
| 从 R2 路径提取的同一布尔压缩及标准化操作 | 2 | 1 | `aten.nonzero.default` 动态输出形状 |
| 实际 `OnlinePhysicsDecoder.training_loss`，带有效行掩码 | 4 | 3 | 张量条件跳转、动态布尔压缩 |

这些是局部函数的图数，不能当作整个 `_cal_grad` 的图数，也不能据此声称已测到 CUDA 训练降速。当前两个根启动器选择 `rep_loss=dreamer`，且 physics decoder 默认 `grad_to_wm=false`，上述两条路径不在这一组合的编译主路径中。

建议在 R2 中使用固定形状的加权均值、方差与相关矩阵；联合物理监督应把验证移到编译边界之外，并使用保持形状的掩码归约。需保持原有样本方差、有效元素归一化及不足两行时的语义。

## F3：穿模拒绝丢失关键证据（P2，源码确认）

`reconstruction_guard.py:145-148` 发现越阈后只抛出 `UnsafeState("penetration")`。`envs/online_physics.py:225-230` 将其压成字符串，再由 `reconstruction.py:268-281` 累计拒绝次数及总体最大深度。虽然 worker 内有 `last_depths`，触发的几何对及对应阈值没有随失败报告返回。

因此，训练日志不能直接回答：哪两个部件、实际深度多大、阈值多大、发生在恢复点还是第几个策略步。总体最大深度也可能属于另一对几何，不能替代该证据。这会妨碍区分 decoder 恢复错误与 guard 判定问题。

建议拒绝报告增加稳定的原因码、geom 对、实际深度、对应阈值、分支策略步；可复用已经算好的深度，无需增加物理查询。

## F4：生命周期依赖多处状态同步，controller 回归覆盖不足（结构风险，未复现状态错误）

一次分支同时涉及：

- worker：`in_branch`、`branch_left`、`branch_steps`；
- controller：`active_candidates`；
- budget：`active` 及 Reservation；
- replay：`pending`；
- trainer：`episode_is_recon`。

这些状态有不同职责，不能简单合并掉。问题是启动、失败和完成的配套操作分散在 `_schedule()`、`_process_results()`、`record()` 和 `close()`。例如拒绝路径要配套 discard、finish、移除 candidate 和统计；完成路径则要配套 commit、finish 和移除 candidate。

本轮检查的 `test_*.py` 没有直接构造 `ReconstructionController`。手动入口 `tests/smoke_online_reconstruction.py` 有单槽成功恢复/提交/更新检查，使用已知真值 core 的 oracle fixture，不属于普通 unittest 发现范围。本轮没有运行此 GPU smoke。

建议先为 controller 加少量 CPU 协作回归：恢复时拒绝、前五步中途拒绝、自然提前结束、同一批中一个槽提交另一个槽丢弃、训练结束仍有 pending。然后把提交/中止的成套清理集中到清楚的小方法，保留 worker 与 replay 各自的职责边界，无需引入通用状态机框架。

## 已核对的近期要求

- 候选池：调用内缓存、分块比较、按分数顺序判定；同来源数量限制已删除。
- Guard：尊重 MuJoCo 的碰撞过滤，不再强制查询被过滤几何对。
- 分支：恢复点及前五个策略步检查穿模，第五步包含在内，第六步以后继续数值有限性检查。
- `max_false_reject`：当前构造默认值和 Hydra 配置均为 `0.01`。
- Replay：未提交分支不可采样，提交检查代际，根种子不被 posterior 回写覆盖，动作移位仍有 `.clone()`。
- 想象：仍完整计算 H+1 后再截断，不能据此声称减少 RSSM rollout 计算。

## 本轮验证及边界

固定 CPU 88、nice 19、OMP/MKL/OpenBLAS 各 1 线程，隐藏 CUDA 设备，临时目录设置为 `/mnt/phx2/tmp`。相关现有测试 **53 项通过，16.443 秒**：

```bash
taskset -c 88 nice -n 19 env \
  TMPDIR=/mnt/phx2/tmp TMP=/mnt/phx2/tmp TEMP=/mnt/phx2/tmp \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYTHONPATH=.:tests \
  .venv/bin/python -m unittest \
  test_candidate_pool test_geometry_guard test_online_physics \
  test_reconstruction_interfaces test_replay_latent_storage -v
```

额外运行了上述配置反例和两个 Dynamo CPU 检查，结果见 `status.json`。没有启动训练、CUDA 编译计时或 GPU smoke，没有重跑全仓库测试；CPU 结果不证明 learned decoder 的恢复精度、正式训练耗时或研究收益。

当前 `uv_run.sh` 顶部为 `dyn_rec=off`，`run.sh` 为 `dyn_rec=on`。本轮只记录现状，没有把这一手动配置差异判为 bug。
