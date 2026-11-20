# 截断重建四项修复（2026-11-19）

已完成 2026-11-17 审查中的 F1–F4。保留原有未提交修改；本次修改八个生产模块及相关文档/测试，没有调整启动脚本、训练预算、候选规则、穿模阈值或五步检查窗口，也没有重启现有训练。

## F1：统一必需配置校验

`Dreamer.validate_reconstruction_requirements()` 集中检查在线物理 decoder、动态截断、可产生截断的 horizon 组合及支持的 representation。`train.main()` 在设置 seed、创建模型或环境之前调用；`configure_reconstruction()` 根据实际模型属性调用同一校验。

开启重建但关闭动态截断，或 `min_horizon >= imag_horizon`，现在会明确报错。重建关闭时不施加这些额外限制，原有普通训练及 decoder-only 模式保留。

## F2：损失保持固定形状

- R2 的 Barlow loss 改为掩码均值、样本方差与相关矩阵；没有行压缩和对角线布尔索引。有效样本不足两行时返回零损失，常数列的梯度保持有限。
- 物理 decoder 的损失和 EMA 使用掩码归约。有效行数为零时，损失和梯度为零，EMA、更新计数保持不变。
- 标签精度/有限性验证保留在 `Dreamer.update()` 的编译边界之外；直接调用 `training_loss()` 默认仍验证，内部已验证调用显式使用 `validate=False`。
- AMP 下 R2 统计量使用 float32 累加。数学定义保留原有 `n-1` 样本方差；不承诺与旧半精度统计逐位相同。

CPU 检查使用 `torch.compile(backend=..., fullgraph=True)`，在固定 batch 形状下依次切换 0、1、3、6、2、0 条有效行。R2 loss 和实际物理 training loss 分别只产生一张图，没有图断裂或因这些掩码值而重新编译。还验证了损失、特征/参数梯度和 normalizer 状态与压缩有效行参考计算的等价性，含周期关节与四元数。

这些是两项损失函数的 CPU 编译检查，不是整个 `_cal_grad` 的图数、CUDA 编译计时或正式训练速度测试。

## F3：保存可追溯的拒绝详情

`UnsafeState` 保留稳定原因码及结构化详情。穿模失败复用已计算的深度，记录实际超阈几何对 ID/名称、深度和该对阈值；多对超阈时选择绝对超阈量最大的一对，不把其他几何对的总体最大深度当作原因。

环境在 reset 前保存恢复/续跑阶段和分支相对策略步：恢复点是 0，执行动作后的检查依次为 1、2……。Controller 补充环境槽、训练步和来源，Trainer 将每个事件追加到运行目录的 `reconstruction_rejections.jsonl`。事件读取后排空，结束时重复写诊断不会重复追加。物理异常使用稳定分类计数，原始异常消息保留在事件中。

没有新增物理查询；MuJoCo 碰撞过滤及恢复点/前五步的检查范围不变。结束时 guard profile 也会刷新，避免训练在同一标定阶段结束时保留旧计数。

## F4：集中分支启动和收尾，补 Controller 协作回归

Controller 的 `_start_branch()` 配套创建请求、预占预算和记录候选；`_finish_branch()` 配套完成 replay 提交/丢弃、统计、预算释放与候选清理。拒绝、完成和关闭均通过同一收尾入口，关闭可重复调用。

新增 Controller 测试同步调用两槽真实 MuJoCo 环境及实际 `ParallelEnv` 打包、TorchRL replay，只替代进程 RPC 和 decoder 输出；renderer 被替换为小型数组。成功恢复使用已知 source core 的 oracle fixture，检查的是协作契约，不代表 learned decoder 精度。覆盖：

- 恢复拒绝：完整退款，存入普通 reset，不创建 pending；
- 分支中途拒绝：整段不可采样，只退未执行预算；
- 同批两槽：一个自然提前结束提交，另一个拒绝丢弃；
- 正常达到预定长度提交、根 latent 不被回写覆盖；
- replay 环形覆盖导致提交失败时，计数及状态清理正确；
- 训练结束仍有 pending 时丢弃，重复 close 不重复计数；
- 拒绝字段在 reset 后保留，JSONL 写入一次。

## 验证

- 全仓库 CPU unittest：**184 项通过，82.143 秒**，日志为本目录 `unittest.log`。
- 新增 18 项回归；其中 Controller 7 项、固定形状损失 6 项、配置接口 3 项、geometry 诊断 2 项。
- Black 24.3.0 / 120 列 / unstable：十个完整文件通过；dreamer/networks 只格式化本次相关片段。
- Flake8：上述十个完整文件，以及 dreamer/networks 的新增/修改行通过。
- isort Black profile：十二个相关文件通过；`git diff --check` 通过。

测试固定 CPU 88、nice 19，OMP/MKL/OpenBLAS 各一线程，隐藏 CUDA 设备，临时目录为 `/mnt/phx2/tmp`。格式化 CLI 的工作进程在当前沙箱中挂起，已清理；最终使用 Black 的串行库接口完成格式检查，没有遗留格式化进程。未运行 GPU smoke、CUDA/Inductor 性能测试或正式训练。

```bash
taskset -c 88 nice -n 19 env \
  TMPDIR=/mnt/phx2/tmp TMP=/mnt/phx2/tmp TEMP=/mnt/phx2/tmp \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
  .venv/bin/python -m unittest discover -s tests -v
```
