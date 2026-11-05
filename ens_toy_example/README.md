# 五个 categorical heads 的分布外分歧实验

已完成的实验与中文判读见 [RESULTS.md](RESULTS.md)。

目的：固定数据、目标和 minibatch 顺序，仅改变网络初始化，检查分布外是否产生分歧，以及分歧是否能排序真实预测误差。受论文 Figure 1 启发；这是独立设计的条件分布回归实验，不是论文 toy 数据或结果的复现。

## 预先固定的协议

- 真分布：`Y|x ~ Uniform(m(x)-s(x), m(x)+s(x))`，`m(x)=2sin(2x)+0.4x`，`s(x)=0.35+0.15cos(2x)^2`。均值与噪声尺度均随 x 变化，真 CDF 已知。
- 训练输入仅在 `[-1,-0.3] ∪ [0.3,1]`；中间缺口 `(-0.3,0.3)` 和外侧 `[-2,-1) ∪ (1,2]` 均没有训练数据。
- 固定 4096 个训练样本、2048 个独立 ID 验证样本。五头看到完全相同的标量 y、相同 minibatches；没有每头数据重采样、额外噪声或随机 prior。
- 直接使用仓库 `networks.MLPHead`：3 层、宽度 64、SiLU、RMSNorm、255 logits、仓库初始化、`outscale=0`。因此初始 logits 相同，隐藏层权重不同，训练后才可能产生分歧。
- 所有成员使用固定输入 `[x,1]`。常数坐标避免一维 x 在零 bias 的第一层 RMSNorm 中丢失幅度；不是给每个成员不同特征。
- `repository` 主实验复用 `symexp_twohot` 的 255 个原始回报 bins，范围约 ±4.85e8；`bounded` 对照使用 [-5,5] 的 255 个等距 bins，覆盖整个真分布。
- bounded 对照同时改变支持范围与中央分辨率，不能将其全部差异仅归因于尾部。另用 `analyze.py` 对同一训练结果计算中央 [-5,5] CDF 积分，以隔离积分范围的影响，不重新训练或归一化概率。
- 两者都用与 `TwoHot.log_prob` 等价的标量 two-hot 交叉熵，已测试 loss 和梯度一致。Adam lr=1e-3，batch=128，3000 次更新，每250步验证；用平均 ID validation NLL 选择同一时刻的整个 ensemble。没有用 OOD 误差选 checkpoint。
- ensemble seeds 11/22/33，每组成员 seed 为 `100*ensemble_seed + [0,1,2,3,4]`。重复之间仍固定数据和 minibatch 顺序，衡量初始化影响；不把它当作独立数据重复。

## 判读

分开测量三个问题：OOD 分歧是否升高；是否区分 ID/OOD；是否预测真实误差。OOD 检测 AUROC 高不等于误差识别好。

- 分歧：10 对 heads 的平均 W1，float64 softmax/CDF/积分，与仓库实现核对；另存 head 均值的 sample std。
- 真误差：ensemble mixture 与连续真分布的精确 W1；另存与精确 two-hot 投影真分布的 W1，以及表示投影本身的误差。
- ID、缺口和外推区域分别计算 Spearman、筛选风险；避免只报告混合全部区域的相关。
- risk–coverage 衡量保留低分歧样本后的真实误差；对照随机筛选期望、距训练支持的距离、oracle 真误差。分数完全相同时用 tie-aware 期望，避免输入排序伪造收益。
- 阈值使用 ID 验证 x 上分歧的 95% 分位数，仅作诊断，不代表有理论保证的置信界。
- 额外报告 W1 中 [-5,5] 外尾部积分占比。仓库巨大支持上，即使中央分位数看似拟合良好，极小尾部概率也可能主导原始回报 W1；中央图明确裁剪显示范围，不能仅凭图判断完整分布拟合良好。
- 主运行的 `w1_tail_fraction` 将跨越 ±5 边界的整个 bin 间隔计入尾部，属于近似诊断；`analyze.py` 输出的 `exact_tail_fraction_median` 使用精确的区间裁切。后续报告使用后者。
- 网格点彼此相关；跨 ensemble seed 的均值/SD 是描述统计，不做网格点独立样本显著性检验。

## 运行

从仓库根目录运行。仅 CPU，脚本内部限制单线程；共享服务器建议低优先级，禁止覆盖已有输出目录。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -m unittest ens_toy_example.test_toy -v

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -u -m ens_toy_example.run \
  --output logdir/ens_toy_example/repeat_001
```

对完整的两种支持实验生成叠加图和中央范围诊断：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -m ens_toy_example.analyze \
  --output logdir/ens_toy_example/repeat_001
```

本次完整结果使用 `run_20261104_v2`；`run_20261104` 保留被 SIGTERM 中断的记录，`smoke_20261104` 仅验证接口。训练每250步保存当前最佳模型和 loss 历史；这些阶段文件不包含完整 optimizer/RNG 续训状态。

依赖已有 torch、numpy、omegaconf，此外使用 scipy 和 matplotlib。结果目录保存数据、source/data hashes、参数、每次验证 loss、选中模型、逐点预测、JSON 指标、PNG/PDF 图和总报告。仅修改本目录，训练主程序不受影响。

## 适用范围

这是监督分布回归，不包含 RSSM、共享可训练 encoder、bootstrap targets、slow targets 或动态截断。Adam 设置和网络宽度是小规模 toy 设置，不是完整 Dreamer 配置。结果只能回答该回归问题上“不同初始化是否足以产生有用分歧”，不能证明或否定真实 RL 中的五头方案。训练区间外可以存在多种与同一训练数据完全兼容的真函数，因此任何单一 toy 成功也不构成通用 OOD 保证。

## 加入五头 quantile 的比较

本次已完成的比较与结论见 [QUANTILE_RESULTS.md](QUANTILE_RESULTS.md)。

`compare_quantile.py` 复用已经完成的 categorical 结果和数据，不重训或改动历史结果。每个 ensemble seed 新训练5个 quantile 网络，仍为3层64宽度、SiLU/RMSNorm、255个输出和零输出层初始化；隐藏层初始化使用独立 seed 流 `(seed+10000)*100+i`。

quantile 输出不做 softmax，而是255个可学习的等概率 atom 位置。用固定中点分位数 `tau_k=(k+0.5)/255` 的精确 pinball loss 拟合同一批标量标签，无 Huber 近似。完全继承 categorical 的数据、batch 顺序、Adam、学习率、3000步预算及每250步验证频率；quantile 按 ID validation pinball loss 选 checkpoint，categorical 按先前 ID validation NLL 选择。两种 loss 数值不能直接比较。

距离均用同一原始 y 坐标的精确 W1：

- C–C：5个 categorical 的10对无序成员平均距离。
- Q–Q：5个 quantile 的10对无序成员平均距离。
- C–Q：全部25对跨组成员的平均距离，不是两组均值或 mixture 之间的距离。

异构支持通过合并 atoms 后积分 CDF 差精确比较；不把 categorical 的 logits 与 quantile 的位置直接相减。quantile 训练不强制单调，评估将输出视为等质量离散 atoms 后排序，另报告相邻分位数 crossing 比例。

每组 quantile 同时与原始 bins、有限 bins categorical 比较，因此总计新训练15个网络而非30个。保留全范围 W1 与 [-5,5] 中央积分对照。记录 ID/缺口/外推的分歧均值、OOD/ID比例、相对 ID 增量、OOD检测AUROC，以及跨组分歧同时高于两种组内分歧的输入比例。跨表示差异在 ID 就可能非零，不能把“大”自动解释为“更准”。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -m unittest ens_toy_example.test_quantile -v

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -u -m ens_toy_example.compare_quantile \
  --categorical-root logdir/ens_toy_example/run_20261104_v2 \
  --output logdir/ens_toy_example/quantile_compare_repeat_001
```

输出的 `uncertainty_comparison.png/pdf` 将三种分歧画在同一坐标系中；`prediction_comparison.png/pdf` 将两类各5个成员的中位数/90%分位数叠加，并显示完整随机真分布的带状范围。阴影区分训练支持与 OOD。

另有线性纵轴的 `bounded_disagreement.png/pdf` 简图。分歧曲线的色带为3个初始化重复的观测最小—最大范围；绿色背景表示训练输入支持。表格仍报告均值和样本SD。

## 平滑变化的数据密度与非等宽区域

已完成的实验、图表与判读见 [DENSITY_RESULTS.md](DENSITY_RESULTS.md)。

`density_sweep.py` 使用新的输入采样分布，保留原有 `Y|x` 条件均匀分布、网络、loss和训练预算：

| x区域 | 覆盖程度 | 未归一化输入密度 |
|---|---|---|
| [0,1) | 多 | 1 平滑下降至0.35 |
| [1,1.5) | 一般 | 0.35 平滑下降至0.08 |
| [1.5,2.1) | 少 | 0.08 平滑下降至0 |
| [2.1,2.5) | 无 | 恒为0 |
| [2.5,3.2) | 少 | 从0平滑升起，最高0.08，再平滑降至0 |
| [3.2,4.2] | 无 | 恒为0 |

前三段使用三次 smoothstep；第二个稀疏区使用 sin² 小峰。内部边界的密度与一阶导数连续，只有绘图/统计区域标签是分段的。密度积分为0.8345，用拒绝采样生成4096个训练和2048个验证样本。空白区域严格无训练或验证数据。记录实际计数与计数/区间长度，不能仅按总数比较不同宽度的区域。

每个模型族仍为5头，初始化重复为11/22/33，三个重复共享数据。训练3种模型族：原始bins categorical、有限bins categorical、quantile；每族每次3000步，单线程CPU、低优先级。模型选择使用与训练相同密度的验证集，没有空白区域标签参与选模；该选择会更重视高密度区域。

评估网格均匀覆盖 [0,4.2]，区域指标是对区域内x均匀平均，**不按训练密度加权**。分别报告 C–C、Q–Q、C–Q 和两类模型的真分布误差。区分两个稀疏区、内部空白和末端空白，不把它们合并。保留全范围和中央[-5,5]积分，尾部效应不会被隐藏。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -m unittest ens_toy_example.test_density -v

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ens_toy_mpl nice -n 19 .venv/bin/python -u -m ens_toy_example.density_sweep \
  --output logdir/ens_toy_example/density_repeat_001
```

主图 `density_comparison.png/pdf` 同时画理论输入密度、实际采样直方图和三种分歧；`predictions.png/pdf` 画10个成员的预测与完整真分布范围。`coverage.json`、`region_summary.csv` 和 `summary.json` 保存区域数据。默认运行相同预算，不根据OOD结果选seed或模型。

这一设计回答“不同覆盖区域里表现如何”。由于密度与x位置、目标函数曲率、插值/外推几何同时变化，它不是隔离数据量因果影响的实验；不能预设分歧必须随样本减少单调增加。与先前实验的x覆盖范围也不同，不直接将两个实验的数值当作同条件性能比较。
