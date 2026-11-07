# 在线重建代码风格与结构审查

结论：基础命名/格式基本一致，主训练结构有实质复用；接口边界和职责分配仍有明显技术债，需要局部整理，不需要推倒重写。

本次只读检查生产代码，仅写审查记录。所有 shell 命令设置 TMPDIR=/mnt/phx2/tmp，没有创建临时文件，没有使用 GPU、启动训练或重跑训练测试。

## 检查方法与限制

- 阅读当前源码、工作区 diff、仓库 pre-commit 配置，检查真实调用关系。
- 使用内存中的 AST、Black.format_str、isort.code 检查，未调用格式化写回。
- 本地版本：Black 24.3.0、isort 5.13.2，与仓库固定版本相同。
- 四个新增生产模块通过 Black 120（含 unstable）及 isort Black profile 的比对。
- buffer/trainer/train/envs 相关文件与新增测试也通过此次常规 Black/isort 比对。
- 新模块及所查 buffer/trainer/train/envs/测试文件通过普通 Flake8 检查。
- dreamer.py/rssm.py 整文件不符合 Black；已具体定位到此次新增候选字典、eligible 表达式、logit_dtype 表达式。既有中文行尾注释的 Flake8 问题不归因于本次新增功能。
- 未运行完整 pre-commit（其 hooks 可能写回代码，且还含额外插件）；不据此声称完整 hooks 全通过。
- 下述接口问题由源码调用关系和构造签名确认；本轮没有运行故障复现或重新验证数值结果。

## 已合理复用的部分

- train.py → OnlineTrainer 的原入口和采集/更新循环，Hydra 配置与原有 checkpoint/optimizer 保存工具。
- Buffer 继续使用 TorchRL LazyTensorStorage、SliceSampler 和 TensorDict；未另造 replay 系统。
- ParallelEnv 继续使用原有进程/RPC 和既有环境槽，未另建生产采样环境。
- RSSM.observe/obs_step、Dreamer.act 和当前 posterior 特征被复用；没有复制第二套世界模型/策略。
- CandidatePool、ReconstructionBudget、GuardCalibration 的基础职责划分可以保留。
- float64 独立优化器、辅助标签、种子保护、事务提交、逐子步检查都是已批准需求，不能为了短代码删除。

## 建议次序

1. 修复继承导出接口不兼容与视频日志未处理 ReplayNotReady 两个明确契约缺口。
2. 抽取在线/离线共用的状态编码与网络基础，避免主训练依赖 probe 实验层；明确各自的归一化/导入导出契约。
3. 给 Buffer 和 Dreamer 提供少量明确接口，收回 Controller 对内部存储、latent 布局、normalizer 缓冲及候选属性的直接操作。
4. 共用环境结果打包，给跨模块数据协议命名；集中管理 MuJoCo 子步 hook 的安装/恢复与版本假设。
5. 仅格式化新增/改动片段，不连带清理其他已有修改。

这应是保持行为的局部重构，不应修改算法、阈值、预算、float64 边界，也不需要引入通用训练框架。
