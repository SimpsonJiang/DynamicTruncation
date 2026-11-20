# 截断重建

本目录集中保存在线截断重建的独立模块。

| 文件 | 职责 |
| --- | --- |
| `controller.py` | 候选池、候选去重、重建预算与分支采集生命周期 |
| `decoder.py` | 在线 float64 物理状态 decoder、训练损失与优化器 |
| `state.py` | 在线与离线 probe 共用的状态编码和网络底座 |
| `guard.py` | 穿模检测、阈值校准、验证及结构化拒绝原因 |
| `environment.py` | 物理标签、仿真状态恢复、分支执行和仿真副本渲染 |
| `protocol.py` | 环境 RPC 返回值与物理上下文的编解码 |
| `visualization.py` | DMC vision 定期生成重建对比图与配套数据 |

调用方直接导入具体模块，例如：

```python
from reconstruction.controller import ReconstructionController
from reconstruction.decoder import OnlinePhysicsDecoder
from reconstruction.protocol import EnvStepResult, PhysicsContext
```

`__init__.py` 不加载具体模块，普通环境使用 RPC 协议时不会额外导入 MuJoCo。

训练接入点仍由原模块负责：`dreamer.py` 产生候选、训练物理 decoder 并提供可视化想象，
`trainer.py` 调度重建及可视化，`buffer.py` 管理分支 replay 事务，
`envs/` 负责通用环境创建与进程通信。通用动态截断算法仍位于 `truncation.py`。
`physics_probe/` 保留离线实验入口，通过 `reconstruction.state` 共用编码与网络底座。

配置、运行方式和算法说明见 [在线重建文档](../docs/online_reconstruction.md)。
