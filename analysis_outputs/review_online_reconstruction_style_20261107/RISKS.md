# 具体发现

## F1 [P2, 明确接口不兼容] 继承离线 decoder 没有保留构造/导入契约

位置：physics_decoder.py:9、:13；physics_probe/decoder.py:132、:157。
OnlinePhysicsDecoder(input_dim, config) 继承 StateDecoder；父类 from_export 调用 cls(input_dim, schema, hidden, dtype=...)，与子类签名不兼容。
通过子类调用继承的 from_export 会发生参数错误。改用 StateDecoder.from_export 也不能直接严格加载在线导出的额外 EMA/schema 缓冲。
当前主训练使用 state_dict 保存/加载，不会自动触发这个入口。问题在于继承暴露了不可用的接口，且主训练依赖了离线 probe 层。
建议：共享 StateCodec/网络底座，在线与离线类各自明确归一化和导入导出接口；不强行复用整个实验类。

## F2 [P2, 明确条件性遗漏] 新的 replay 可用性契约没有覆盖视频日志

位置：buffer.py:85；trainer.py:207、:238。
更新循环捕获 ReplayNotReady 并保留额度，但 video_pred_log 路径仍直接调用 replay_buffer.sample()。
当 video_pred_log=true 且没有可采样窗口时，该异常会从日志路径传播出 begin()，而不是等待后续数据。
建议：覆盖所有 sample 调用点，视频日志在此状态跳过；补针对该配置分支的测试。

## F3 [P2, 维护风险] 重建控制器穿透组件边界，职责过多

位置：reconstruction.py:176、:195、:293、:331、:359、:367。
Controller 直接开启 replay.transactional、检查 replay.pending、读取 replay._buffer.storage.max_size；同时手工拆 RSSM feature、读取 decoder normalizer、清空 agent.reconstruction_candidates。
step() 同时调度、推进环境、处理失败、标定、管理事务、组装字段和写 JSON；act()/ingest()/record() 又掌握模型/存储细节。
建议：保留一个薄协调器，给 Buffer 增加事务状态/保留范围查询，给 agent 明确的 seed/action 和候选取出接口；持久化由上层调用。
这些是已存在的耦合，不是已证明的当前训练故障。

## F4 [P3, 重复逻辑] 环境结果打包有三份实现

位置：envs/parallel.py:70、:85、:107。
普通、仅标签、重建路径分别堆叠观测、处理 reward、构造 TensorDict、pin_memory 和 lift_dim。
pin_memory 的条件已经不完全一致。后续改字段/dtype/设备逻辑需要同步多处。
建议：保留不同 RPC 入口，共用结果打包函数，不另造通信框架。

## F5 [P3, 可读性/协议风险] 跨文件协议依赖裸元组与位置索引

位置：envs/online_physics.py:100、:216；envs/parallel.py:113；reconstruction.py:133、:184。
context_vector 的 0/1/2/3:，7元组返回值的 r[3]/r[4]/r[5]/r[6]，以及 active[slot]=[reserved,used]，都需跨文件记忆含义。
建议：为 RPC 结果和预算记录命名；replay 保留紧凑张量格式，但将字段编码/解码集中到统一 schema/helper。

## F6 [P2, 兼容性风险] 环境 wrapper 层次与 MuJoCo hook 的假设隐式存在

位置：envs/online_physics.py:17、:144；reconstruction_guard.py:74、:117。
time_limit=env.env 固定了 wrapper 层次；GeometryMonitor 构造时直接替换 physics.step，并维护 dm_control 的积分调用顺序。
逐子步检测有实际技术需求，不能简单移除或改为逐策略步检查。应集中为明确的适配层，约束重复安装/恢复，并保留版本与无干扰测试。
目前不能把这种风险直接当成已经产生动力学错误的证据。

## F7 [P3, 风格] 新模块排版整齐，插入旧模块的片段未完全收尾

位置：dreamer.py:718、:724；rssm.py:157。
这些新增表达式/字典布局会被 Black 改写；函数内的标准库 import、字段命名（config.physics_decoder 对应 agent.physics_head）也可做局部统一。
DMC/MuJoCo 的按需 import 有可选依赖方面的理由，不宜一律移到全局。
建议：只整理此次新增片段；不把其他工作区已有的注释风格问题混入本次范围。
