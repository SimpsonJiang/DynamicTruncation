# 适配注意事项

- 上游 MetricAccumulator.add 对每个张量执行 reshape(())。本地 dreamer.py:555 返回 (H,1) 的不确定性指标，trainer.py:183 使用 vector 日志。H>1 时直接接入该累积器会失败；应保留向量形状逐元素累积或单独记录。
- 第一个提交改变离散 latent 梯度、KL 和计算精度；即使只引入修复，也不能把新旧训练结果当作相同实验设置。适配需保留本地 value ensemble 和动态截断逻辑。
- 第二个提交默认开启新窗口优先采样并关闭 latent 回写，改变数据分布及 RSSM 上下文来源；不是纯日志修复。
- TwoHot 新计算利用 bins 正负反对称。本地 symexp_twohot 满足；自定义非对称 bins 的用法应另行核查。
- BlockLinear 原标准初始化会清零 bias，此项主要增强初始化前的安全性，不应推断旧训练必然使用垃圾 bias。
- BF16 的适用性、吞吐和训练效果未在当前硬件验证。ReturnEMA 的 quantile 修复不涵盖本地动态截断额外调用的 torch.quantile。
- [Issue #9](https://github.com/NM512/r2dreamer/issues/9) 报告的 action 损坏比例属于报告者环境，不能外推为本地实际比例。
