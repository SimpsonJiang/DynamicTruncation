# 2026-10-17 上游提交核对

核对日期：2026-10-30。依据 GitHub API 完整提交 diff、Issue #9 和当前工作区静态阅读；没有运行训练、修改源码或执行 pull/merge。

## 上游提交

- [7a93691](https://github.com/NM512/r2dreamer/commit/7a93691bb5424c2010c901d2931f83d22ee3afbd)：修复 replay action 重叠写入；categorical straight-through 梯度；RSSM KL 的 unimix 一致性；TwoHot 对称 bins 的数值抵消；训练和 act 改为 BF16，移除 GradScaler，非有限梯度跳过 optimizer/scheduler；BlockLinear bias 初始化；ReturnEMA 可追踪分位数；AGC 使用 no_grad。
- [becf338](https://github.com/NM512/r2dreamer/commit/becf3387a495e5291d2fc7af94763bebc2f9afef)：采集时构建每环境 T+1 步窗口，完整在线 batch 优先使用，随后回退普通 replay；在线队列最多两批，过量丢弃旧窗口；新增 online=True / update_latents=False 默认值；视频采样不消费在线队列；张量标量指标改为日志区间均值，非张量指标保留最新值。

## 本地结论

本地 HEAD 为 abcabb1，origin 指向 SimpsonJiang/DynamicTruncation，存在未提交修改。涉及修复的核心位置仍是旧实现：buffer.py:41、distributions.py:16/78、rssm.py:222、dreamer.py:350/588、networks.py:41/400、optim/agc.py:40。

建议在独立同步分支优先适配第一个提交。第二个提交的采样协议和 latent 回写应独立决定。若引入第二个提交但保留旧 replay 策略，可设 buffer.online=false buffer.update_latents=true；这不能恢复第一个提交之前的全部数值行为。

基线与动态截断方法应使用同一修复版本及 replay 配置。未验证分数、速度或稳定性提升。
