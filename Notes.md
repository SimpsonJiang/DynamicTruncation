由于tools.setup_console_log()的作用，所有print在终端上的东西都会被存储到log中，这是个非常重要的功能，甚至之后我自己的代码库也可以采用之。


# Replay buffer
本项目的 episode 实际是环境槽 ID，因此采样片段可能跨真实 episode；模型依靠 is_first 重置记忆

Buffer把data存储为TensorDict形式，其ndim=2(即(N,T_total)，后面维度都是对不齐的，这里N是并行环境数，)
```python
# TensorDict简介
TensorDict({
    "image":  ...,  # 所有样本的图像组成一个 Tensor (N, T_total, H, W, C)
    "action": ...,  # 所有样本的动作组成一个 Tensor (N, T_total, A)
    "reward": ...,   
    "is_first": ...,# (N, T_total, 1)
    ...
}, batch_size=(N, T_total)), 

data["image"]  # 取 image 字段，得到 Tensor
data[0]       # 取第 0 段序列，得到 TensorDict
data[0, 3]    # 取第 0 段序列的第 3 个时间位置
data[:, 1:]   # 所有序列去掉首步，所有字段一起切片
# 这正是它比普通字典方便的地方：既能按字段取值，也能像 Tensor 一样按 batch 和时间维度索引
```
Buffer各种操作接收（或返回）的张量分别为:

- add_transition()接收(N),即单步时刻的状态(N,1)。
- sample() -> (data, index, initial),data为TensorDict(B,T)，其state的action是“产生该state的action”；index为本次sample的(B,T)个数据在原始replay buffer里的坐标，index[0]是时间坐标(B,T)，index[1]是环境坐标(B,T)；initial是初始值(stoch_0, deter_0)，用于输入GRU的h_0
- update()
用来刷新replay的latent

**每一个iteration搞完，replaybuffer通过一次add_transition增加N条数据，通过一次update更新B*T个数据**所以更新的比增加的多的多

