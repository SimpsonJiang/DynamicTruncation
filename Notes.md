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

# make env
dmc用了NormalizeActions的wrapper，让action看到的动作都是[-1,1]归一化的，等到agent和环境交互时，再把agent的数据映射回环境原本的范围，对于无界的动作，则直接使用其原始动作

# Dreamer
- \_\_init_\_()：从config里取出各种参数，根据参数创建了所有网络（包括slow value网络），创建了LaProp优化器，agc,scaler，、scheduler

- get_initial_state(self, B) -> TensorDict:
返回0时刻的初始状态
```python
state = TensorDict({
    "stoch": stoch,        # [16,32,16]，初始为零
    "deter": deter,        # [16,2048]，初始为零
    "prev_action": action, # [16,A]，初始为零
}, batch_size = (B,))
```

- act(self, obs, state, eval=False -> (action, ):
计算下一时刻的stoch, deter, 用actor head计算下一步的action分布，再采样

- \_cal_grad(self, data, initial) -> ((post_stoch, post_deter), metrics): 
计算一个batch的梯度

# rssm
- \_\_init_\_(): 
使用BlockGRU作为deter_net，保留h表达能力的同时减少了计算量。创建obs_net与img_net，并对三者进行初始化(权重使用LeCun初始化，偏置为0)。

- initial(self, batch_size) -> (deter, stoch):
返回(B, D), (B, S, K)且初始化为0的deter和stoch

- observe(self, embed, action, initial, reset) -> (stochs, deters, logits): 
接受*T步*的embed, action, reset，以及0步时的initial=(stoch, deter), 返回T步的stochs, deters, logits

- obs_step(stoch, deter, action, embed, reset) -> (stoch, deter, logit): 
后验计算。接受*一步*的stoch, deter, action, embed, reset, （如果reset为True,直接把stoch和deter设为0），先由stoch, deter, action经过deter_net(GRU)计算下一步的deter，再通过obs_net计算下一步的logit和stoch

- img_step(stoch, deter, prev_action) -> (stoch, deter): 
先验计算。接受*一步*的stoch, deter, prev_action,先由stoch, deter, action经过deter_net(GRU)计算下一步的deter，再由img_net想象下一步的stoch

- prior(self, deter) -> (stoch, logit):
由imag_net从deter想象下一步的stoch,logit

- imagine_with_action(self, stoch, deter, actions) -> (stochs, deters):
给出初始第0步的stoch,deter和T步的actions, 由img_step想象T步的stochs, deters

- kl_loss(self, post_logit, prior_logit, free) -> (dyn_loss, rep_loss):
计算KL散度（含free-bits），维度为(B, T)

# OnlineTrainer
- begin():
episode_ids = torch.arange(envs.env_num, dtype=torch.int32, device=agent.device)，使得SliceSampler“误认为”只有B个episode，就可以方便的实现“采样序列跨越同一个环境的多个episode”，RSSM通过is_first识别每个episode的起始值
有Train的 log是训练时的数据，episode是采样数据的结果，eval是评估结果（平均）