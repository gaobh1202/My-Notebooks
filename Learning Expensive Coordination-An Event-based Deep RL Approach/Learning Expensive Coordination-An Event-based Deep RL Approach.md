# Learning Expensive Coordination: An Event-based Deep RL Approach

## 摘要：

针对多智能体强化学习，其中领导者对于每个智能体有效的合作给出奖励，称作expensive coordination。

其中存在的主要问题是：

1. leader给出奖励时需要考虑长期利益和智能体未来的行为（predict）
2. 由于个体之间的复杂交互使得训练过程难收敛

针对这个问题本文提出了一种事件驱动的深度强化学习方法

## 简介：

一般的多智能体强化学习都假设个体是无私的，即可以为了全局目标而牺牲个人利益。

但实际例子中，个体是有self-interest的，所以为了满足全局目标，leader应该为那些牺牲个人利益的智能体施加bonus。——称作expensive coordination

**过去方法存在的问题：**

1. 只关注了静态决定，即每个智能体只做一个决定
2. 每个智能体的preference是固定的，忽略了领导的policy对其的影响

**expensive coordination中需要关注的重点问题：**

1. 领导在选择奖励机制时需要考虑对于**自身的长期影响**以及**个体的长期行为**

2. 领导与个体之间的复杂交互，个体会调整自己的策略从而最大程度的利用奖励政策

**具体方案：**

1. 用semi-Markov Decision Process来建模领导者的决策过程
2. ***一种event-based policy gradient to learn the leader's policy????***

3. 建立leader-follower consistency scheme，并基于该体系提出follower-aware module, follower-specific attention module, sequential decision module来获取个体的行为，并作出回应
4. 提出一种action abstraction-based policy gradient algorithm，通过减少个体的决策空间来加速训练过程

## Related Works

**Leader-follower RL**:一种强化学习方法，通过对非合作个体non-cooperative提供奖励，来最大化领导者个人利益。过往的研究大多只关注于简单表格游戏或小规模马尔可夫游戏。本文的目标是针对基于RL的个体，计算领导者的政策。

**Temporal abstraction RL**：将决策过程划分为2层，高层决定meta-goal（全局目标？），底层决定primitive action（个体的动作？）。然而，本文的决策过程可以自然地抽象为间歇决策过程**semi-MDP**

**Event-based RL & Planning：**本文中，通过将领导者的动作视为某个时间步的时间，来设计一种事件驱动的policy gradient，从而学习领导者的长期政策

## Stackelberg Markov Games:

## Methodology:

- 将领导者的决策过程建模为semi-MDP；提出一种事件驱动的政策梯度event-based policy gradient，来让领导者只在关键步作决策
- 构建一个follower-aware module，其中包括follower-specific attention和sequential decision module。主要用于预测个体的行为，并作出回复
- 提出一种action abstraction-based policy gradient来简化决策过程，加速训练

