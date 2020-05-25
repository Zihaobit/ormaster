---
layout: post
title:  "Bayesian Method and Probabilistic Programming"
date:   2017-03-28 00:00:04
categories: liuqianchao update
---


### 1. Basic Defination

&emsp; **概率图模型**，构造该模型的目的是为方便表达变量间的概率相关关系。概率图模型大致可以分为两类：“有向无环”的贝叶斯网络和“无向图”马尔科夫网。我们常说的**马尔科夫链**，其实是有方向的贝叶斯网络。在时间序列中其方向表示时间的推进。之所以称之为“隐”马尔科夫链，是因为我们能获取的信息是马尔科夫链上的各个Node是观测值，而实际上，我们假设每个Node是状态变量，变量空间是N个可能的离散取值；同时在每个Node上的观测值依赖于该状态变量，其取值可以是离散或连续的。   
&emsp; **马尔科夫链**， 其某一时刻的状态变量仅与上一时刻相关（无记忆性，构造该特性的目的是为了简化复杂的现实模型）。马尔科夫链的组成元素包括状态变量空间、观测变量空间以及：    
  1. 状态转移概率$$a_{ij}= P(y_{t+1} = s_j \mid y_t=s_i)$$， 状态转移矩阵是一个$$N\times N$$的矩阵   
  2. 输出观测值概率$$b_{ij} = P(x_t = o_j \mid y_t=s_i)$$   
  3. 初始状态概率，即在初始时刻各状态出现的概率$$\pi_i = P(y_1 = s_i)$$ 其中 $$1\leq i \leq N$$   

&emsp; **马尔科夫链的应用**  
  1. 根据模型参数（$$a_{ij}, b_{ij}, \pi$$）推测观测，根据历史的观测值，预测未来的观测值。比如根据历史时间序列，预测将来的时间序列。    
  2. 根据模型参数（$$a_{ij}, b_{ij}, \pi$$）以及观测值$$x$$推测状态变量。比如语音识别任务中，将语音认为是观测，文本认为是状态变量。
  3. 给定观测值x, 推测模型参数（$$a_{ij}, b_{ij}, \pi$$）。


### 2. 马尔科夫随机场


### Plus 马尔科夫链中的经典问题

&emsp; **Gambler's ruin problem**, A gambler, at each play of the game: Probability $$p$$ of winning 1$, probability $$1-q$$ of losing 1$. At the beign, he has i$. Question: The probability that fortune will reach N before reaching 0.     
&emsp; 对于马氏链上的Node，我们定义状态$$X_n$$为时间n下该赌徒的财富，基于该状态可以定义状态转移概率，我们有$$P_{00} = P_{NN} = 1$$。同时$$\forall i \in 1...N-1$$， 有$$P_{i, i+1} = p, P_{i+1, i} = 1-p $$。 我们用$$P_{i}$$表示从i美刀，到能取得N财富的概率，则可以建立递推关系：   

$$P_{i} = p \times P_{i+1} + (1-p) P_{i-1}$$

&emsp; 该式子等价于：$$(P_{i-1} - P_i) = \frac{p}{q}(P_i - P_{i+1})$$, 递推，两边求和，且$$P_0 = 0$$，可以得到$$P_i$$的公式。   
&emsp; 或者，该问题可以使用Recurent, Transient来解释，在该赌博问题中i = 1....N-1为Transient状态，而i = 1或N 为Recurent状态。根据从$$T\rightarrow R$$下

$$f_{ij} = \sum_{k \in T} P_{ik}f_{kj} + \sum_{k\in R} P_{ik}$$

&emsp; 故，有：

$$f_{iN} = P_{i,i+1}f_{i+1, N} + P_{i,i-1}f_{i-1, N}\\
= pf_{i+1, N} + (1-p)f_{i-1, N}
$$


Reference:   
周志华《机器学习》
