---
layout: post
title:  "Online Learning"
date:   2017-07-27 00:00:04
categories: liuqianchao update
---


### 0. Question Description

&emsp; 能否快速地迭代产品决定了其最终能否取得成功，比如通过引入风洞，莱特兄弟能够快速及时地测试、更新自己的设计，最终取得了试飞的成功。而在机器学习领域的“在线学习”则是为快速迭代模型应运而生的。其解决的核心问题是如何用新产生的用户数据，动态地更新我们训练好的模型。   
&emsp; 理论上，如果机器学习算法在“上线”前，我们已经获取到“完全”的数据并且在模型空间中包含了“正确”模型，那么我们在线下就可以完成模型的训练。之所以要进行线上学习，是因为我们在算法上线前只掌握局部的、有偏的数据。   
&emsp; 大部分的机器学习模型本身是不具备“持续”更新的特性的，比如经典的Logistic Regression, 其加入新的数据后，往往需要重新训练所有的参数。那么有没有什么更新模型的pipeline（注意在线学习不是一种新的机器学习的模型，其只是一种操作流程），可以使我们的模型能够在加入新数据后，不必重头开始训练呢？下面介绍两种最经典的思路: BPR(Bayesian Probit Regression)和FTRL(Follow The Regularized Leader).

### 1. Online Learning and regret minimization
&emsp; 对于在线学习，我们最小化的目标一般叫做“后悔”(regret)。regret用来刻画我们的某种参数更迭方案与使用当前所有数据能得到的一组固定参数的差。用公式来表示**Online Learning的目标**，对于每轮我们拿到的数据$$z_t$$:

$$regret_k = \frac{1}{k}\sum_{t=1}^{k}f(\theta_t, z_t)-min_{\theta^*\in \Theta}\frac{1}{k}\sum_{t=1}^{k}f(\theta^*, z_t)$$

&emsp; 我们希望能当k趋向于无穷时，$$regret_k$$能趋向于0。这里可以对比一下**Offline学习的目标**：

$$regret_k = \frac{1}{k}\sum_{t=1}^{k}f(\theta_t, z_{k})-min_{\theta^*\in \Theta}\frac{1}{k}\sum_{t=1}^{k}f(\theta^*, z_t)$$

&emsp; 可以看出，对于Offline学习，我们只是希望训练最终output出的参数能离全局最优参数比较近；而对于Online学习,我们认为每个之前轮次的样本t训练产生的loss均被进入最终的loss，所以这就要求我们关注整个训练过程任何时刻产生的loss。由上面两组目标的差异，我们不难理解前者更适合线上数据的实时迭代，一定程度上保证了前面批次的样本的预测结果不至于太差；此外，在线学习的另一好处是帮助我们处理大规模（TB级）数据，因为每个样本我们仅需要在训练时使用一次，所以可以边读取数据边训练模型。

&emsp; 一种最简单的在线学习的方法是使用OGD(Online Gradient Descent)，这里需要指出OGD实际上就是我们了解的SGD，这里使用online，只是为了强调在绚练的过程中不再强调样本是IID的了。

$$\theta_{k+1}=proj_{\Theta}(\theta_k-\eta_k\Delta  l(\theta_k))$$

&emsp; 其中$$proj_{\Theta}(v) = argmin_w\|w-v\|_{2}$$是将v进行映射的函数。这中映射一般发生在参数需要归一化等情况下，只是对梯度下降得到参数做变形。

<!-- ### 2. BPR

&emsp; 贝叶斯在线学习（Bayesian Online Learning）把上一轮更新后的模型看做是先验分布，通过加入新的样本数据，更新得到后验分布。后验分布就是本轮更新模型的结果。熟悉贝叶斯思想的同学应该了解，经典的MCMC算法中便运用了该思想。   
&emsp; 由上述的基本思想可知，BPR要求先验分布和后验分布是同分布(先验与似然函数共轭)的，否则这一轮的后验分布不能作为下一轮的先验分布。在实践中当这一假设不成立时，可以将后验分布近似，近似(比如KL距离大小)到其先验的分布。
&emsp; BRP是贝叶斯在线学习最经典的例子，该模型认为可以假设机器学习模型的参数服从独立的正态分布，

$$
p(w|y) \sim p(y|w) p(w)
$$ -->

### 2. FTRL-Proximal
&emsp; **FTRL**是从传统的求解方法Gradient Descent的基础上一步步发展而来的。从FTRL的名字（Follow the regularized leader）中可以看出其尝试解决Regularization的问题，因为在线学习的过程中，我们一般希望得到的模型更具泛化能力，而不是严重依赖于某些数据批次，因此在SGD的基础上，我们想要进一步产生稀疏的参数w（当参数w大部分为0时，也方便了数据存储），下面介绍如何实现这一目标。   
&emsp; 一种方法是直接加入参数的L1正则项，这也是**LASSO**(Least Absolute Shrinkage and Selection Operator)的最关键思想。但是由于Norm 1的存在，该惩罚函数在$$x=0$$处不可导，可以使用次梯度（Subgradient）来代替梯度。    
&emsp; 虽然正则项的加入，可以一定程度上趋势在线学习的结果具有一定的稀疏性，但是其实践效果并不佳。主要原因是即使引入了L1正则项，更新后的参数恰好为0的概率也微乎其微（一旦不恰好为0，那么数据稀疏化的目标就没真正实现）。根据这个问题，也有一些解决方案提出（Truncated Gradient, 2009 Microsoft），比如如果参数足够小（设置阈值），直接令其等于0，但这种方法也有一定的局限性，因为参数小，有可能是因为其确实是无意义的特征；但也有可能是因为该特征刚被更新少数次，这种情况下做截断就不合理了。   
&emsp; 另外一个解决方案是Google的研究员提出了一种叫做**FOBOS**（Forword Backward Splitting）的更新策略，其将参数更新的过程分为两步，第一步进行梯度下降，第二步处理正则化加入稀疏性。：

$$w_{t+\frac{1}{2}} = w_t - \eta_tg_t\\
w_{t+1} = \mathop{\arg\min}_{w} \frac{1}{2}\|w-w_{t+\frac{1}{2}}\|_2^2 + \eta_{t+\frac{1}{2}}\lambda\|w\|_1
$$

&emsp; 两步合并之后可得：

$$
w_{t+1} = \mathop{\arg\min}_{w} \frac{1}{2}\|w-w_{t}+\eta_tg_t\|_2^2 + \eta_{t+\frac{1}{2}}\lambda\|w\|_1
$$

&emsp; 该式右边我们对$$w$$求导，可以得到：

$$w-w_t+\eta_tg_t + \eta_{t+\frac{1}{2}}\delta(\lambda\|w\|_1)=0$$

&emsp; 有上式可知$$\mid w_t-\eta_tg_t \mid <\eta_{t+\frac{1}{2}}\lambda \rightarrow w_{t+1} = 0$$ 这一动作替代了替代了参数截断的方法，实现了稀疏性。

&emsp; **FTRL**则与FOBOS师出一脉，下面给出FTRL的参数更新方程。

$$
\begin{eqnarray}
w_{t+1} &=& \mathop{\arg\min}_{w} \ (g_{1:t}w+\frac{1}{2}\sum_{s=1}^{t}\sigma_s\|w-w_s\|_2^2 +\lambda_1\|w\|_1)\\
 &=& \mathop{\arg\min}_{w} \ (g_{1:t} - \sum_{s=1}^t\sigma_sw_s)w + \sum_{s=1}^t\sigma_s\|w\|_2^2 + \lambda_1\|w\|_1+(const)
\end{eqnarray}
$$

&emsp; 其中$$g_{1:t} = \sum_{s=1}^tg_s$$且$$g_s$$表示第s组sample对应的梯度，$$\sigma_s$$表示learning rate, 对应的有$$\sigma_{1:t} =\sum_{s=1}^t \sigma_s = \frac{1}{\eta_t}$$
