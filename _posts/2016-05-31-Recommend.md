---
layout: post
title:  "Recommender System"
date:   2016-04-31 10:00:04
categories: liuqianchao update
---

&nbsp;&nbsp;&nbsp;&nbsp;近些年来，提到推荐系统，我们更多的是指个性化推荐系统，而个性化推荐系统是较大众化推荐而言的。   
&nbsp;&nbsp;&nbsp;&nbsp;大众化推荐经常被用于检索推荐，比如在京东网购时，按照商品“综合排序”展示检索结果，就属于大众化推荐，当然大众化推荐也经常与个性化推荐搭配使用。本文将对个性化推荐展开介绍，大众化推荐不属于本文范畴。   
&nbsp;&nbsp;&nbsp;&nbsp;本文根据推荐方法的思想不同，主要介绍了基于人口统计学的推荐等方法。


### 1.基于人口统计学的推荐
&nbsp;&nbsp;&nbsp;&nbsp;基于人口统计学的方法(Demographic-Based Recommendation)的思想源自“相似的用户有着相似的爱好”。该方法通过计算系统中用户的相似度，相似度的依据主要是指性别、年龄、工作等，而不包括对商品的评分、购买记录（该方法特指基于用户的协同过滤方法），对a用户推荐与他相似的b用户喜欢的商品。   
&nbsp;&nbsp;&nbsp;&nbsp;该方法的特点是，推荐思想简单明了，主要的计算来自对用户相似度的计算，且用户的相似度计算方法十分简单，常用的方法包括欧式距离、余弦相似度等。应用场景十分广泛，京东在其“用户画像”技术中，常用的推荐方法就是该方法，除此之外，该方法用多的应用是与其他的推荐方法相结合。    
&nbsp;&nbsp;&nbsp;&nbsp;用户$$X_{1}$$,$$X_{2}$$的欧式距离的计算示例如下:   

<div align="center">$$S(X_{1},X_{2}) = \sum_{i=1}^N (x_{1i}-x_{2i})^2 $$</div>

### 2.基于内容的推荐
&nbsp;&nbsp;&nbsp;&nbsp;和上一个方法十分相似，只不过是假设产生了变化，基于内容的推荐(Content-Based Recommendation)是假设“用户喜欢他之前喜欢的东西“。比如我玩过魔兽，炉石，系统就会为给推荐新出的游戏“守望先锋”。   
&nbsp;&nbsp;&nbsp;&nbsp;基于内容的推荐的计算过程大致可以划分为两类，一类是直接计算商品间的相似性，按照相似性由高到低排序进行推荐；另一类是通过构建用户画像，将用户浏览过、购买过的商品的属性作为用户的画像，这个画像描述了用户对于商品属性的偏好特征，然后计算待推荐商品与用户画像之间的相似度，进行推荐。   
&nbsp;&nbsp;&nbsp;&nbsp;基于内容的推荐方法有一个优点就是能很好的解决冷启动(Cold－Start)的问题。冷启动是指新产品的推出时如何进行推荐的问题，之前的一些方法的推荐往往是通过该商品的历史评分、历史购买记录进行的，基于内容的推荐则解决该问题。继续拿“守望先锋”来举例，我可以通过TF-IDF来提取该游戏商品介绍文本中的关键词，提取到“暴雪”等字段，根据该“属性”就可以推荐给暴雪系游戏的玩家。   
&nbsp;&nbsp;&nbsp;&nbsp;同时需要注意的是，由于要计算商品之间的相似性，所以主要需要需要维护商品到商品属性的矩阵，比如对于一款游戏商品，其商品属性包括游戏类型、出品商等属性字段，因此对于不同类别的的商品需要设计不同的商品属性字段。

### 3.基于协同过滤的推荐
&nbsp;&nbsp;&nbsp;&nbsp;首相我想要介绍一下，什么是协同过滤，维基上给出的解释是: ***collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources, etc***.   
&nbsp;&nbsp;&nbsp;&nbsp;这里有一组关键词，collaboration among multiple agents，借助群体的信息，通过别人的对商品的浏览、购买、评价记录来完成对某一个体的推荐，这就是协同过滤的特点。   
&nbsp;&nbsp;&nbsp;&nbsp;基于协同过滤的推荐(Collaborative Filtering-Based Recommendation)，又可以细分为基于用户的(User-based),基于物品的(Item-based)以及基于模型的(Model-based)。
&nbsp;&nbsp;&nbsp;&nbsp;在介绍各种Collaborative Filtering方法之前，我们先介绍几个计算相似度的方法：

- 欧式距离：两组向量元素级别的差的平方和。
- 余弦距离：内积（点乘）/两个向量的Norm 2范数。相比于欧几里得距离，余弦距离不依赖于绝对值，比如两个人分别对于A,B,C产品的打分为(5,5,0)和(3,3,0)，则余弦距离认为两个人极为相似（相似度为1）。
- 皮尔逊相似度：X，Y分别标准化（减去相应的均值）之后进行余弦距离的计算。皮尔逊相关系数排除了“欧式距离不同维度的数量级对结果影响显著"的问题。

#### 3.1 基于用户的协同过滤   
&nbsp;&nbsp;&nbsp;&nbsp;**基于用户的协同过滤**，与上文中提到的基于人口统计的推荐不同之处的计算用户相似时用的是用户对于商品的评分或者购买浏览记录，而不是人口统计特征。基于用户的协同过滤维护这样一个$$m × k$$的矩阵$$M$$.其中有$$m$$个用户和$$k$$个商品,通过计算行与行之间（用户之间）的相似度，来推荐和你相似的人浏览过而你没有浏览过的内容。

<div id="content">

    <table cellspacing="0">
    <tr><th></th><td>item1</td><td>item2</td><td>item3</td></tr>
    <tr><td>user1</td><td>3</td><td>3</td><td></td></tr>
    <tr><td>user2</td><td>1</td><td>1</td><td></td></tr>
    <tr><td>user3</td><td></td><td></td><td>5</td></tr>

    </table>

</div>
<div align="center">基于用户的协同过滤</div>
&nbsp;&nbsp;&nbsp;&nbsp;具体操作步骤如下：    

- 计算用户之间的相似度，得到用户相似度矩阵（是一个对称矩阵）。
- 补全用户x未打分商品i的得分：W(x与y用户的相似度)*S(y用户对商品i的打分)。其中y是所有对商品i打过分的用户
- 由高到低向用户x推荐其未打分的商品。

#### 3.2 基于物品的协同过滤   

&nbsp;&nbsp;&nbsp;&nbsp;**基于物品的协同过滤**，与上文中提到的基于内容的推荐比较相似，不同之处是相似物品的的计算不是通过商品的属性，而是通过网络中用户对商品的历史浏览记录。   
&nbsp;&nbsp;&nbsp;&nbsp;如果用户同时浏览过item1和item2，则下表中(1,2)和(2,1)处的值为3，这样可以表征商品之间的关联相似性，这样，一些经常被同时购买的商品会被推荐出来。上述矩阵是通过$$M^{T}M$$的到的，$$M$$是基于用户的协同过滤维护的矩阵。


<div id="content">

    <table cellspacing="0">
    <tr><th></th><td>user1</td><td>user2</td><td>user3</td></tr>
    <tr><td>item1</td><td>3</td><td>1</td><td></td></tr>
    <tr><td>item2</td><td>3</td><td>1</td><td></td></tr>
    <tr><td>item3</td><td></td><td></td><td>5</td></tr>

    </table>

</div>
<div align="center">基于物品的协同过滤</div>

&nbsp;&nbsp;&nbsp;&nbsp; 基于物品的协同过滤相较于基于用户的具有以下优势，因此目前在亚马逊，被采用的是基于物品的协同过滤方法。

- 相较于用户，物品之间的相似度计算变化相对来说较小，该相似度计算过程可以离线计算、定期更新。


#### 3.3 基于模型的协同过滤   

&nbsp;&nbsp;&nbsp;&nbsp;无论是User-Based还是Item-Based的方法，都需要进行人与人或物与物之间的相似性，这部分的计算量很大，难以实现实时在线响应，而**基于模型的推荐**(Model-Based)能够很好地解决这个问题，该方法事先根据历史信息“训练”好模型，使用该模型进行推荐。   
&nbsp;&nbsp;&nbsp;&nbsp;基于模型的协同过滤方法常见的技术手段包括语义分析(Latent Semantic Analysis),贝叶斯网络(Bayesian Networkds)以及矩阵分解(Matrix Factorization)，下面将以矩阵分解的方法为例，讲解模型的构建和训练。   
&nbsp;&nbsp;&nbsp;&nbsp;首先这里要阐述一下矩阵分解的目的，这里，矩阵分解(svd等方法)，并不是为了降维，而是把user-item矩阵分解为user-factor矩阵和item-factor矩阵相乘的形式，下面介绍矩阵分解技术基本流程。   
<br>
&nbsp;&nbsp;&nbsp;&nbsp;1、假设已有稀疏矩阵$$X \in R^{m \times n}$$表示用户对商品的打分（浏览、购买记录），该矩阵同“基于用户的协同过滤”中的矩阵，设定$$U \in R^{m \times r}$$和$$V \in R^{n \times r}$$是矩阵$$X$$的低秩分解。   
&nbsp;&nbsp;&nbsp;&nbsp;2、建立优化问题的目标函数：

<div align="center">$$argmin [D_{w}(X,f(UV^{T}))+R(U,V)]$$</div>

&nbsp;&nbsp;&nbsp;&nbsp;其中，$$D_{w}(X,f(UV^{T}))$$是$$U$$,$$V$$对于$$X$$的损失函数，比如可以定义为平方损失函数。$$R(U,V)$$是正则化因子(regularizaiton loss),用于避免过度拟合，该正则项可以取L2正则函数。   

&nbsp;&nbsp;&nbsp;&nbsp;3、到这里，已经把问题转换为最优值求解问题，常见的方法包括梯度下降等。

&nbsp;&nbsp;&nbsp;&nbsp;在完成上述的矩阵分解后，便可以通过$$UV^{T}$$来补全用户对未评价商品的得分。剩余的思路与其他协调过滤的方法一致，只需要将补全的未评价商品得分由高到低排序，推荐给用户即可。

&nbsp;&nbsp;&nbsp;&nbsp;**下面我们具体化损失函数，并使用Python来实现矩阵分解的主要步骤：**

将原始评分R矩阵分解成P矩阵和Q矩阵

$$R_{m\times n} = P_{m\times k}Q_{k \times n}$$

故如果使用平方差作为损失函数，有：

$$e_{i,j}^2 = (r_{ij}-\hat{r_{ij}})^2= (r_{ij}-\sum_{k=1}^Kp_{ik}q_{kj})^2$$

求损失函数的梯度，故：

$$\frac{\delta e_{ij}^2}{\delta p_{ik}}=-2(r_{ij}-\sum_{k=1}^Kp_{ik}q_{kj})q_{kj} = -2e_{ij}q_{kj}$$

一般会对目标函数加入正则项，防止过度拟合：

$$e_{i,j}^2 = (r_{ij}-\sum_{k=1}^Kp_{ik}q_{kj})^2 + \frac{\beta}{2}\sum_{i=1}^k(p_{ik}^2+q_{kj}^2)$$

相应地求其损失函数的梯度：

$$\frac{\delta e_{ij}^2}{\delta p_{ik}}=-2e_{ij}q_{kj}+\beta p_{ik}$$

因此我们有更新方程为：

$$p_{ik}' = p_{ik} - \alpha \frac{\delta e_{ij}^2}{\delta p_{ik}} = p_{ik} + \alpha(2e_{ij}q_{kj}-\beta p_{ik})$$


{% highlight python %}
def mf(originMatrix, k, alpha, beta, lossThreshold, maxIter):
	m, n = np.shape(originMatrix)
	# 初始化p和q
	p = np.mat(np.random.random((m, k)))
	q = np.mat(np.random.random((k, n)))

	# 开始训练
	for step in range(maxIter):
		for i in range(m):
			for j in range(n):
				if originMatrix[i, j] > 0:
					# 计算e_ij
					error = originMatrix[i, j]
					for r in range(k):
						error = error - p[i, r] * q[r, j]
					# 更新
					for r in range(k):
						p[i, r] = p[i, r] + alpha*(2*error*q[r, j] - beta*p[i, r])
						q[r, j] = q[r, j] + alpha*(2*error*p[i, r] - beta*q[r, j])
		loss = 0.0
		for i in range(m):
			for j in range(n):
				if originMatrix[i, j] > 0:
					error = 0
					for r in range(k):
						error = error + p[i, r] * q[r, j]
					loss = (originMatrix[i, j] - error)**2
					for r in range(k):
						loss += 0.5*beta*(p[i, r]**2 + q[r, j]**2)
		if loss < lossThreshold:
			break
		if step % 100 == 0:
			print step, loss
	return p, q
{% endhighlight %}


#### Reference
1. Zhiyuan Liu [Big Data Intelligence](http://www.amazon.com/s/ref=nb_sb_noss?url=search-alias%3Daps&field-keywords=big+data+intelligence&rh=i%3Aaps%2Ck%3Abig+data+intelligence)
