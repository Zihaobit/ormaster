---
layout: post
title:  "Theano & Tensorflow"
date:   2016-02-02 10:00:04
categories: liuqianchao update
---

&nbsp;&nbsp;&nbsp;&nbsp;市面上的深度学习框架不断发布，包括：ConvNet,Caffe(图像),Torch以及Tensorflow，其中最引入瞩目的莫过于来自Google的Tensorflow，在这篇文章中，将对提供Python API的Tensorflow以及Theano作简要介绍。

### 1.背景   
&nbsp;&nbsp;&nbsp;&nbsp;Theano最初是被设计成一套符号表达系统，Tensorflow类似于Theano，也属于符号编程框架	（微软开源的CNTK也是如此）。

&nbsp;&nbsp;&nbsp;&nbsp;选择合适的层数，每层的神经元数量，激活函数，损失函数，正则化的参数，然后使用validation数据来判定这次训练的效果。

### 2.简介   
&nbsp;&nbsp;&nbsp;&nbsp;下面将以Theano为例，通过官方给出的Tutorial介绍其基本框架:   

#### 2.1 Theano   

#### 2.1.1 代数基础   
&nbsp;&nbsp;&nbsp;&nbsp;在Theano中所有的数据对象都必须是Theano类型([Theano Type](http://deeplearning.net/software/theano/extending/graphstructures.html#type))的,这里在代数运算中引入**标量(scalar)**的概念,将每一个标量数据通过theano.tensor定义成与python存储数据类型（int,float,double等）相一致的Theano类型(iscalar,fscalar,dscalar等)，且这种对应是一对一(即Python存在的数据类型，Theano中均有实现)。   

&nbsp;&nbsp;&nbsp;&nbsp;除了标量(scalar)之外，Theano中还包含**vector**,**matrix**，在定义这两种类型的Theano变量时，使用相应的dvector(vector contains double elements),dmatrix即可。   

[**theano.function**方法](http://deeplearning.net/software/theano/library/compile/function.html)，第一个参数是函数的输入参数列表，第二个参数是返回值的列表，且当列表元素数目为1时，可以省去[].
{% highlight python %}
import numpy
import theano.tensor as T
from theano import function
from theano import In
x,y = T.dscalar('x','y')
z = x + y
f = function([x,y],z)
#using In to set default value: set default value 5 for y, taking place of y by In(y, default=5) 
#shared value, to define a value that can be 

#Shared变量
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
#UPDATES: An expression which indicates updates to the Value after each function call, also means that expressions for new SharedVariable values
#UPDATES: (iterable over pairs (shared_variable, new_expression). List, tuple or dict.) 
{% endhighlight %} 
这里要提一下，在代数运算系统中的一些tips：   
1. **缺省值**   
&nbsp;&nbsp;&nbsp;&nbsp;首先就是给function中的变量定义缺省值,通过使用`In(y,default=5)`代替原来的变量`y`.   
2. **Shared变量**   
&nbsp;&nbsp;&nbsp;&nbsp;当你需要在不同的函数中持续使用某一个变量时，一般会定义一个shared变量，通过function中的updates参数来改变shared变量的值，在Theano中使用updates的主要目的是提高运算速度（在GPU上），Theano对其进行了特别的优化。同时，还可以通过使用function的`givens=[(var1,var2)]`参数(var1的值被var2替换)，来使用state而不改变state.value.   
&nbsp;&nbsp;&nbsp;&nbsp;引入shared变量的原因：在进行大量求导运算时(GPU擅长)，需要把gradients数据从GPU传输到CPU，通过shared变量，可以省去这个步骤(计算(GPU)与更新(CPU)可以放在一起进行)，一般在训练网络的过程中会将weights定义为shared variable(that persist in the graph between calls).

3. **随机数**   
{% highlight python %}
>>> from theano.tensor.shared_randomstreams import RandomStreams
>>> from theano import function
>>> srng = RandomStreams(seed=234) #where rng means that random number generator
>>> rv_u = srng.uniform((2,2))
>>> rv_n = srng.normal((2,2))
>>> f = function([], rv_u)
>>> f()
array([[ 0.70574274,  0.80222456],
       [ 0.25976164,  0.18285402]])
>>> g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
>>> g()
array([[ 0.37328447, -0.65746672],
       [-0.36302373, -0.97484625]])
>>> nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;在给随机数设置seed时，一种方法是直接给RandomStreams设置（如上例所示，或srng.seed(234)），另一种方法是对RandomStreams的某一随机分布变量设置seed，比如对rv_u设置seed,需要通过以下的方法,即使用rng.ser_value和rng.get_value()：
{% highlight python %}
>>> rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
>>> rng_val.seed(89234)                         # seeds the generator
>>> rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng
{% endhighlight %} 
&nbsp;&nbsp;&nbsp;&nbsp;这里要注意到RandomStream仅工作在CPU环境下（MRG31k3p工作在CPU和GPU下，CURAND仅工作在GPU下）。


#### 2.1.2 导数   
&nbsp;&nbsp;&nbsp;&nbsp;在Theano中，使用`T.grad(y,x)`来计算表达式y（代价函数）关于x（自变量）的导数。   
&nbsp;&nbsp;&nbsp;&nbsp;此外可以使用`theano.gradient.jacobian()`来计算雅可比矩阵（多元函数的一阶偏导数矩阵），使用`theano.gradient.hessian()`来计算海森矩阵（二阶偏导数矩阵）。

#### 2.1.3 条件   
&nbsp;&nbsp;&nbsp;&nbsp;由于Theano是一种类似于函数式编程的语言，在使用中，Python的if语句只在编译时起作用，编译时会将if判断后的结果进行编译，所以这里需要单独引入条件函数IfElse和Switch。   
&nbsp;&nbsp;&nbsp;&nbsp;`theano.ifelse(cond, ift, iff)`有三个参数，一个boolean类型的表达式和两个返回变量，`tensor.switch(cond, ift, iff)`则为一个tensor和两个返回变量。
{% highlight Python %}

from theano import tensor as T
from theano.ifelse import ifelse
import theano, time, numpy

a,b = T.scalars('a', 'b')
x,y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a, b, x, y], z_switch,
                           mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a, b, x, y], z_lazy,
                               mode=theano.Mode(linker='vm'))
{% endhighlight %} 

#### 2.1.4 scan   
&nbsp;&nbsp;&nbsp;&nbsp;设计scan的目的是为了实现循环（递归）地影响一个对象，其主要有四个参数，fn为每次迭代要进行的操作，是一个函数；sequences(y,p)为迭代序列（for(i in range(10)) 中的range(10)）,其中y为要迭代的次数(如果sequences=None，要通过n_steps参数来指定迭代次数)；outputs_info描述使用到前几次迭代结果作为lambda的参数，non_sequences是非序列化的输入，通常用来存储固定的指定值。(good for RNNs)

{% highlight Python %}
results, updates = theano.scan(fn=lambda y, p, x_tm2, x_tm1, A: y+p+x_tm2+x_tm1+A,
                    sequences=[Y, P[::-1]],
                    outputs_info=[dict(initial=X, taps=[-2, -1])],
                    non_sequences=A)
{% endhighlight %} 

#### 2.1.5 稀疏   
&nbsp;&nbsp;&nbsp;&nbsp;Theano专门对稀疏矩阵制定了处理策略，稀疏矩阵中，只有非0元素才会被存储。这里，稀疏矩阵的存储格式有两种csc和csr，当矩阵的行比较多时，建议使用csc:基于矩阵列的存储格式；当行数较少时，则应选择csr:基于矩阵列的存储格式，两者的区别仅在于存储数据的位置。

{% highlight Python %}
from theano import sparse
x = sparse.csc_matrix(name='x', dtype='float32')  
y = sparse.dense_from_sparse(x) #将稀疏矩阵转换为密集矩阵
data, indices, indptr, shape = sparse.csm_properties(x)  #使用sparse.csm_properties来返回一个tensor变量的元组，来表示稀疏矩阵的内部特征。
#data 属性是一个一维的 ndarray ，它包含稀疏矩阵所有的非0元素。
#indices 和indptr 属性是用来存储稀疏矩阵中数据的位置的。
#shape 属性，准确的说是和密集矩阵的shape属性一样的。如果从前三个属性上没法推断，那么它可以在稀疏矩阵创建的时候显式指定 。

{% endhighlight %} 


#### 2.1.6 Other key words:   
**theano.tensor.ones_like(x)**   
Parameters:	x – tensor that has same shape as output   
Returns a tensor filled with 1s that has same shape as x.

**theano.tensor.dot(x,y)**   
tensor 变量的点乘操作

**theano.tensor.nnet.softmax(x)** or **theano.tensor.nnet.sigmoid(x)** or **theano.tensor.nnet.relu()**      
Returns the standard sigmoid nonlinearity applied to x   
Returns the softmax function of x:   
Compute the element-wise rectified linear activation function(激活函数为Rectified Linear Units,Relu易于求导便于反向传播求误差梯度，由于会使一部分神经元输出为0，从而造成网络的稀疏性，减少了参数的相互依存关系).

此外使用Lasagne、Keras等建立在theano基础上的library可以作为theano的部分或全部替代,此外这些包支持Pretrained model。


#### 2.2 Tensorflow   
&nbsp;&nbsp;&nbsp;&nbsp;Tensorflow和theano非常相似，它的一些亮点包括：可视化(TensorBoard),multi-GPU and multi-node training.

placeholder来代替tensor
Variables来代替shared variables

#### Reference
1. LISA lab [Theano 0.7 documentation](http://www.deeplearning.net/tutorial/index.html)
2. 赵孽 [如何评价 Theano？](https://www.zhihu.com/question/35485591)
3. 松尾丰 [人工智能狂潮](http://book.douban.com/subject/26698202/)
4. Kenneth Tran [Evaluation of Deep Learning Toolkits](https://github.com/zer0n/deepframeworks/blob/master/README.Kenneth Tran md?utm_source=tuicool&utm_medium=referral)
5. Stanford/vision [CS231n Winter 2016: Convolutional Neural Networks for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/index.html)(Lecture 12)

