---
layout: post
title:  "Effective Python"
date:   2016-01-15 10:00:04
categories: liuqianchao update
---


How to modify your Python code style, and use it in a more effective way, this post will introduce the pythonic code style to you.

####Language sugar and suggested code style

- **slice**   
	You may see [::-1] in python, you also may see [begin:end], which is a brief edition of [begin:end:step]. [begin:end] is suggested, and [begin:end:step] often causes unexpected bugs, for example, it will break for Unicode characters

- **str.format()**    
	In C++ ect., %s is used to express a string, it works for python, but the suggested way to express string in python is:
	{% highlight python %}
	value={'greet':'Hello world','language':'Python'}
	print '%(greet)s from %(language)s.' % value
	#or
	print '{greet} from {language}.'.format(greet='Hello world',language='Python')
	{% endhighlight %} 

- **join**   
	{% highlight python %}
	''.join(['str1','str2']) #instead of using str1+str2.
	{% endhighlight %} 

- **with**   
	To close a file immediately, use 'with' to operate the file.
	{% highlight python %}
 	with open('files/tex.txt') as f:
 		f.write('first line')
	{% endhighlight %} 

- **zip**
	The zip() built-in function can be used to iterate over multiple iterators in parallel. And in Python 2, zip returns the full result as a list of tuples.
	{% highlight python %}
	x=[1,2,3]
	y=[4,5,6,7]
	zip(x,y) #[(1,4),(2,5),(3,6)]
	{% endhighlight %} 	

- **assert**   
	use assert expression1,expression2 to capture the constraint defined by the user. 

- **deepcopy**   
	Distinguish shallow copy from deep copy and '='
	{% highlight python %}
	#= reference
	import copy
	copy.copy(object)#shallow copy, one time of coping
	copy.deepcopy(object)#deep copy, copy objects iteratively
	{% endhighlight %} 	

- **map & filter**   
	use `map( func, seq1[, seq2...] )` to call a function and return a list of the results, which is equal to `[f(x) for x in iterable]`.
	{% highlight python %}
	#first situation: map(func,iterable) where func only has one para
	#second situation: map(func,iterable1,iterable2) where func(para1,para2) and first elem of iteable1 and iterable2 are used as para1 and para2.
	#if the func is none, map() is equal to zip()
	#all constraint to: length of return is equal to length of iterable
	{% endhighlight %} 	


- **List comprehensions**   
	When programming, frequently we want to transform one type of data into another. With the help of list comprehensions we can do it more effectively.
	{% highlight python %}
	nums = [0, 1, 2, 3, 4]
	even_squares = [x ** 2 for x in nums if x % 2 == 0]
	print even_squares  # Prints "[0, 4, 16]"

	# filter
	x = filter(lambda x:x>1,range(-10,10))
	>>>[2, 3, 4, 5, 6, 7, 8, 9]
	{% endhighlight %} 	

- **Concurrency & Parallelism**   
	To make full use of the CPU resource of computer. Concurrent(Thread) program may run thousands of separate paths of execution simultaneously. In contrast, the time parallelism(Process) takes to do the total work is cut in half.   
	In python, The existence of GIL(Global Interpreter Lock) means your program could utilize only one thread at the same time.   
	**I/O-bound**:Use `Threading`(false parallelism) for blocking I/O, which may take more time to execute the CPU-bound program. Also use `Lock` class in the `Threading` to avoid data races which may not be avoid by GIL.
	**CPU-bound**:Use `Multiprocessing`   

- **\*args and \*\*kwargs**   
	In `def(arg, *args, **kwargs)`, `*args` means all the default value of arguments, and `**kwargs` means all the default key-value arguments. Inside of a function, we use args and kwargs(without *) to call the value passed to this function.

- **docstring**   
	Use docstring(Triple double-quoted strings) to describe your function and class. 
	{% highlight python %}
	def load_data(f, arg):
	    """
	    :param f:first argument
	    :param arg: another argument
	    :return:{item_key(header of excel/csv):item_value}
    	    """
	{% endhighlight %} 

- **Generators**   
	Generator is a kind of Iterator(which has next or \_\_next\_\_ def), However, during the iteration, the result of return will be created when they are called instead of storing them all in memory.   
	{% highlight python %}
	def generator_func():
		for item in range(10):
			yield item
	# fist way to call
	gen = generator_func()
	next(gen) # for iterable such as str, use iter() instead of next() for itertor
	# second way to call
	for item in generator_func():
		print item
	{% endhighlight %} 

#### Reference
1. Brett Slatkin *Effective Python*.
2. Y.Zhang, Yh.Lai *Writing Solid Python Code*.

