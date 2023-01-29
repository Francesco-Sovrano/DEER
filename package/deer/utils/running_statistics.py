# -*- coding: utf-8 -*-
from runstats import Statistics, Regression

class RunningPercentile:
	def __init__(self, percentile=0.5, step=0.1):
		self.step = step
		self.step_up = 1.0 - percentile
		self.step_down = percentile
		self.x = None

	def push(self, observation):
		if self.x is None:
			self.x = observation
			return

		if self.x > observation:
			self.x -= self.step * self.step_up
		elif self.x < observation:
			self.x += self.step * self.step_down
		if abs(observation - self.x) < self.step:
			self.step /= 2.0

class RunningStats(object):
	
	def __init__(self, window_size=None):
		if window_size:
			assert window_size > 1, 'window_size must be greater than 1'
		self.window_size = window_size
		self.running_stats = Statistics()
		
	def push(self, x):
		if self.window_size and len(self.running_stats) >= self.window_size:
			self.running_stats *= (self.window_size-1)/self.window_size
		self.running_stats.push(x)

	@property
	def mean(self):
		return self.running_stats.mean() if len(self.running_stats) > 0 else 0.
	
	@property
	def std(self):
		return self.running_stats.stddev() if len(self.running_stats) > 1 else 0.

	@property
	def var(self):
		return self.running_stats.variance() if len(self.running_stats) > 1 else 0.
	
# from collections import deque
# import random 
# import numpy as np

# b1 = RunningStats(window_size=2**8)
# b2 = deque(maxlen=2**8)

# k=2**10
# i=0
# while True:
# 	i = i+1
# 	if not i%2**10:
# 		k-=10
# 	n = random.randint(0,max(10,k))
# 	b1.push(n)
# 	b2.append(n)
# 	print(b1.mean, np.mean(b2))