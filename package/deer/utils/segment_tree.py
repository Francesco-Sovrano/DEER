import numpy as np
import random

def is_tuple(val):
	return type(val) in [list,tuple]

class SegmentTree(object):
	__slots__ = ('_capacity','_value')
	
	def __init__(self, capacity, neutral_element):
		"""Build a Segment Tree data structure.
		https://en.wikipedia.org/wiki/Segment_tree
		Can be used as regular array, but with two
		important differences:
			a) setting item's value is slightly slower.
			   It is O(log capacity) instead of O(1).
			b) user has access to an efficient ( O(log segment size) )
			   `reduce` operation which reduces `operation` over
			   a contiguous subsequence of items in the array.
		Paramters
		---------
		capacity: int
			Total size of the array - must be a power of two.
		operation: lambda obj, obj -> obj
			and operation for combining elements (eg. sum, max)
			must form a mathematical group together with the set of
			possible values for array elements (i.e. be associative)
		neutral_element: obj
			neutral element for the operation above. eg. float('-inf')
			for max and 0 for sum.
		"""
		assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
		self._capacity = capacity
		self._value = [neutral_element for _ in range(2 * capacity)]
		self._neutral_element = neutral_element
		self.inserted_elements = 0

	def resize(self, new_capacity):
		if new_capacity == self._capacity:
			return
		assert new_capacity > 0 and new_capacity & (new_capacity - 1) == 0, "new capacity must be positive and a power of 2."
		# assert self.inserted_elements <= new_capacity, "cannot resize because new_capacity is lower than inserted_elements"
		old_value = self._value
		old_capacity = self._capacity
		self._value = [neutral_element for _ in range(2 * new_capacity)]
		self._capacity = new_capacity
		self.inserted_elements = 0
		for i,v in enumerate(old_value[:min(old_capacity,new_capacity)]):
			self[i] = v

	def reduce(self, start=0, end=None):
		"""Applies `self._operation` to subsequence of our values.
		Subsequence is contiguous, includes `start` and excludes `end`.
		  self._operation(
			  arr[start], operation(arr[start+1], operation(... arr[end])))
		Args:
			start (int): Start index to apply reduction to.
			end (Optional[int]): End index to apply reduction to (excluded).
		Returns:
			any: The result of reducing self._operation over the specified
				range of `self._value` elements.
		"""
		if end is None:
			end = self.inserted_elements
		elif end < 0:
			end += self.inserted_elements

		# Init result with neutral element.
		result = self._neutral_element
		# Map start/end to our actual index space (second half of array).
		start += self._capacity
		end += self._capacity

		# Example:
		# internal-array (first half=sums, second half=actual values):
		# 0 1 2 3 | 4 5 6 7
		# - 6 1 5 | 1 0 2 3

		# tree.sum(0, 3) = 3
		# internally: start=4, end=7 -> sum values 1 0 2 = 3.

		# Iterate over tree starting in the actual-values (second half)
		# section.
		# 1) start=4 is even -> do nothing.
		# 2) end=7 is odd -> end-- -> end=6 -> add value to result: result=2
		# 3) int-divide start and end by 2: start=2, end=3
		# 4) start still smaller end -> iterate once more.
		# 5) start=2 is even -> do nothing.
		# 6) end=3 is odd -> end-- -> end=2 -> add value to result: result=1
		#	NOTE: This adds the sum of indices 4 and 5 to the result.

		# Iterate as long as start != end.
		while start < end:

			# If start is odd: Add its value to result and move start to
			# next even value.
			if start & 1:
				result = self._operation(result, self._value[start])
				start += 1

			# If end is odd: Move end to previous even value, then add its
			# value to result. NOTE: This takes care of excluding `end` in any
			# situation.
			if end & 1:
				end -= 1
				result = self._operation(result, self._value[end])

			# Divide both start and end by 2 to make them "jump" into the
			# next upper level reduce-index space.
			start //= 2
			end //= 2

			# Then repeat till start == end.

		return result

	def __setitem__(self, idx, val):
		"""
		Inserts/overwrites a value in/into the tree.
		Args:
			idx (int): The index to insert to. Must be in [0, `self._capacity`[
			val (float): The value to insert.
		"""
		assert 0 <= idx < self._capacity

		# Index of the leaf to insert into (always insert in "second half"
		# of the tree, the first half is reserved for already calculated
		# reduction-values).
		idx += self._capacity
		if self._value[idx] == self._neutral_element:
			if val is None:
				return
			self.inserted_elements += 1
		elif val is None:
			self.inserted_elements -= 1
		self._value[idx] = val if val is not None else self._neutral_element

		# Recalculate all affected reduction values (in "first half" of tree).
		idx = idx >> 1  # Divide by 2 (faster than division).
		while idx >= 1:
			update_idx = 2 * idx  # calculate only once
			# Update the reduction value at the correct "first half" idx.
			self._value[idx] = self._operation(self._value[update_idx], self._value[update_idx + 1])
			idx = idx >> 1  # Divide by 2 (faster than division).

	def __getitem__(self, idx):
		assert 0 <= idx < self._capacity
		return self._value[idx + self._capacity]


class SumSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=0., with_min_tree=True, with_max_tree=False):
		super(SumSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		self.min_tree = MinSegmentTree(capacity, neutral_element=(float('inf'),-1)) if with_min_tree else None
		self.max_tree = MaxSegmentTree(capacity, neutral_element=(float('-inf'),-1)) if with_max_tree else None

	def resize(self, new_capacity):
		super().resize(new_capacity)
		if self.min_tree:
			self.min_tree.resize(new_capacity)
		if self.max_tree:
			self.max_tree.resize(new_capacity)
	
	@staticmethod
	def _operation(a, b):
		return a+b

	def __setitem__(self, idx, val): # O(log)
		super().__setitem__(idx, val)
		if self.min_tree:
			self.min_tree[idx] = (val,idx) if val is not None else None
		if self.max_tree:
			self.max_tree[idx] = (val,idx) if val is not None else None

	def sum(self, start=0, end=None): # O(log)
		"""Returns arr[start] + ... + arr[end]"""
		return super(SumSegmentTree, self).reduce(start, end)

	def find_prefixsum_idx(self, prefixsum_fn, check_min=True): # O(log)
		"""Find the highest index `i` in the array such that
			sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
		if array values are probabilities, this function
		allows to sample indexes according to the discrete
		probability efficiently.
		Parameters
		----------
		perfixsum: float
			upperbound on the sum of array prefix
		Returns
		-------
		idx: int
			highest index satisfying the prefixsum constraint
		"""
		if self.inserted_elements == 0:
			return None
		if self.inserted_elements == 1:
			return 0
		if self.min_tree and check_min:
			min_p = self.min_tree.min()[0] # O(log)
			scaled_prefix = min_p < 0
		else:
			min_p = 0
			scaled_prefix = False
		mass = self.sum() # O(log)
		if scaled_prefix: # Use it in case of negative elements in the sumtree, they would break the tree invariant
			mass -= min_p*self.inserted_elements # Scale mass by min priority
			if mass <= 0: # This may happen due to numerical errors, when all elements have approx. the same (negative) value
				return random.randrange(0,self.inserted_elements) # Return any element randomly, they all have the same priority.
			minimum = min(self._neutral_element, min_p)
			summed_elements = self._capacity
		prefixsum = prefixsum_fn(mass)
		# prefixsum = np.clip(prefixsum, 0, mass)
		# print(prefixsum,mass)
		assert 0 <= prefixsum <= mass + 1e-5, f"prefixsum is {prefixsum} and mass is {mass} for {self.inserted_elements} inserted elements with minimum priority equal to {min_p}, but prefixsum must be: 0 <= prefixsum <= mass + 1e-5"
		idx = 1
		# While non-leaf (first half of tree).
		while idx < self._capacity:
			update_idx = 2 * idx
			value = self._value[update_idx]
			if scaled_prefix:
				summed_elements /= 2
				value -= minimum*summed_elements
			if value > prefixsum:
				idx = update_idx
			else:
				prefixsum -= value
				idx = update_idx + 1
		idx -= self._capacity
		if idx == self.inserted_elements:
			idx -= 1
		assert 0 <= idx < self.inserted_elements, f"{idx} has to be lower than {self.inserted_elements} and greater than 0"
		return idx
	
class MinSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=float('inf')):
		super(MinSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		
	@staticmethod
	def _operation(a, b):
		return a if a < b else b

	def min(self, start=0, end=None): # O(log)
		"""Returns min(arr[start], ...,  arr[end])"""
		return super(MinSegmentTree, self).reduce(start, end)

class MaxSegmentTree(SegmentTree):
	def __init__(self, capacity, neutral_element=float('-inf')):
		super(MaxSegmentTree, self).__init__(
			capacity=capacity,
			neutral_element=neutral_element
		)
		
	@staticmethod
	def _operation(a, b):
		return a if a > b else b

	def max(self, start=0, end=None): # O(log)
		"""Returns min(arr[start], ...,  arr[end])"""
		return super(MaxSegmentTree, self).reduce(start, end)

if __name__ == "__main__":
	print('Test 1')
	test = SumSegmentTree(4)
	test[2] = -10.0001
	test[3] = -10.0001
	test[0] = -10.0001
	test[1] = -10.0001
	print('unscaled', test.sum())
	print('scaled', test.sum()-test.min_tree.min()[0]*test.inserted_elements)
	i = test.find_prefixsum_idx(lambda x:x*random.random())
	print(i,test[i] )

	print('Test 2')
	test = SumSegmentTree(4)
	test[2] = -10
	test[3] = -5
	test[0] = 1
	test[1] = 2
	print('unscaled', test.sum())
	print('scaled', test.sum()-test.min_tree.min()[0]*test.inserted_elements)
	i = test.find_prefixsum_idx(lambda x:23)
	print(i,test[i] )



