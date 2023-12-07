# -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import time
from deer.experience_buffers.buffer.buffer import Buffer
from deer.utils.segment_tree import SumSegmentTree, MinSegmentTree, MaxSegmentTree
import copy
import uuid
from deer.utils.running_statistics import RunningStats
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.policy.sample_batch import SampleBatch

logger = logging.getLogger(__name__)

discard_batch = lambda x: all(map(lambda y: y.get('discard',False), x[SampleBatch.INFOS]))
get_batch_infos = lambda x: x[SampleBatch.INFOS][0]
get_batch_indexes = lambda x: get_batch_infos(x)['batch_index']
get_batch_uid = lambda x: get_batch_infos(x)['batch_uid']
get_training_step = lambda x: get_batch_infos(x)['training_step']

class PseudoPrioritizedBuffer(Buffer):
	
	def __init__(self, 
		priority_id,
		priority_aggregation_fn,
		cluster_size=None, 
		global_size=50000, 
		prioritization_alpha=0.6, 
		prioritization_importance_beta=0.4, 
		prioritization_importance_eta=1e-2,
		prioritization_epsilon=1e-6,
		prioritized_drop_probability=0, 
		global_distribution_matching=False,
		stationarity_window_size=None, 
		stationarity_smoothing_factor=1,
		cluster_prioritisation_strategy='highest',
		cluster_prioritization_alpha=1,
		cluster_level_weighting=True,
		clustering_xi=1, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# clip_cluster_priority_by_max_capacity=False,
		priority_lower_limit=None,
		max_age_window=None,
		seed=None,
		**args
	): # O(1)
		assert not prioritization_importance_beta or prioritization_importance_beta > 0., f"prioritization_importance_beta must be > 0, but it is {prioritization_importance_beta}"
		assert not prioritization_importance_eta or prioritization_importance_eta > 0, f"prioritization_importance_eta must be > 0, but it is {prioritization_importance_eta}"
		assert clustering_xi >= 1, f"clustering_xi must be >= 1, but it is {clustering_xi}"
		if stationarity_window_size:
			assert stationarity_smoothing_factor >= 1, "stationarity_smoothing_factor must be >= 1"
		self._priority_id = priority_id
		self._priority_lower_limit = priority_lower_limit
		self._priority_can_be_negative = priority_lower_limit is None or priority_lower_limit < 0
		self._priority_aggregation_fn = eval(priority_aggregation_fn) if self._priority_can_be_negative else (lambda x: eval(priority_aggregation_fn)(np.abs(x)))
		self._prioritization_alpha = prioritization_alpha # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._prioritization_importance_beta = prioritization_importance_beta # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		self._prioritization_importance_eta = prioritization_importance_eta # Eta is a value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		self._prioritization_epsilon = prioritization_epsilon # prioritization_epsilon to add to the priorities when updating priorities.
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		self._global_distribution_matching = global_distribution_matching
		self._stationarity_window_size = stationarity_window_size
		self._stationarity_smoothing_factor = stationarity_smoothing_factor
		self._cluster_prioritisation_strategy = cluster_prioritisation_strategy
		self._cluster_prioritization_alpha = cluster_prioritization_alpha
		self._cluster_level_weighting = cluster_level_weighting
		self._clustering_xi = clustering_xi
		# self._clip_cluster_priority_by_max_capacity = clip_cluster_priority_by_max_capacity
		self._weight_importance_by_update_time = self._max_age_window = max_age_window
		logger.warning(f'Building new buffer with: prioritized_drop_probability={prioritized_drop_probability}, global_distribution_matching={global_distribution_matching}, stationarity_window_size={stationarity_window_size}, stationarity_smoothing_factor={stationarity_smoothing_factor}')
		super().__init__(cluster_size=cluster_size, global_size=global_size, seed=seed)
		self._it_capacity = 1
		while self._it_capacity < self.cluster_size:
			self._it_capacity *= 2
		# self.priority_stats = RunningStats(window_size=self.global_size)
		self._base_time = time.time()
		self.min_cluster_size = 1
		self.max_cluster_size = self.cluster_size
		self.__historical_min_priority = float('inf')

	def is_weighting_expected_values(self):
		return self._prioritization_importance_beta
		
	def set(self, buffer): # O(1)
		assert isinstance(buffer, PseudoPrioritizedBuffer)
		super().set(buffer)
	
	def clean(self): # O(1)
		super().clean()
		self._sample_priority_tree = []
		if self._prioritized_drop_probability > 0:
			self._drop_priority_tree = []
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree = []
		if self._weight_importance_by_update_time:
			self._update_times = []
			
	def _add_type_if_not_exist(self, type_id): # O(1)
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = type_ = len(self.type_keys)
		self.type_values.append(type_)
		self.type_keys.append(type_id)
		self.batches.append([])
		new_sample_priority_tree = SumSegmentTree(
			self._it_capacity, 
			with_min_tree=self._prioritization_importance_beta or (self._cluster_prioritisation_strategy is not None) or self._priority_can_be_negative or (self._prioritized_drop_probability > 0 and not self._stationarity_window_size), 
			with_max_tree=self._priority_can_be_negative, 
		)
		self._sample_priority_tree.append(new_sample_priority_tree)
		if self._prioritized_drop_probability > 0:
			self._drop_priority_tree.append(
				MinSegmentTree(self._it_capacity,neutral_element=((float('inf'),float('inf')),-1))
				if self._stationarity_window_size else
				new_sample_priority_tree.min_tree
			)
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		if self._weight_importance_by_update_time:
			self._update_times.append([])
		return True

	def resize_buffer(self):
		# print(random.random())
		# self.min_cluster_size, self.max_cluster_size = self.get_cluster_min_max_size(count_only_valid_clusters=False)
		## no need to remove redundancies now if the overall cluster's priority is capped
		new_min_cluster_size, new_max_cluster_size = self.get_cluster_min_max_size(count_only_valid_clusters=False)
		if new_max_cluster_size == self.max_cluster_size:
			return
		for t in self.type_values: # remove redundancies
			elements_to_remove = max(0, self.count(t)-new_max_cluster_size)
			for _ in range(elements_to_remove):
				self.remove_batch(t, self.get_less_important_batch(t))
		self.min_cluster_size = new_min_cluster_size
		self.max_cluster_size = new_max_cluster_size
	
	def normalize_priority(self, priority): # O(1)
		# if np.absolute(priority) < self._prioritization_epsilon:
		# 	priority = 0
		# always add self._prioritization_epsilon so that there is no priority equal to the neutral value of a SumSegmentTree
		return (-1 if priority < 0 else 1)*(np.absolute(priority) + self._prioritization_epsilon)**self._prioritization_alpha

	def get_priority(self, idx, type_id):
		type_ = self.get_type(type_id)
		return self._sample_priority_tree[type_][idx]

	def remove_batch(self, type_, idx): # O(log)
		last_idx = len(self.batches[type_])-1
		assert idx <= last_idx, 'idx cannot be greater than last_idx'
		type_id = self.type_keys[type_]
		del get_batch_indexes(self.batches[type_][idx])[type_id]
		if idx == last_idx: # idx is the last, remove it
			if self._prioritized_drop_probability > 0 and self._stationarity_window_size:
				self._drop_priority_tree[type_][idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[type_][idx] = None # O(log)
			if self._weight_importance_by_update_time:
				self._update_times[type_].pop()
			self._sample_priority_tree[type_][idx] = None # O(log)
			self.batches[type_].pop()
		elif idx < last_idx: # swap idx with the last element and then remove it
			if self._prioritized_drop_probability > 0 and self._stationarity_window_size:
				self._drop_priority_tree[type_][idx] = (self._drop_priority_tree[type_][last_idx][0],idx) # O(log)
				self._drop_priority_tree[type_][last_idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[type_][idx] = (self._insertion_time_tree[type_][last_idx][0],idx) # O(log)
				self._insertion_time_tree[type_][last_idx] = None # O(log)
			if self._weight_importance_by_update_time:
				self._update_times[type_][idx] = self._update_times[type_].pop()
			self._sample_priority_tree[type_][idx] = self._sample_priority_tree[type_][last_idx] # O(log)
			self._sample_priority_tree[type_][last_idx] = None # O(log)
			batch = self.batches[type_][idx] = self.batches[type_].pop()
			get_batch_indexes(batch)[type_id] = idx

	def count(self, type_=None):
		if type_ is None:
			if len(self.batches) == 0:
				return 0
			return sum(t.inserted_elements for t in self._sample_priority_tree)
		return self._sample_priority_tree[type_].inserted_elements

	def get_available_clusters(self):
		return [x for x in self.type_values if not self.is_empty(x)]

	def get_valid_clusters(self):
		return [x for x in self.type_values if self.has_atleast(self.min_cluster_size, x)]

	def get_avg_cluster_size(self):
		return int(np.floor(self.global_size/len(self.type_values)))

	def get_cluster_min_max_size(self, count_only_valid_clusters=False):
		C = len(self.get_available_clusters() if not count_only_valid_clusters else self.get_valid_clusters())
		S_min = int(np.floor(max(
			1,
			self.global_size/(C*self._clustering_xi)
		)))
		S_max = int(np.ceil(min(
			self.cluster_size,
			S_min + S_min*C*(self._clustering_xi-1)
		)))
		return S_min, S_max

	def get_cluster_capacity(self, segment_tree):
		return segment_tree.inserted_elements/self.max_cluster_size

	def get_relative_cluster_capacity(self, segment_tree):
		return segment_tree.inserted_elements/max(map(self.count, self.type_values))

	def get_cluster_priority(self, segment_tree, min_priority=0):
		def build_full_priority():
			if segment_tree.inserted_elements == 0:
				return 0
			if self._cluster_prioritisation_strategy == 'weighted_avg':
				avg_cluster_priority = (segment_tree.sum()/segment_tree.inserted_elements) - min_priority # O(log)
				assert avg_cluster_priority >= 0, f"avg_cluster_priority is {avg_cluster_priority}, it should be >= 0 otherwise the formula is wrong"
				# if self._clip_cluster_priority_by_max_capacity:
				# 	return min(1,self.get_cluster_capacity(segment_tree))*avg_cluster_priority
				return self.get_cluster_capacity(segment_tree)*avg_cluster_priority
			elif self._cluster_prioritisation_strategy == 'avg':
				avg_cluster_priority = (segment_tree.sum()/segment_tree.inserted_elements) - min_priority # O(log)
				assert avg_cluster_priority >= 0, f"avg_cluster_priority is {avg_cluster_priority}, it should be >= 0 otherwise the formula is wrong"
				return avg_cluster_priority
			# elif self._cluster_prioritisation_strategy == 'sum':
			sum_cluster_priority = segment_tree.sum() - min_priority*segment_tree.inserted_elements # O(log)
			assert sum_cluster_priority >= 0, f"sum_cluster_priority is {sum_cluster_priority}, it should be >= 0 otherwise the formula is wrong"
			# if self._clip_cluster_priority_by_max_capacity:
			# 	if segment_tree.inserted_elements > self.max_cluster_size: # redundancies have not been removed yet, cluster's priority is to capped to avoid cluster over-estimation
			# 		sum_cluster_priority *= self.max_cluster_size/segment_tree.inserted_elements
			return sum_cluster_priority
		return build_full_priority()**self._cluster_prioritization_alpha

	def get_cluster_capacity_dict(self):
		return dict(map(
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_capacity(x[1])), 
			enumerate(self._sample_priority_tree)
		))

	def get_cluster_priority_dict(self):
		min_priority = min(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)) if self._priority_lower_limit is None else 0 # O(log)
		return dict(map(
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_priority(x[1], min_priority)), 
			enumerate(self._sample_priority_tree)
		))

	def get_less_important_batch(self, type_):
		ptree = self._drop_priority_tree[type_] if random.random() <= self._prioritized_drop_probability else self._insertion_time_tree[type_]
		_,idx = ptree.min() # O(log)
		return idx

	def remove_less_important_batches(self, n):
		# Pick the right tree list
		if random.random() <= self._prioritized_drop_probability: 
			# Remove the batch with lowest priority
			tree_list = self._drop_priority_tree
		else: 
			# Remove the oldest batch
			tree_list = self._insertion_time_tree
		# Build the generator of the less important batch in every cluster
		# For all cluster to have the same size Y, we have that Y = N/C.
		# If we want to guarantee that every cluster contains at least pY elements while still reaching the maximum capacity of the whole buffer, then pY is the minimum size of a cluster.
		# If we want to constrain the maximum size of a cluster, we have to constrain with q the remaining (1-p)YC = (1-p)N elements so that (1-p)N = qpY, having that the size of a cluster is in [pY, pY+qpY].
		# Hence (1-p)N = qpN/C, then 1-p = qp/C, then p = 1/(1+q/C) = C/(C+q).
		# Therefore, we have that the minimum cluster's size pY = N/(C+q).
		less_important_batch_gen = (
			(*tree_list[type_].min(), type_) # O(log)
			for type_ in self.type_values
			if self.has_atleast(self.min_cluster_size, type_)
			# if not self.is_empty(type_)
		)
		less_important_batch_gen_len = len(self.type_values)
		# Remove the first N less important batches
		assert less_important_batch_gen_len > 0, "Cannot remove any batch from this buffer, it has too few elements"
		if n > 1 and less_important_batch_gen_len > 1:
			batches_to_remove = sorted(less_important_batch_gen, key=lambda x: x[0])
			n = min(n, len(batches_to_remove))
			for i in range(n):
				_, idx, type_ = batches_to_remove[i]
				self.remove_batch(type_, idx)
		else:
			_, idx, type_ = min(less_important_batch_gen, key=lambda x: x[0])
			self.remove_batch(type_, idx)
		if len(self.batches[type_]) == 0:
			logger.warning(f'Removed an old cluster with id {self.type_keys[type_]}, now there are {len(self.get_available_clusters())} different clusters.')
			self.resize_buffer()

	def _is_full_cluster(self, type_):
		return self.has_atleast(min(self.cluster_size,self.max_cluster_size), type_)
		
	def add(self, batch, type_id=0, update_prioritisation_weights=False): # O(log)
		self._add_type_if_not_exist(type_id)
		type_ = self.get_type(type_id)
		type_batch = self.batches[type_]
		################################
		# idx = None
		# if self._is_full_cluster(type_): # this cluster is full, remove one element from it
		# 	idx = self.get_less_important_batch(type_)
		# elif self.is_full_buffer(): # if full buffer, remove the less important batch in the whole buffer
		# 	self.remove_less_important_batches(1)
		# # Add new element to buffer
		# if idx is None:
		# 	idx = len(type_batch)
		# 	type_batch.append(batch)
		# 	if self._weight_importance_by_update_time:
		# 		self._update_times[type_].append(self._max_age_window)
		# else:
		# 	del get_batch_indexes(type_batch[idx])[type_id]
		# 	type_batch[idx] = batch
		# 	if self._weight_importance_by_update_time:
		# 		self._update_times[type_][idx] = self._max_age_window
		################################
		if self._is_full_cluster(type_) or self.is_full_buffer(): # if full buffer, remove the less important batch in the whole buffer
			self.remove_less_important_batches(1)
		# Add new element to buffer
		idx = len(type_batch)
		type_batch.append(batch)
		if self._weight_importance_by_update_time:
			self._update_times[type_].append(self._max_age_window)
		################################
		# Update batch infos
		batch_infos = get_batch_infos(batch)
		if 'batch_index' not in batch_infos:
			batch_infos['batch_index'] = {}
		batch_infos['batch_index'][type_id] = idx
		batch_infos['batch_uid'] = str(uuid.uuid4()) # random unique id
		# Set insertion time
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree[type_][idx] = (self.get_relative_time(), idx) # O(log)
		# Set drop priority
		if self._global_distribution_matching:
			if self._prioritized_drop_probability > 0 and self._stationarity_window_size:
				stationarity_stage_id = batch_infos['training_step']//self._stationarity_window_size
				if self._stationarity_smoothing_factor > 1:
					if random.random() >= 1/self._stationarity_smoothing_factor: # smoothly change stage without saturating the buffer with experience from the last episode
						stationarity_stage_id = max(0, stationarity_stage_id-1)
				# logger.warning((stationarity_stage_id,random.random()))
				self._drop_priority_tree[type_][idx] = (  # O(log)
					(
						stationarity_stage_id,
						random.random()
					), 
					idx
				)
		# Set priority
		self.update_priority(batch, idx, type_id) # add batch
		# Resize buffer
		if len(type_batch) == 1:
			logger.warning(f'Added a new cluster with id {type_id}, now there are {len(self.get_available_clusters())} different clusters.')
			self.resize_buffer()
		if self._prioritization_importance_beta:
			if update_prioritisation_weights: # Update weights after updating priority
				self._cache_priorities()
				self.update_beta_weights(batch, idx, type_)
			# elif PRIO_WEIGHTS not in batch: # Add default weights
			elif batch[PRIO_WEIGHTS] is None: # Add default weights
				batch[PRIO_WEIGHTS] = np.ones(batch.count, dtype=np.float32)
		if self.global_size:
			assert self.count() <= self.global_size, 'Memory leak in replay buffer; v1'
			assert super().count() <= self.global_size, 'Memory leak in replay buffer; v2'
		return idx, type_id

	def _cache_priorities(self):
		if self._prioritization_importance_beta or self._cluster_prioritisation_strategy is not None:
			self.__min_priority_list = tuple(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)) # O(log)
			self.__min_priority = min(self.__min_priority_list)
			self.__historical_min_priority = min(self.__min_priority,self.__historical_min_priority)
		if self._prioritization_importance_beta:
			self.__tot_priority_list = tuple(map(lambda x: x.sum(), self._sample_priority_tree)) # O(log)
			self.__tot_priority = sum(self.__tot_priority_list)
			self.__tot_elements_list = tuple(map(lambda x: x.inserted_elements, self._sample_priority_tree)) # O(1)
			self.__tot_elements = sum(self.__tot_elements_list)
		# if self._prioritization_importance_beta and self._priority_lower_limit is None:
		# 	self.__max_priority_list = tuple(map(lambda x: x.max_tree.max()[0], self._sample_priority_tree)) # O(log)
		# 	self.__max_priority = max(self.__max_priority_list)
		if self._cluster_prioritisation_strategy is not None:
			self.__cluster_priority_list = tuple(map(lambda x: self.get_cluster_priority(x, self.__min_priority if self._priority_lower_limit is None else 0), self._sample_priority_tree)) # always > 0
			self.__tot_cluster_priority = sum(self.__cluster_priority_list)
			# eta_normalise = lambda x: self.eta_normalisation(x, np.min(x), np.max(x), np.abs(np.std(x)/np.mean(x))) # using the coefficient of variation as eta
			# self.__cluster_priority_list = eta_normalise(eta_normalise(self.__cluster_priority_list)) # first eta-normalisation makes priorities in (0,1], but it inverts their magnitude # second eta-normalisation guarantees original priorities magnitude is preserved
			self.__min_cluster_priority = min(self.__cluster_priority_list)

	def sample_cluster(self):
		if self._cluster_prioritisation_strategy is not None:
			type_cumsum = np.cumsum(self.__cluster_priority_list) # O(|self.type_keys|)
			type_mass = random.random() * type_cumsum[-1] # O(1)
			assert 0 <= type_mass, f'type_mass {type_mass} should be greater than 0'
			assert type_mass <= type_cumsum[-1], f'type_mass {type_mass} should be lower than {type_cumsum[-1]}'
			type_,_ = next(filter(lambda x: x[-1] >= type_mass and not self.is_empty(x[0]), enumerate(type_cumsum))) # O(|self.type_keys|)
		else:
			type_ = random.choice(tuple(filter(lambda x: not self.is_empty(x), self.type_values)))
		type_id = self.type_keys[type_]
		return type_id, type_

	def sample(self, n=1, recompute_priorities=True): # O(log)
		if recompute_priorities:
			self._cache_priorities()
		type_id, type_ = self.sample_cluster()
		cluster_sum_tree = self._sample_priority_tree[type_]
		type_batch = self.batches[type_]
		idx_list = [
			cluster_sum_tree.find_prefixsum_idx(prefixsum_fn=lambda mass: mass*random.random(), check_min=self._priority_can_be_negative) # O(log)
			for _ in range(n)
		]
		batch_list = [
			type_batch[idx] # O(1)
			for idx in idx_list
		]
		# Update weights
		if self._prioritization_importance_beta: # Update weights
			for batch,idx in zip(batch_list,idx_list):
				self.update_beta_weights(batch, idx, type_)
		return batch_list

	def get_age_weight(self, type_, idx):
		return max(1,self._update_times[type_][idx])/self._max_age_window

	@staticmethod
	def normalise_priority(priority, historical_min_priority, n=1):
		historical_min_priority *= n
		assert priority >= historical_min_priority, f"priority must be >= historical_min_priority, but it is {priority} while historical_min_priority is {historical_min_priority}"
		return (priority - historical_min_priority)#/(upper_min_priority - lower_min_priority)

	def get_transition_probability(self, priority, type_=None, norm_fn=None):
		if norm_fn is None:
			if self._priority_lower_limit is not None:
				norm_fn = (lambda p,n: p)
			else:
				if priority < self.__historical_min_priority: self.__historical_min_priority = priority
				norm_fn = (lambda x,n: self.normalise_priority(x, self.__historical_min_priority, n=n))
		if type_ is None:
			return norm_fn(priority, 1) / norm_fn(self.__tot_priority, self.__tot_elements)
		p_cluster = self.__cluster_priority_list[type_] / self.__tot_cluster_priority # clusters priorities are already > 0
		p_transition_given_cluster = norm_fn(priority, 1) / norm_fn(self.__tot_priority_list[type_], self.__tot_elements_list[type_])
		# print(p_cluster, p_transition_given_cluster)
		return p_cluster*p_transition_given_cluster # joint probability of dependent events

	def update_beta_weights(self, batch, idx, type_):
		##########
		# Get priority weight
		this_priority = self._sample_priority_tree[type_][idx]
		# assert self.__min_priority_list == tuple(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)), "Wrong beta updates"
		if self._cluster_level_weighting and self._cluster_prioritisation_strategy is not None:
			this_probability = self.get_transition_probability(this_priority, type_)
			min_probability = min((self.get_transition_probability(self.__min_priority_list[x], x) for x in self.type_values))
		else:
			this_probability = self.get_transition_probability(this_priority)
			min_probability = self.get_transition_probability(self.__min_priority)
		weight = min_probability/this_probability
		weight = weight**self._prioritization_importance_beta
		##########
		# Add age weight
		# if self._weight_importance_by_update_time:
		# 	weight *= self.get_age_weight(type_, idx) # batches with outdated priorities should have a lower weight, they might be just noise
		##########
		batch[PRIO_WEIGHTS] = np.full(batch.count, weight, dtype=np.float32)

	def get_batch_priority(self, batch):
		priority_batch = batch[self._priority_id]
		if not isinstance(priority_batch,(list,tuple,np.ndarray)):
			priority_batch = list(priority_batch)
		return self._priority_aggregation_fn(priority_batch)
	
	def update_priority(self, new_batch, idx, type_id=0): # O(log)
		type_ = self.get_type(type_id)
		if type_ is None:
			return
		if idx >= len(self.batches[type_]):
			return
		if get_batch_uid(new_batch) != get_batch_uid(self.batches[type_][idx]):
			return
		# for k,v in self.batches[type_][idx].data.items():
		# 	if not np.array_equal(new_batch[k],v):
		# 		print(k,v,new_batch[k])
		new_priority = self.get_batch_priority(new_batch)
		if self._priority_lower_limit is not None:
			assert new_priority >= self._priority_lower_limit, f"new_priority must be > priority_lower_limit, but it is {min_priority}"
			new_priority -= self._priority_lower_limit
		normalized_priority = self.normalize_priority(new_priority)
		# self.priority_stats.push(normalized_priority)
		# Update priority
		if self._weight_importance_by_update_time:
			normalized_priority *= self.get_age_weight(type_, idx) # batches with outdated priorities should have a lower weight, they might be just noise
		self._sample_priority_tree[type_][idx] = normalized_priority # O(log)
		if self._weight_importance_by_update_time:
			self._update_times[type_][idx] = self._update_times[type_][idx] - 1 # O(1)
		# Set drop priority
		if not self._global_distribution_matching: 
			if self._prioritized_drop_probability > 0 and self._stationarity_window_size:
				batch_infos = get_batch_infos(new_batch)
				stationarity_stage_id = batch_infos['training_step']//self._stationarity_window_size
				if self._stationarity_smoothing_factor > 1:
					if random.random() >= 1/self._stationarity_smoothing_factor: # smoothly change stage without saturating the buffer with experience from the last episode
						stationarity_stage_id = max(0, stationarity_stage_id-1)
				# logger.warning((stationarity_stage_id,random.random()))
				self._drop_priority_tree[type_][idx] = (  # O(log)
					(
						stationarity_stage_id,
						normalized_priority
					), 
					idx
				)
		return normalized_priority

	def get_relative_time(self):
		return time.time()-self._base_time

	def stats(self, debug=False):
		stats_dict = super().stats(debug)
		stats_dict.update({
			'cluster_capacity':self.get_cluster_capacity_dict(),
			'cluster_priority': self.get_cluster_priority_dict(),
		})
		return stats_dict
