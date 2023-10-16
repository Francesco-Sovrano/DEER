import collections
import logging
import numpy as np
import platform
from more_itertools import unique_everseen
from itertools import islice
import copy
import threading 
import random

# Import ray before psutil will make sure we use psutil's bundled version
import ray  # noqa F401
import psutil  # noqa E402

from deer.experience_buffers.buffer.pseudo_prioritized_buffer import PseudoPrioritizedBuffer, get_batch_infos, get_batch_indexes, get_batch_uid, discard_batch
from deer.experience_buffers.buffer.buffer import Buffer
from deer.experience_buffers.explanation_cluster_manager import *
from deer.utils import ReadWriteLock

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.util.iter import ParallelIteratorWorker
from ray.util.timer import _Timer as TimerStat

logger = logging.getLogger(__name__)


def get_clustered_replay_buffer(config):
	assert config.batch_mode == "complete_episodes" or not config.clustering_options["cluster_with_episode_type"], f"This algorithm requires 'complete_episodes' as batch_mode when 'cluster_with_episode_type' is True"
	clustering_scheme_type = config.clustering_options.get("clustering_scheme", None)
	# no need for unclustered_buffer if clustering_scheme_type is none
	ratio_of_samples_from_unclustered_buffer = config.clustering_options["ratio_of_samples_from_unclustered_buffer"] if clustering_scheme_type else 0
	local_replay_buffer = LocalReplayBuffer(
		config.buffer_options, 
		learning_starts=config.num_steps_sampled_before_learning_starts,
		seed=config.seed,
		cluster_selection_policy=config.clustering_options["cluster_selection_policy"],
		ratio_of_samples_from_unclustered_buffer=ratio_of_samples_from_unclustered_buffer,
	)
	clustering_scheme = ClusterManager(clustering_scheme_type, config.clustering_options["clustering_scheme_options"])
	return local_replay_buffer, clustering_scheme


def assign_types(multi_batch, clustering_scheme, batch_fragment_length,
				 with_episode_type=True, training_step=None):
	if not isinstance(multi_batch, MultiAgentBatch):
		multi_batch = MultiAgentBatch({DEFAULT_POLICY_ID: multi_batch}, multi_batch.count)
	
	# if not with_episode_type:
	# 	batch_list = multi_batch.timeslices(batch_fragment_length) if multi_batch.count > batch_fragment_length else [multi_batch]
	# 	for i,batch in enumerate(batch_list):
	# 		for pid,sub_batch in batch.policy_batches.items():
	# 			get_batch_infos(sub_batch)['batch_type'] = clustering_scheme.get_batch_type(sub_batch, training_step=training_step, episode_step=i, agent_id=pid)		
	# 	return batch_list

	batch_dict = {}
	for pid, meta_batch in multi_batch.policy_batches.items():
		batch_dict[pid] = []
		batch_list = meta_batch.split_by_episode() if with_episode_type else [meta_batch]
		for batch in batch_list:
			sub_batch_count = int(np.ceil(len(batch)/batch_fragment_length))
			sub_batch_list = [
				batch[i*batch_fragment_length : (i+1)*batch_fragment_length]
				for i in range(sub_batch_count)
			] if len(batch) > batch_fragment_length else [batch]
			episode_type = clustering_scheme.get_episode_type(sub_batch_list) if with_episode_type else None
			for i,sub_batch in enumerate(sub_batch_list):
				batch_type = clustering_scheme.get_batch_type(
					sub_batch, 
					episode_type=episode_type, 
					training_step=training_step, 
					episode_step=i, 
					agent_id=pid
				)
				sub_batch[SampleBatch.INFOS] = [{'batch_type': batch_type,'training_step': training_step}] # remove unnecessary infos to save some memory
			batch_dict[pid] += sub_batch_list
	return [
		MultiAgentBatch(
			{
				pid: b
				for pid,b in zip(batch_dict.keys(),b_list)
			},
			b_list[0].count
		)
		for b_list in zip(*batch_dict.values())
	]


def add_buffer_metrics(results, buffer):
	results['buffer']=buffer.stats()
	return results


def apply_to_batch_once(fn, batch_list):
	updated_batch_dict = {
		get_batch_uid(x): fn(x) 
		for x in unique_everseen(batch_list, key=get_batch_uid)
	}
	return list(map(lambda x: updated_batch_dict[get_batch_uid(x)], batch_list))


class MultiAgentBatchWithDefaultAgent(MultiAgentBatch):

	def __init__(self, policy_batches, env_steps, default_agent_id):
		super().__init__(policy_batches, env_steps)
		self._default_agent_id = default_agent_id

	def __getitem__(self, key):
		# print(12, key)
		data = self.policy_batches[self._default_agent_id]
		return data[key] if key in data else None

	def __setitem__(self, key, value):
		# if key!='infos':
		# 	print(11, key,value)
		data = self.policy_batches[self._default_agent_id]
		data[key] = value

	# @property
	# def count(self):
	# 	return self.policy_batches[self._default_agent_id].count

	@staticmethod
	def from_multi_agent_batch(b, agent_id=None):
		if isinstance(b, MultiAgentBatchWithDefaultAgent):
			b._default_agent_id = agent_id
			return b
		assert isinstance(b, MultiAgentBatch)
		return MultiAgentBatchWithDefaultAgent(b.policy_batches, b.count, agent_id)

	def to_multi_agent_batch(self):
		return MultiAgentBatch(self.policy_batches, self.count)

	def to_sample_batch(self):
		return self.policy_batches[self._default_agent_id]


class SimpleReplayBuffer:
	"""Simple replay buffer that operates over batches."""

	def __init__(self, num_slots, seed=None):
		"""Initialize SimpleReplayBuffer.

		Args:
			num_slots (int): Number of batches to store in total.
		"""
		self.num_slots = num_slots
		self.replay_batches = []
		self.replay_index = 0
		random.seed(seed)
		np.random.seed(seed)

	def can_replay(self):
		return len(self.replay_batches) >= self.num_slots

	def add_batch(self, sample_batch):
		if discard_batch(sample_batch):
			return 0
		# if self.batch_dropout_rate and np.random.random() < self.batch_dropout_rate:
		# 	return 0
		if self.num_slots > 0:
			if len(self.replay_batches) < self.num_slots:
				self.replay_batches.append(sample_batch)
			else:
				self.replay_batches[self.replay_index] = sample_batch
				self.replay_index = (self.replay_index+1)%self.num_slots
		return 1

	def replay(self, batch_count=1):
		return random.sample(self.replay_batches, batch_count)


class LocalReplayBuffer(ParallelIteratorWorker):
	"""A replay buffer shard.

	Ray actors are single-threaded, so for scalability multiple replay actors
	may be created to increase parallelism."""

	def __init__(self, 
		buffer_options, 
		learning_starts=1000,
		seed=None,
		cluster_selection_policy='random_uniform',
		ratio_of_samples_from_unclustered_buffer=0,
	):
		self.buffer_options = buffer_options
		self.prioritized_replay = self.buffer_options['prioritized_replay']
		self.centralised_buffer = self.buffer_options['centralised_buffer']
		logger.warning(f'Building LocalReplayBuffer with centralised_buffer = {self.centralised_buffer}')
		self.replay_integral_multi_agent_batches = self.buffer_options.get('replay_integral_multi_agent_batches', False)
		dummy_buffer = PseudoPrioritizedBuffer(**self.buffer_options)
		self.buffer_size = dummy_buffer.global_size
		self.is_weighting_expected_values = dummy_buffer.is_weighting_expected_values()
		self.replay_starts = learning_starts
		self.batch_dropout_rate = self.buffer_options.get('batch_dropout_rate', 0)
		self._buffer_lock = ReadWriteLock()
		self._cluster_selection_policy = cluster_selection_policy
		
		random.seed(seed)
		np.random.seed(seed)

		ParallelIteratorWorker.__init__(self, None, False)

		def new_buffer():
			return PseudoPrioritizedBuffer(**self.buffer_options, seed=seed) if self.prioritized_replay else Buffer(**self.buffer_options, seed=seed)

		self.replay_buffers = collections.defaultdict(new_buffer)
		self.ratio_of_old_elements = np.clip(1-ratio_of_samples_from_unclustered_buffer, 0,1)
		self.buffer_of_recent_elements = collections.defaultdict(new_buffer) if ratio_of_samples_from_unclustered_buffer > 0 else None

		# Metrics
		self.add_batch_timer = TimerStat()
		self.replay_timer = TimerStat()
		self.update_priorities_timer = TimerStat()
		self.num_added = 0

	def add_batch(self, batch, update_prioritisation_weights=False):
		# Handle everything as if multiagent
		if not isinstance(batch, MultiAgentBatch):
			batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
		added_batches = 0
		with self.add_batch_timer:
			self._buffer_lock.acquire_write()
			for policy_id in batch.policy_batches.keys():
				sub_batch = MultiAgentBatchWithDefaultAgent.from_multi_agent_batch(batch, policy_id)
				if discard_batch(sub_batch):
					continue
				if self.batch_dropout_rate and np.random.random() < self.batch_dropout_rate:
					continue
				buffer_id = DEFAULT_POLICY_ID if self.centralised_buffer else policy_id
				batch_type = get_batch_infos(sub_batch)["batch_type"]
				####################################
				if not isinstance(batch_type,(tuple,list)):
					sub_type_list = (batch_type,)
				elif len(batch_type) == 1:
					sub_type_list = (batch_type[0],)
				elif self._cluster_selection_policy == 'random_uniform_after_filling':
					sub_type_list = tuple(filter(lambda x: not self.replay_buffers[buffer_id].is_valid_cluster(x), batch_type))
					if len(sub_type_list) == 0:
						sub_type_list = (random.choice(batch_type),)
				elif self._cluster_selection_policy == 'random_uniform':
					# # If has_multiple_types is True: no need for duplicating the batch across multiple clusters unless they are invalid, just insert into one of them, randomly. It is a prioritised buffer, clusters will be fairly represented, with minimum overhead.
					sub_type_list = (random.choice(batch_type),)
				elif self._cluster_selection_policy == 'random_max':
					cluster_cumsum = np.cumsum(list(map(lambda x: self.replay_buffers[buffer_id].get_cluster_size(x)+1, batch_type)))
					cluster_mass = random.random() * cluster_cumsum[-1] # O(1)
					batch_type_idx,_ = next(filter(lambda x: x[-1] >= cluster_mass, enumerate(cluster_cumsum))) # O(|self.type_keys|)
					sub_type_list = (batch_type[batch_type_idx],)
				elif self._cluster_selection_policy == 'max':
					sub_type_list = (max(
						batch_type, 
						key=lambda x: (self.replay_buffers[buffer_id].get_cluster_size(x),random.random())
					),)
				elif self._cluster_selection_policy == 'min':
					sub_type_list = (min(
						batch_type, 
						key=lambda x: (self.replay_buffers[buffer_id].get_cluster_size(x),random.random())
					),)
				else: #if self._cluster_selection_policy == 'none':
					sub_type_list = batch_type
				####################################
				for sub_type in sub_type_list: 
					# Make a deep copy so the replay buffer doesn't pin plasma memory.
					sub_batch = MultiAgentBatchWithDefaultAgent.from_multi_agent_batch(batch.copy(), policy_id)
					# Make a deep copy of infos so that for every sub_type the infos dictionary is different
					sub_batch[SampleBatch.INFOS] = copy.deepcopy(sub_batch[SampleBatch.INFOS])
					self.replay_buffers[buffer_id].add(batch=sub_batch, type_id=sub_type, update_prioritisation_weights=update_prioritisation_weights)
					added_batches += 1
					if self.buffer_of_recent_elements is not None:
						# Make a deep copy so the replay buffer doesn't pin plasma memory.
						sub_batch = MultiAgentBatchWithDefaultAgent.from_multi_agent_batch(batch.copy(), policy_id)
						# Make a deep copy of infos so that for every sub_type the infos dictionary is different
						sub_batch[SampleBatch.INFOS] = copy.deepcopy(sub_batch[SampleBatch.INFOS])
						self.buffer_of_recent_elements[buffer_id].add(batch=sub_batch, update_prioritisation_weights=update_prioritisation_weights)
			self._buffer_lock.release_write()
		self.num_added += added_batches
		return added_batches

	def can_replay(self):
		return self.num_added >= self.replay_starts

	def replay(self, batch_count=1, cluster_overview_size=None, update_replayed_fn=None):
		output_batches = []
		if self.buffer_of_recent_elements is not None:
			n_of_old_elements = max(1,int(np.ceil(batch_count*self.ratio_of_old_elements))) #random.randint(0,batch_count)
			# if n_of_old_elements > 0:
			output_batches += self.sample_from_buffer(
				self.replay_buffers,
				n_of_old_elements,
				cluster_overview_size,
				update_replayed_fn,
			)
			if n_of_old_elements != batch_count:
				output_batches += self.sample_from_buffer(
					self.buffer_of_recent_elements,
					batch_count-n_of_old_elements,
					cluster_overview_size,
					update_replayed_fn,
				)
		else:
			output_batches += self.sample_from_buffer(
				self.replay_buffers,
				batch_count,
				cluster_overview_size,
				update_replayed_fn,
			)
		# if output_batches:
		# 	print(13, output_batches[0])
		return output_batches

	def sample_from_buffer(self, buffer_dict, batch_count=1, cluster_overview_size=None, update_replayed_fn=None):
		if not self.can_replay():
			return []
		if not cluster_overview_size:
			cluster_overview_size = batch_count
		else:
			cluster_overview_size = min(cluster_overview_size,batch_count)

		with self.replay_timer:
			batch_list = [{} for _ in range(batch_count)]
			buffer_dict_items = [x for x in buffer_dict.items() if not x[-1].is_empty()]
			for buffer_idx,(policy_id, replay_buffer) in enumerate(buffer_dict_items):
				# if replay_buffer.is_empty():
				# 	continue
				# batch_iter = replay_buffer.sample(batch_count)
				batch_size_list = [cluster_overview_size]*(batch_count//cluster_overview_size)
				if batch_count%cluster_overview_size > 0:
					batch_size_list.append(batch_count%cluster_overview_size)
				self._buffer_lock.acquire_read()
				batch_iter = []
				for i,n in enumerate(batch_size_list):
					batch_iter += replay_buffer.sample(n,recompute_priorities=i==0)
				self._buffer_lock.release_read()
				if update_replayed_fn:
					self._buffer_lock.acquire_write()
					batch_iter = apply_to_batch_once(update_replayed_fn, batch_iter)
					self._buffer_lock.release_write()
				if not self.replay_integral_multi_agent_batches:
					for i,batch in enumerate(batch_iter):
						batch_list[i][policy_id] = batch.to_sample_batch()
				else:
					for i,batch in enumerate(batch_iter):
						# print((buffer_idx+i)%len(buffer_dict_items), buffer_idx, i, len(buffer_dict_items))
						if (buffer_idx+i)%len(buffer_dict_items) == 0: # every batch has information about every agent, hence we take batch_list/len(buffer_dict_items) batches per buffer with this formula
							batch_list[i] = batch.to_multi_agent_batch()
		# print(batch_list)
		return (
			MultiAgentBatch(samples, max(map(lambda x:x.count, samples.values()))) 
			if isinstance(samples,dict) else 
			samples
			for samples in batch_list
		)

	def increase_train_steps(self, t=1):
		for replay_buffer in self.replay_buffers.values():
			replay_buffer.increase_steps(t)

	def get_train_steps(self):
		return max((replay_buffer.timesteps for replay_buffer in self.replay_buffers.values())) if self.replay_buffers else 0

	def update_priorities(self, prio_dict):
		if not self.prioritized_replay:
			return
		with self.update_priorities_timer:
			self._buffer_lock.acquire_write()
			for policy_id, new_batch in prio_dict.items():
				for type_id,batch_index in get_batch_indexes(new_batch).items():
					self.replay_buffers[policy_id].update_priority(new_batch, batch_index, type_id)
					if self.buffer_of_recent_elements is not None:
						self.buffer_of_recent_elements[policy_id].update_priority(new_batch, batch_index)
			self._buffer_lock.release_write()

	def stats(self, debug=False):
		stat = {
			"add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
			"replay_time_ms": round(1000 * self.replay_timer.mean, 3),
			"update_priorities_time_ms": round(1000 * self.update_priorities_timer.mean, 3),
		}
		for policy_id, replay_buffer in self.replay_buffers.items():
			stat.update({
				policy_id: replay_buffer.stats(debug=debug)
			})
		return stat
