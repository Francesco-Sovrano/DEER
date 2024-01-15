"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501
from more_itertools import unique_everseen
from ray.rllib.utils.annotations import override
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights, DQNConfig, DQN
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.metrics import (
	NUM_ENV_STEPS_SAMPLED,
	NUM_AGENT_STEPS_SAMPLED,
)
from ray.rllib.execution.train_ops import (
	train_one_step,
	multi_gpu_train_one_step,
)
from ray.rllib.execution.common import (
	LAST_TARGET_UPDATE_TS,
	NUM_TARGET_UPDATES,
	STEPS_TRAINED_COUNTER,
)
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER

from deer.experience_buffers.replay_buffer_wrapper_utils import get_clustered_replay_buffer, assign_types, add_buffer_metrics
from deer.agents.xadqn.xadqn_tf_policy import XADQNTFPolicy

from deer.agents.xadqn.xadqn_torch_policy import XADQNTorchPolicy, add_policy_signature, torch
from ray.rllib.utils.schedules.linear_schedule import *

from deer.experience_buffers.buffer.buffer import Buffer
from deer.models.torch.head_generator.siamese_model_wrapper import SiameseAdaptiveModel
import gym
from ray.rllib.utils.framework import try_import_torch
from collections import deque, defaultdict

torch, nn = try_import_torch()
import random
import numpy as np
import time

get_batch_infos = lambda x: x[SampleBatch.INFOS][0]
get_batch_indexes = lambda x: get_batch_infos(x)['batch_index']
get_batch_uid = lambda x: get_batch_infos(x)['batch_uid']
get_batch_type = lambda x: get_batch_infos(x)['batch_type'][0]
get_training_step = lambda x: get_batch_infos(x)['training_step']


class PolicySignatureListCollector(SimpleListCollector):
	def get_inference_input_dict(self, policy_id):
		batch = super().get_inference_input_dict(policy_id)
		policy = self.policy_map[policy_id]
		return add_policy_signature(batch,policy)


def init_xa_config(self):
	self.num_steps_sampled_before_learning_starts = 2**14
	self.min_train_timesteps_per_iteration = 1
	self.siamese_config = {
		"use_siamese": True,
		"buffer_size": 10,
		"update_frequency": 10000,
		"embedding_size": 64,
	}
	self.n_step_annealing_scheduler = {
		'fn': None, # function name in string format; one of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
		'args': {} # the arguments to pass to the function; for more details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
	}
	self.n_step_random_sampling = False # a Boolean
	self.buffer_options = {
		'prioritized_replay': True,
		#### MARL
		'centralised_buffer': True,  # for MARL
		'replay_integral_multi_agent_batches': False, # for MARL, set this to True for MADDPG and QMIX
		'batch_dropout_rate': 0, # Probability of dropping a state transition before adding it to the experience buffer. Set this to any value greater than zero to randomly drop state transitions
		#### ER
		'priority_id': 'td_errors', # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_lower_limit': 0, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'global_size': 2**14, # Default 50000. Maximum number of batches stored in all clusters (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'n_step' (default is 1).
		#### PER
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		#### XAER
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'n_step' (default is 1).
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 1, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Default is False. Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity. In fact, until al the clusters have reached the minimum size, some clusters may have more elements than the maximum size, to avoid shrinking the buffer capacity with clusters having not enough transitions (i.e. 1 transition).
		#### DEER
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far. 
		'stationarity_window_size': None, # If lower than float('inf') and greater than 0, then the stationarity_window_size W is used to guarantee that every W training-steps the buffer is emptied from old state transitions.
		'stationarity_smoothing_factor': 1, # A number >= 1, where 1 means no smoothing. The larger this number, the smoother the transition from a stationarity stage to the next one. This should help avoiding experience buffers saturated by one single episode during a stage transition. The optimal value should be equal to ceil(HORIZON*number_of_agents/EXPERIENCE_BUFFER_SIZE)*stationarity_window_size.
		#### Extra
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	}
	self.clustering_options = {
		'clustering_scheme': ['Who','How_Well','Why','Where','What','How_Many'], # Which scheme to use for building clusters. Set it to None or to a list of the following: How_WellOnZero, How_Well, When_DuringTraining, When_DuringEpisode, Why, Why_Verbose, Where, What, How_Many, Who
		'clustering_scheme_options': {
			"n_clusters": {
				"who": 4,
				# "why": 8,
				# "what": 8,
			},
			"default_n_clusters": 8,
			"frequency_independent_clustering": False, # Setting this to True can be memory expensive, especially for who explanations
			"agent_action_sliding_window": 2**4,
			"episode_window_size": 2**6, 
			"batch_window_size": 2**8, 
			"training_step_window_size": 2**2,
		},
		'cluster_selection_policy': "min", # Which policy to follow when clustering_scheme is not "none" and multiple explanatory labels are associated to a batch. One of the following: 'random_uniform_after_filling', 'random_uniform', 'random_max', 'max', 'min', 'none'
		'cluster_with_episode_type': False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
		'cluster_overview_size': 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
		'collect_cluster_metrics': False, # Whether to collect metrics about the experience clusters. It consumes more resources.
		'ratio_of_samples_from_unclustered_buffer': 0, # 0 for no, 1 for full. Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
	}


class XADQNConfig(DQNConfig):

	def __init__(self, algo_class=None):
		"""Initializes a DQNConfig instance."""
		super().__init__(algo_class=algo_class or XADQN)

		# Changes to SimpleQConfig's default:
		init_xa_config(self)

	@override(DQNConfig)
	def validate(self):
		# Call super's validation method.
		super().validate()

		if self.model["custom_model_config"].get("add_nonstationarity_correction", False):
			self.sample_collector = PolicySignatureListCollector


class XADQN(DQN):
	def __init__(self, *args, **kwargs):
		self._allow_unknown_subkeys += ["clustering_options"]
		self.sample_batch_size = None
		self.replay_batch_size = None
		self.clustering_scheme = None
		self.siamese_model = None
		self.positive_buffer = None
		self.triplet_buffer = None
		self.siamese_config = None
		self.optimizer = None
		self.loss_fn = None
		self.use_siamese = None
		self.s_buffer_size = None
		super().__init__(*args, **kwargs)



	@classmethod
	@override(DQN)
	def get_default_config(cls):
		return XADQNConfig()

	@classmethod
	@override(DQN)
	def get_default_policy_class(self, config):
		return XADQNTorchPolicy if config['framework'] == "torch" else XADQNTFPolicy

	def update_n_steps(self):
		assert self.n_step_annealing_scheduler

		def _update_worker_n_steps(w):
			print(f"Updating n_step for worker {w.worker_index}")
			for policy in w.policy_map.values():
				policy.config['n_step'] = self.n_step_annealing_scheduler.value(self._counters['training_steps']+1)
		if self.n_step_annealing_scheduler:
			self.workers.foreach_worker(_update_worker_n_steps)

	@override(DQN)
	def setup(self, config):
		if config.n_step_annealing_scheduler['args'].get('initial_p',None):
			assert config.n_step_annealing_scheduler['args']['initial_p'] <= config.rollout_fragment_length, f"n_step_annealing_scheduler['args']['initial_p'] ({config.n_step_annealing_scheduler['args']['initial_p']}) must be lower than or equal to the rollout_fragment_length ({config.rollout_fragment_length})"
		else:
			assert config.n_step <= config.rollout_fragment_length, f'n_step ({config.n_step}) must be lower than or equal to the rollout_fragment_length ({config.rollout_fragment_length})'

		random.seed(config.seed)
		np.random.seed(config.seed)
		super().setup(config)

		self._counters['training_steps'] = 0
		if self.config.n_step_annealing_scheduler['fn']:
			self.n_step_annealing_scheduler = eval(self.config.n_step_annealing_scheduler['fn'])(**self.config.n_step_annealing_scheduler['args'])
			self.update_n_steps()
		else:
			self.n_step_annealing_scheduler = None
		
		self.replay_batch_size = self.config.train_batch_size

		self.sample_batch_size = 1
		# if self.sample_batch_size and self.sample_batch_size > 1:
		# 	self.replay_batch_size = int(max(1, self.replay_batch_size // self.sample_batch_size))

		############
		self._counters['last_siamese_update'] = 0
		self.siamese_config = config.get("siamese_config", {})
		self.use_siamese = self.siamese_config.get('use_siamese', False)
		self.s_buffer_size = self.siamese_config.get('buffer_size', 10000)
		self.positive_buffer = Buffer(global_size=self.s_buffer_size, seed=42)
		self.triplet_buffer = {
			'anchor': deque(maxlen=self.s_buffer_size),
			'positive': deque(maxlen=self.s_buffer_size),
			'negative': deque(maxlen=self.s_buffer_size),
		}
		self.local_replay_buffer, self.clustering_scheme = (
			get_clustered_replay_buffer(self.config, siamese=self.use_siamese))

		if self.use_siamese:
			device = self.get_policy().device
			_, env_creator = self._get_env_id_and_creator(config.env, config)
			tmp_env = env_creator(config["env_config"])
			embedding_size = self.siamese_config.get('embedding_size', 64)
			self.siamese_model = SiameseAdaptiveModel(gym.spaces.Dict({
				f"obs": tmp_env.observation_space,
				f"new_obs": tmp_env.observation_space,
				f"actions": tmp_env.action_space,
				f"rewards": gym.spaces.Box(
					low=float('-inf'), high=float('inf'),
					shape=(1,), dtype=np.float32), }),
				embedding_size=embedding_size,
				env=config.env)
			self.loss_fn = torch.nn.TripletMarginLoss(
				self.siamese_config.get('loss_margin', 1.0), p=2)
			self.optimizer = torch.optim.Adam(
				self.siamese_model.parameters(), lr=1e-3, weight_decay=1e-10)
			self.siamese_model.to(device)
		
		def add_view_requirements(w):
			for policy in w.policy_map.values():
				# policy.view_requirements[SampleBatch.T] = ViewRequirement(SampleBatch.T, shift=0)
				policy.view_requirements[SampleBatch.INFOS] = ViewRequirement(SampleBatch.INFOS, shift=0)
				if config.buffer_options["priority_id"] == "td_errors":
					policy.view_requirements["td_errors"] = ViewRequirement("td_errors", shift=0)
				if config.model["custom_model_config"].get("add_nonstationarity_correction", False):
					policy.view_requirements["policy_signature"] = ViewRequirement("policy_signature", used_for_compute_actions=True, shift=0)
		self.workers.foreach_worker(add_view_requirements)

	def format_transition_for_siamese_input(self, x):
		embedding_size = self.siamese_config.get('embedding_size',64)
		return {
			f"s_t": x[CUR_OBS],
			f"s_(t+1)": x[NEXT_OBS],
			f"a_t": x[ACTIONS],
			f"r_t": x[REWARDS],
		}

	@override(DQN)
	def training_step(self):
		"""DQN training iteration function.

		Each training iteration, we:
		- Sample (MultiAgentBatch) from workers.
		- Store new samples in replay buffer.
		- Sample training batch (MultiAgentBatch) from replay buffer.
		- Learn on training batch.
		- Update remote workers' new policy weights.
		- Update target network every `target_network_update_freq` sample steps.
		- Return all collected metrics for the iteration.

		Returns:
			The results dict from executing the training iteration.
		"""
		start = time.time()
		train_results = {}

		# We alternate between storing new samples and sampling and training
		store_weight, sample_and_train_weight = calculate_rr_weights(self.config)
		siamese_losses = []
		for _ in range(store_weight):
			# Sample (MultiAgentBatch) from workers.
			new_sample_batch = synchronous_parallel_sample(
				worker_set=self.workers, concat=True
			)

			# Update counters
			self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
			self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

			# Store new samples in replay buffer.
			sub_batch_iter = assign_types(
				new_sample_batch, self.clustering_scheme,
				self.sample_batch_size,
				with_episode_type=self.config.clustering_options['cluster_with_episode_type'],
				training_step=self.local_replay_buffer.get_train_steps())

			############
			if self.use_siamese:
				explanation_batch_dict = defaultdict(list)
				for sub_batch in sub_batch_iter:
					pol_sub_batch = sub_batch['default_policy']
					explanatory_label = get_batch_type(pol_sub_batch)
					explanation_batch_dict[explanatory_label].append(pol_sub_batch)
					self.positive_buffer.add(pol_sub_batch, explanatory_label)

				if len(explanation_batch_dict.keys()) >= 2: # TODO: check if there is a way to avoid this check
					anchor_class, negative_class = random.sample(list(
						explanation_batch_dict.keys()), 2)
					# TODO: check if there is a way to avoid this check too
					if len(self.positive_buffer.get_batches(anchor_class)) < 1:
						print(f"Warning: not enough positive samples for "
							  f"class {anchor_class} at time step "
							  f"{self._counters['training_steps']}")
						continue
					self.triplet_buffer['anchor'].append(random.choice(
						explanation_batch_dict[anchor_class]))
					self.triplet_buffer['positive'].append(random.choice(
						self.positive_buffer.get_batches(anchor_class)))
					self.triplet_buffer['negative'].append(random.choice(
						explanation_batch_dict[negative_class]))
			############

			total_buffer_additions = sum(map(self.local_replay_buffer.add_batch, sub_batch_iter))

		global_vars = {
			"timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
		}
		
		# Update target network every `target_network_update_freq` sample steps.
		cur_ts = self._counters[
			NUM_AGENT_STEPS_SAMPLED
			if self.config.count_steps_by == "agent_steps"
			else NUM_ENV_STEPS_SAMPLED
		]

		def update_priorities(samples, info_dict):
			self.local_replay_buffer.increase_train_steps()
			if not self.config.buffer_options['prioritized_replay']:
				return info_dict
			priority_id = self.config.buffer_options["priority_id"]
			if priority_id == "td_errors":
				for policy_id, info in info_dict.items():
					td_errors = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
					# samples.policy_batches[policy_id].set_get_interceptor(None)
					samples.policy_batches[policy_id]["td_errors"] = td_errors
			# IMPORTANT: split train-batch into replay-batches, using batch_uid, before updating priorities
			policy_batch_list = []
			for policy_id, batch in samples.policy_batches.items():
				if self.sample_batch_size > 1 and self.config.batch_mode == "complete_episodes":
					sub_batch_indexes = [
						i
						for i, infos in enumerate(batch['infos'])
						if "batch_uid" in infos
					] + [batch.count]
					sub_batch_iter = (
						batch.slice(sub_batch_indexes[j], sub_batch_indexes[j+1])
						for j in range(len(sub_batch_indexes)-1)
					)
				else:
					sub_batch_iter = batch.timeslices(self.sample_batch_size)
				sub_batch_iter = unique_everseen(sub_batch_iter, key=get_batch_uid)
				for i, sub_batch in enumerate(sub_batch_iter):
					if i >= len(policy_batch_list):
						policy_batch_list.append({})
					policy_batch_list[i][policy_id] = sub_batch
			for policy_batch in policy_batch_list:
				self.local_replay_buffer.update_priorities(policy_batch)
			return info_dict

		# print(cur_ts > self.config.num_steps_sampled_before_learning_starts, cur_ts, self.config.num_steps_sampled_before_learning_starts)
		if cur_ts > self.config.num_steps_sampled_before_learning_starts:
			train_start = time.time()
			print(f"Time to start training: {train_start-start} seconds")
			for _ in range(sample_and_train_weight):
				# Sample training batch (MultiAgentBatch) from replay buffer.
				train_batch = concat_samples(self.local_replay_buffer.replay(
					batch_count=self.replay_batch_size, 
					cluster_overview_size=self.config.clustering_options['cluster_overview_size'],
				))

				# Postprocess batch before we learn on it
				post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
				train_batch = post_fn(train_batch, self.workers, self.config)

				# for policy_id, sample_batch in train_batch.policy_batches.items():
				#	 print(len(sample_batch["obs"]))
				#	 print(sample_batch.count)

				# Learn on training batch.
				# Use simple optimizer (only for multi-agent or tf-eager; all other
				# cases should use the multi-GPU optimizer, even if only using 1 GPU)
				if self.config.get("simple_optimizer") is True:
					train_results = train_one_step(self, train_batch)
				else:
					train_results = multi_gpu_train_one_step(self, train_batch)
				self._counters['training_steps'] += 1

				# Update replay buffer priorities.
				update_priorities(train_batch, train_results)

				last_update = self._counters[LAST_TARGET_UPDATE_TS]
				if cur_ts - last_update >= self.config.target_network_update_freq:
					to_update = self.workers.local_worker().get_policies_to_train()
					self.workers.local_worker().foreach_policy_to_train(
						lambda p, pid: pid in to_update and p.update_target()
					)
					self._counters[NUM_TARGET_UPDATES] += 1
					self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

				# Update weights and global_vars - after learning on the local worker -
				# on all remote workers.
				with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
					self.workers.sync_weights(global_vars=global_vars)

			############
			if self.use_siamese:
				anchor = self.triplet_buffer['anchor']
				positive = self.triplet_buffer['positive']
				negative = self.triplet_buffer['negative']

				if anchor and positive and negative:
					if 'siamese' not in train_results:
						train_results['siamese'] = {}

					self.siamese_model.train()
					self.optimizer.zero_grad()  # Clear gradients

					out_a = self.siamese_model(anchor)  # Forward pass
					out_p = self.siamese_model(positive)  # Forward pass
					out_n = self.siamese_model(negative)  # Forward pass

					loss = self.loss_fn(out_a, out_p, out_n)  # Compute the loss
					loss.backward()  # Backward pass (compute gradients)
					self.optimizer.step()  # Update parameters
					train_results['siamese']['siamese_loss'] = loss.item()

					last_siamese_update = self._counters['last_siamese_update']
					if cur_ts - last_siamese_update >= self.siamese_config["update_frequency"]:
						self.siamese_model.eval()
						start_cluster = time.time()
						print(f"Building clusters at timestep {cur_ts}")
						print(f"time to start building clusters: {start_cluster-start} seconds")

						self.local_replay_buffer.build_clusters(self.siamese_model)
						self._counters['last_siamese_update'] = cur_ts

						end_cluster = time.time()
						print(f"Clusters built at timestep {cur_ts}")
						print(f"Time spent building clusters: {end_cluster-start_cluster} seconds")

						for name, buffer in self.local_replay_buffer.replay_buffers.items():
							if name not in train_results['siamese']:
								train_results['siamese'][name] = {}
							train_results['siamese'][name]['num_clusters'] = len(buffer.get_available_clusters())
			############

			train_end = time.time()
			print(f"Time spent training: {train_end-train_start} seconds")
			if self.n_step_annealing_scheduler:
				self.update_n_steps()

		if self.config.clustering_options['collect_cluster_metrics']:
			add_buffer_metrics(train_results, self.local_replay_buffer)
		# Return all collected metrics for the iteration.
		return train_results
		