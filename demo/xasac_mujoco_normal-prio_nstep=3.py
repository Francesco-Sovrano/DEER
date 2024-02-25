# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
import copy
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from deer.utils.workflow import train
from deer.agents.xasac import XASAC, XASACConfig

# Import environments
import sys
sys.path.append(".")  # Adds the parent directory to the system path.
sys.path.append("..")  # Adds the parent directory to the system path.
from environment import *

SELECTED_ENV = "Humanoid-v4"
EXPERIENCE_BUFFER_SIZE = 2**12
CENTRALISED_TRAINING = True
TRAINING_STEPS = 2**17
HORIZON = 2**10

default_options = {
	"framework": "torch",
	"model": {
		"custom_model": "adaptive_multihead_network",
		# "custom_model_config": {
		# 	"comm_range": VISIBILITY_RADIUS,
		# 	"add_nonstationarity_correction": False, # Experience replay in MARL may suffer from non-stationarity. To avoid this issue a solution is to condition each agent’s value function on a fingerprint that disambiguates the age of the data sampled from the replay memory. To stabilise experience replay, it should be sufficient if each agent’s observations disambiguate where along this trajectory the current training sample originated from. # cit. [2017]Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning
		# },
	},
	"_disable_env_checking": True,
	"no_done_at_end": False, # IMPORTANT: if set to True it allows lifelong learning with decent bootstrapping
	"horizon": HORIZON, # Number of steps after which the episode is forced to terminate. Defaults to `env.spec.max_episode_steps` (if present) for Gym envs.
	# "num_workers": 4, # Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the  actor.
	# "num_envs_per_worker": 1, # Number of environments to evaluate vector-wise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	# "vf_loss_coeff": 1.0, # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model's config.
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDelivery requires a longer horizon.
	"seed": 42, # This makes experiments reproducible.
	# "multiagent": {
	# 	# Optional list of policies to train, or None for all policies.
	# 	"policies_to_train": None,
	# 	# When replay_mode=lockstep, RLlib will replay all the agent transitions at a particular timestep together in a batch. This allows the policy to implement differentiable shared computations between agents it controls at that timestep. When replay_mode=independent, transitions are replayed independently per policy.
	# 	"replay_mode": "independent", # XAER does not support "lockstep", yet
	# 	# Which metric to use as the "batch size" when building a MultiAgentBatch. The two supported values are: env_steps: Count each time the env is "stepped" (no matter how many multi-agent actions are passed/how many multi-agent observations have been returned in the previous step), agent_steps: Count each individual agent step as one step.
	# },
	# "batch_dropout_rate": 0.5, # Probability of dropping a state transition before adding it to the experience buffer. Set this to any value greater than zero to randomly drop state transitions
	###########################
	"batch_mode": "truncate_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"rollout_fragment_length": 1, # Divide episodes into fragments of this many steps each during rollouts. Default is 1.
	"train_batch_size": 2**8, # Number of 'n_step' transitions per train-batch. Default is: 100 for TD3, 256 for TD3 and DDPG, 32 for TD3, 500 for APPO.
	###########################
	"min_train_timesteps_per_iteration": 1,
	"num_steps_sampled_before_learning_starts": 2**8, # How many steps of the model to sample before learning starts.
	"count_steps_by": "agent_steps", # XAER does not support "env_steps"?
}
algorithm_options = {
	"grad_clip": None, # no need of gradient clipping with huber loss
	# "dueling": True,
	# "double_q": True,
	# "num_atoms": 21,
	# "v_max": 2**5,
	# "v_min": -1,
	# "n_step": 10,
	"n_step_random_sampling": True, # a Boolean
	"n_step_annealing_scheduler": {
		'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
		'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
			'schedule_timesteps': TRAINING_STEPS//2,
			'final_p': 1, # final n-step
			'framework': None,
			'initial_p': 10 # initial n-step
		}
	},
}
xa_default_options = {
	##############################
	"buffer_options": {
		"siamese_num_clusters": 50,
		"num_samples_to_recluster": 2000,
		"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
		"centralised_buffer": True, # for MARL
		'global_size': EXPERIENCE_BUFFER_SIZE, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'priority_id': 'td_errors',
		'priority_lower_limit': 0,
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		#################
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (whose number depends on the clustering scheme) of the experience buffer. Every batch has size 'sample_batch_size' (default is 1).
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_level_weighting': True, # Whether to use cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 4,
		#################
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In TD3 default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far. 
		'stationarity_window_size': None, # If lower than float('inf') and greater than 0, then the stationarity_window_size W is used to guarantee that every W training-steps the buffer is emptied from old state transitions.
		'stationarity_smoothing_factor': 1, # A number >= 1, where 1 means no smoothing. The larger this number, the smoother the transition from a stationarity stage to the next one. This should help avoiding experience buffers saturated by one single episode during a stage transition. The optimal value should be equal to ceil(HORIZON*number_of_agents/EXPERIENCE_BUFFER_SIZE)*stationarity_window_size.
		#################
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_options": {
		'clustering_scheme': None,
		'clustering_scheme': [
			'Why',
			# 'Who',
			'How_Well',
			# 'How_Fair',
			# 'Where',
			# 'What',
			# 'How_Many'
			# 'UWho',
			# 'UWhich_CoopStrategy',
		],
		"clustering_scheme_options": {
			"n_clusters": {
				"who": 4,
				# "why": 8,
				# "what": 8,
			},
			"default_n_clusters": 8,
			"frequency_independent_clustering": False, # Setting this to True can be memory expensive, especially for WHO explanations
			"agent_action_sliding_window": 2**3,
			"episode_window_size": 2**6, 
			"batch_window_size": 2**8, 
			"training_step_window_size": 2**2,
		},
		"cluster_selection_policy": "min", # Which policy to follow when clustering_scheme is not "none" and multiple explanatory labels are associated to a batch. One of the following: 'random_uniform_after_filling', 'random_uniform', 'random_max', 'max', 'min', 'none'
		"cluster_with_episode_type": False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
		"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
		"collect_cluster_metrics": True, # Whether to collect metrics about the experience clusters. It consumes more resources.
		"ratio_of_samples_from_unclustered_buffer": 0, # 0 for no, 1 for full. Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
	},
}
siamese_options = {
	"siamese_config": {
		"use_siamese": True,
		"training_batch_size": 52,
		"buffer_size": 520,
		"update_frequency": 10000,
		"embedding_size": 16,
	},
	"_disable_preprocessor_api": True,
}

def copy_dict_and_update(d,u):
	new_dict = copy.deepcopy(d)
	new_dict.update(u)
	return new_dict

def copy_dict_and_update_with_key(d,k,u):
	new_dict = copy.deepcopy(d)
	if k not in new_dict:
		new_dict[k] = {}
	new_dict[k].update(u)
	return new_dict

CONFIG = default_options
CONFIG = copy_dict_and_update(CONFIG, xa_default_options)
CONFIG = copy_dict_and_update(CONFIG, algorithm_options)
CONFIG = copy_dict_and_update(CONFIG, siamese_options)
CONFIG["callbacks"] = CustomEnvironmentCallbacks

# Setup MARL training strategy: centralised or decentralised
env = _global_registry.get(ENV_CREATOR, SELECTED_ENV)(CONFIG.get("env_config",{}))
obs_space = env.observation_space
act_space = env.action_space
if not CENTRALISED_TRAINING:
	policy_graphs = {
		f'agent-{i}'
		for i in range(number_of_agents)
	}
	policy_mapping_fn = lambda agent_id, episode, worker: f'agent-{agent_id}'
else:
	# policy_graphs = {DEFAULT_POLICY_ID: (None, obs_space, act_space, CONFIG)}
	policy_graphs = {DEFAULT_POLICY_ID}
	policy_mapping_fn = lambda agent_id, episode, worker: DEFAULT_POLICY_ID

CONFIG["multiagent"] = {
	"policies": policy_graphs,
	"policy_mapping_fn": policy_mapping_fn,
	# Optional list of policies to train, or None for all policies.
	"policies_to_train": None,
	# Optional function that can be used to enhance the local agent
	# observations to include more state.
	# See rllib/evaluation/observation_function.py for more info.
	"observation_fn": None,
	# When replay_mode=lockstep, RLlib will replay all the agent
	# transitions at a particular timestep together in a batch. This allows
	# the policy to implement differentiable shared computations between
	# agents it controls at that timestep. When replay_mode=independent,
	# transitions are replayed independently per policy.
	"replay_mode": "independent", # XAER does not support "lockstep", yet
	# Which metric to use as the "batch size" when building a
	# MultiAgentBatch. The two supported values are:
	# env_steps: Count each time the env is "stepped" (no matter how many
	#   multi-agent actions are passed/how many multi-agent observations
	#   have been returned in the previous step).
	# agent_steps: Count each individual agent step as one step.
	# "count_steps_by": "agent_steps", # XAER does not support "env_steps"?
}
print('Config:', CONFIG)

# Register models
from ray.rllib.models import ModelCatalog
from deer.models import get_model_catalog_dict
for k,v in get_model_catalog_dict('sac', CONFIG.get("framework",'torch')).items():
	ModelCatalog.register_custom_model(k, v)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(XASAC, XASACConfig, CONFIG, SELECTED_ENV, test_every_n_step=TRAINING_STEPS*CONFIG["train_batch_size"], stop_training_after_n_step=TRAINING_STEPS*CONFIG["train_batch_size"])
