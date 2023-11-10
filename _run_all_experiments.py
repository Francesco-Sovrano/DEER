import sys
import json
import copy
import shlex
import numpy as np
import argparse
from random import randint
from time import sleep
import os
import subprocess

############################################################################################
############################################################################################

parser = argparse.ArgumentParser(description='Check servers')
parser.add_argument('-y', '--restart_if_dead', dest='restart_if_dead', action='store_true')
parser.add_argument('-t', '--max_trials_count', dest='max_trials_count', type=int, default=float('inf'))
parser.set_defaults(restart_if_dead=False)
ARGS = parser.parse_args()

restart_if_dead = ARGS.restart_if_dead
max_trials_count = ARGS.max_trials_count

############################################################################################
############################################################################################

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

############################################################################################
############################################################################################

training_steps = 2**16
train_batch_size = 2**8
stop_training_after_n_step = training_steps*train_batch_size
save_n_checkpoints = 1
save_gifs = True
episodes_per_test = 10
test_every_n_step = int(np.ceil(stop_training_after_n_step/save_n_checkpoints))
centralised_training = True

get_experiment_id = lambda *arg: f"deer4nstep-task_{'-'.join(map(str,arg))}"

config_list = []

default_options = {
	"no_done_at_end": False, # IMPORTANT: if set to True it allows lifelong learning with decent bootstrapping
	
	"gamma": 0.99,
	# "num_workers": 4, # Number of rollout worker actors to create for parallel sampling. Setting this to 0 will force rollouts to be done in the  actor.
	# "num_envs_per_worker": 1, # Number of environments to evaluate vector-wise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	# "vf_loss_coeff": 1.0, # Coefficient of the value function loss. IMPORTANT: you must tune this if you set vf_share_layers=True inside your model's config.
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"seed": 42, # This makes experiments reproducible.
	# "multiagent": {
	# 	# Optional list of policies to train, or None for all policies.
	# 	"policies_to_train": None,
	# 	# When replay_mode=lockstep, RLlib will replay all the agent transitions at a particular timestep together in a batch. This allows the policy to implement differentiable shared computations between agents it controls at that timestep. When replay_mode=independent, transitions are replayed independently per policy.
	# 	"replay_mode": "independent", # XAER does not support "lockstep", yet
	# 	# Which metric to use as the "batch size" when building a MultiAgentBatch. The two supported values are: env_steps: Count each time the env is "stepped" (no matter how many multi-agent actions are passed/how many multi-agent observations have been returned in the previous step), agent_steps: Count each individual agent step as one step.
	# 	"count_steps_by": "agent_steps", # XAER does not support "env_steps"?
	# },
	# "batch_dropout_rate": 0.5, # Probability of dropping a state transition before adding it to the experience buffer. Set this to any value greater than zero to randomly drop state transitions
	###########################
	"batch_mode": "truncate_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"rollout_fragment_length": 2**6, # Divide episodes into fragments of this many steps each during rollouts. Default is 1.
	"train_batch_size": train_batch_size, # Number of 'n_step' transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for SAC, 500 for APPO.
	###########################
	"min_train_timesteps_per_iteration": 1,
	# "num_steps_sampled_before_learning_starts": 2**9, # How many steps of the model to sample before learning starts.
}
xa_default_options = {
	##############################
	"buffer_options": {
		"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
		"centralised_buffer": True, # for MARL
		# 'global_size': EXPERIENCE_BUFFER_SIZE, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
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
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In SAC default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far. 
		'stationarity_window_size': None, # If lower than float('inf') and greater than 0, then the stationarity_window_size W is used to guarantee that every W training-steps the buffer is emptied from old state transitions.
		'stationarity_smoothing_factor': 1, # A number >= 1, where 1 means no smoothing. The larger this number, the smoother the transition from a stationarity stage to the next one. This should help avoiding experience buffers saturated by one single episode during a stage transition. The optimal value should be equal to ceil(HORIZON*number_of_agents/EXPERIENCE_BUFFER_SIZE)*stationarity_window_size.
		#################
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_options": {
		'clustering_scheme': None,
		# 'clustering_scheme': [
		# 	'Why',
		# 	# 'Who',
		# 	'How_Well',
		# 	# 'How_Fair',
		# 	# 'Where',
		# 	# 'What',
		# 	# 'How_Many'
		# 	# 'UWho',
		# 	# 'UWhich_CoopStrategy',
		# ],
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
default_experiment_options = default_options
default_experiment_options = copy_dict_and_update(default_experiment_options, xa_default_options)

############################################################################################
############################################################################################

default_algorithm = 'DQN'
algorithm_options = {
	"framework": "torch",
	"model": {
		"custom_model": "adaptive_multihead_network",
		# "custom_model_config": {
		# 	"comm_range": VISIBILITY_RADIUS,
		# 	"add_nonstationarity_correction": False, # Experience replay in MARL may suffer from non-stationarity. To avoid this issue a solution is to condition each agent’s value function on a fingerprint that disambiguates the age of the data sampled from the replay memory. To stabilise experience replay, it should be sufficient if each agent’s observations disambiguate where along this trajectory the current training sample originated from. # cit. [2017]Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning
		# },
	},
	
	# "horizon": 2**5,

	"td_error_loss_fn": "mse",
	"grad_clip": 2,
	"lr": 5e-4,
	"hiddens": [256,256],
	"target_network_update_freq": 500,

	"dueling": True,
	"double_q": True,
	"noisy": True,
	"sigma0": 0.5,
	# "num_atoms": 51,
	# "v_max": 2**5,
	# "v_min": -1,

	# "n_step": 3,
	# "n_step_sampling_procedure": 'random_sampling', # either None or 'random_sampling'
	# "n_step_annealing_scheduler": {
	# 	'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
	# 	'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
	# 		'schedule_timesteps': TRAINING_STEPS//2,
	# 		'final_p': 1, # final n-step
	# 		'framework': None,
	# 		'initial_p': 10 # initial n-step
	# 	}
	# },
}
default_environment_list = [
	'GridDrive-Medium', 
	'GridDrive-Hard'
]
number_of_agents_list = [1]


for default_environment in default_environment_list:
	for num_agents in number_of_agents_list:
		# Experiment 1
		experiment1_options = default_experiment_options
		experiment1_options = copy_dict_and_update(experiment1_options, algorithm_options)
		experiment1_options = copy_dict_and_update(experiment1_options, {
			"n_step": 1,
		})
		# Experiment 2
		experiment2_options = copy_dict_and_update(experiment1_options, {
			"n_step": 10,
		})
		# Experiment 3
		experiment3_options = copy_dict_and_update(experiment1_options, {
			"n_step_sampling_procedure": 'random_sampling', # either None or 'random_sampling'
			"n_step_annealing_scheduler": {
				'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
				'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
					'schedule_timesteps': training_steps//2,
					'final_p': 1, # final n-step
					'framework': None,
					'initial_p': 10 # initial n-step
				}
			},
		})
		# Experiment 4
		experiment4_options = copy_dict_and_update(experiment1_options, {
			"n_step_sampling_procedure": None, # either None or 'random_sampling'
			"n_step_annealing_scheduler": {
				'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
				'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
					'schedule_timesteps': training_steps//2,
					'final_p': 1, # final n-step
					'framework': None,
					'initial_p': 10 # initial n-step
				}
			},
		})
		## build experiments
		eid = get_experiment_id(default_environment.replace('/','_'), num_agents)
		config_list += [
			#
			('XA'+default_algorithm, default_environment, f'exp1-{default_algorithm}-{eid}', num_agents, experiment1_options),
			#
			('XA'+default_algorithm, default_environment, f'exp2-{default_algorithm}-{eid}', num_agents, experiment2_options),
			#
			('XA'+default_algorithm, default_environment, f'exp3-{default_algorithm}-{eid}', num_agents, experiment3_options),
			#
			('XA'+default_algorithm, default_environment, f'exp4-{default_algorithm}-{eid}', num_agents, experiment4_options),
		]

############################################################################################
############################################################################################

default_algorithm = 'SAC'
algorithm_options = {
	"framework": "torch",
	"model": {
		"custom_model": "adaptive_multihead_network",
		# "custom_model_config": {
		# 	"comm_range": VISIBILITY_RADIUS,
		# 	"add_nonstationarity_correction": False, # Experience replay in MARL may suffer from non-stationarity. To avoid this issue a solution is to condition each agent’s value function on a fingerprint that disambiguates the age of the data sampled from the replay memory. To stabilise experience replay, it should be sufficient if each agent’s observations disambiguate where along this trajectory the current training sample originated from. # cit. [2017]Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning
		# },
	},
	
	# "horizon": 2**5,
	# "gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"tau": 1e-4,

	"grad_clip": 2,
}
default_environment_list = [
	'GraphDrive-Medium', 
	'GraphDrive-Hard'
]
number_of_agents_list = [1]


for default_environment in default_environment_list:
	for num_agents in number_of_agents_list:
		# Experiment 1
		experiment1_options = default_experiment_options
		experiment1_options = copy_dict_and_update(experiment1_options, algorithm_options)
		experiment1_options = copy_dict_and_update(experiment1_options, {
			"n_step": 1,
		})
		# Experiment 2
		experiment2_options = copy_dict_and_update(experiment1_options, {
			"n_step": 10,
		})
		# Experiment 3
		experiment3_options = copy_dict_and_update(experiment1_options, {
			"n_step_sampling_procedure": 'random_sampling', # either None or 'random_sampling'
			"n_step_annealing_scheduler": {
				'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
				'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
					'schedule_timesteps': training_steps//2,
					'final_p': 1, # final n-step
					'framework': None,
					'initial_p': 10 # initial n-step
				}
			},
		})
		# Experiment 4
		experiment4_options = copy_dict_and_update(experiment1_options, {
			"n_step_sampling_procedure": None, # either None or 'random_sampling'
			"n_step_annealing_scheduler": {
				'fn': 'LinearSchedule', # One of these: 'ConstantSchedule', 'PiecewiseSchedule', 'ExponentialSchedule', 'PolynomialSchedule'. 
				'args': { # For details about args see: https://docs.ray.io/en/latest/rllib/package_ref/utils.html?highlight=LinearSchedule#built-in-scheduler-components
					'schedule_timesteps': training_steps//2,
					'final_p': 1, # final n-step
					'framework': None,
					'initial_p': 10 # initial n-step
				}
			},
		})
		## build experiments
		eid = get_experiment_id(default_environment.replace('/','_'), num_agents)
		config_list += [
			#
			('XA'+default_algorithm, default_environment, f'exp1-{default_algorithm}-{eid}', num_agents, experiment1_options),
			#
			('XA'+default_algorithm, default_environment, f'exp2-{default_algorithm}-{eid}', num_agents, experiment2_options),
			#
			('XA'+default_algorithm, default_environment, f'exp3-{default_algorithm}-{eid}', num_agents, experiment3_options),
			#
			('XA'+default_algorithm, default_environment, f'exp4-{default_algorithm}-{eid}', num_agents, experiment4_options),
		]

############################################################################################
############################################################################################

for experiment in config_list:
	algorithm,environment,experiment_id,number_of_agents,options = experiment
	options_string = json.dumps(options)
	print('Running:', algorithm, environment, experiment_id, options_string)
	arg_list = list(map(str, ["./setup_n_run.sh", algorithm.lower(), environment, experiment_id, test_every_n_step, stop_training_after_n_step, episodes_per_test, save_gifs, centralised_training, number_of_agents, options_string]))
	with open('/dev/null', 'w') as devnull:
		subprocess.Popen(arg_list, stdout=devnull, stderr=devnull)
