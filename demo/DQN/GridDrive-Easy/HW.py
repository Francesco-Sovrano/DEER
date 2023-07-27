# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from deer.utils.workflow import train

from deer.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
from environments import *
from deer.models.dqn import TFAdaptiveMultiHeadDQN
from ray.rllib.models import ModelCatalog
# Register the models to use.
# ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
SELECT_ENV = "GridDrive-Easy"
# SELECT_ENV = "SpecialBreakoutNoFrameskip-v4"

CONFIG = XADQN_DEFAULT_CONFIG.copy()
CONFIG.update({
	# "model": { # this is for GraphDrive and GridDrive
	# 	"custom_model": "adaptive_multihead_network",
	# },
	# "preprocessor_pref": "rllib", # this prevents reward clipping on Atari and other weird issues when running from checkpoints
	"seed": 42, # This makes experiments reproducible.
	"rollout_fragment_length": 2**6, # Divide episodes into fragments of this many steps each during rollouts. Default is 1.
	"train_batch_size": 2**8, # Number of transitions per train-batch. Default is: 100 for TD3, 256 for SAC and DDPG, 32 for DQN, 500 for APPO.
	# "batch_mode": "truncate_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	###########################
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	'buffer_size': 2**14, # Size of the experience buffer. Default 50000
	"learning_starts": 2**14, # How many steps of the model to sample before learning starts.
	"prioritized_replay_alpha": 0.6,
	"prioritized_replay_beta": 0.4, # The smaller this is, the stronger is over-sampling
	"prioritized_replay_eps": 1e-6,
	###########################
	# 'lr': .0000625,
	# 'adam_epsilon': .00015,
	# 'exploration_config': {
	# 	'epsilon_timesteps': 200000,
	# 	'final_epsilon': 0.01,
	# },
	# 'timesteps_per_iteration': 10000,
	###########################
	"grad_clip": None, # no need of gradient clipping with huber loss
	"dueling": True,
	"double_q": True,
	"num_atoms": 21,
	"v_max": 2**5,
	"v_min": -1,
	##################################
	"buffer_options": {
		'priority_id': 'td_errors', # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_lower_limit': 0, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**14, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'clustering_xi': 1, # Let X be the minimum cluster's size, and q be the clustering_xi, then the cluster's size is guaranteed to be in [X, X+qX]. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_scheme": "HW", # Which scheme to use for building clusters. One of the following: "none", "positive_H", "H", "HW", "long_HW", "W", "long_W".
	"clustering_scheme_options": {
		"episode_window_size": 2**6, 
		"batch_window_size": 2**8, 
		"n_clusters": None,
	},
	"cluster_selection_policy": "min", # Which policy to follow when clustering_scheme is not "none" and multiple explanatory labels are associated to a batch. One of the following: 'random_uniform_after_filling', 'random_uniform', 'random_max', 'max', 'min', 'none'
	"cluster_with_episode_type": False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"collect_cluster_metrics": False, # Whether to collect metrics about the experience clusters. It consumes more resources.
	"ratio_of_samples_from_unclustered_buffer": 0, # 0 for no, 1 for full. Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(XADQNTrainer, CONFIG, SELECT_ENV, test_every_n_step=4e7, stop_training_after_n_step=4e7)
