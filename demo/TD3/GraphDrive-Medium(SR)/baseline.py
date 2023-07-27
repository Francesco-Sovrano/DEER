# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from deer.utils.workflow import train

from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from environments import *
from ray.rllib.models import ModelCatalog
from deer.models.ddpg import TFAdaptiveMultiHeadDDPG
# ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDDPG)

# SELECT_ENV = "CescoDrive-V1"
SELECT_ENV = "GraphDrive-Medium-Sparse"

CONFIG = TD3_DEFAULT_CONFIG.copy()
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
	"prioritized_replay_alpha": 0.6,
	"prioritized_replay_beta": 0.4, # The smaller, the stronger is over-sampling
	"prioritized_replay_eps": 1e-6,
	"learning_starts": 2**14, # How many steps of the model to sample before learning starts.
	###########################
	"gamma": 0.999, # We use an higher gamma to extend the MDP's horizon; optimal agency on GraphDrive requires a longer horizon.
	"tau": 1e-4,
	##################################
})
CONFIG["callbacks"] = CustomEnvironmentCallbacks

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(TD3Trainer, CONFIG, SELECT_ENV, test_every_n_step=4e7, stop_training_after_n_step=4e7)
