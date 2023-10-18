# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = "1" # Tell PyTorch to use only 2 CPUs
import json
import ray
from deer.utils.workflow import train
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from environment import *

from ray.rllib.models import ModelCatalog
from deer.models import get_model_catalog_dict

def get_algorithm_by_name(alg_name):
	#### DQN
	if alg_name == 'dqn':
		from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
		return DQNConfig, DQN
	if alg_name == 'xadqn':
		from deer.agents.xadqn import XADQN, XADQNConfig
		return XADQNConfig, XADQN
	#### DDPG
	if alg_name == 'ddpg':
		from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
		return DDPGConfig, DDPG
	if alg_name == 'xaddpg':
		from deer.agents.xaddpg import XADDPG, XADDPGConfig
		return XADDPGConfig, XADDPG
	#### TD3
	if alg_name == 'td3':
		from ray.rllib.algorithms.td3.td3 import TD3, TD3Config
		return TD3Config, TD3
	if alg_name == 'xatd3':
		from deer.agents.xaddpg import XATD3, XATD3Config
		return XATD3Config, XATD3
	#### SAC
	if alg_name == 'sac':
		from ray.rllib.algorithms.sac.sac import SAC, SACConfig
		return SACConfig, SAC
	if alg_name == 'xasac':
		from deer.agents.xasac import XASAC, XASACConfig
		return XASACConfig, XASAC

import sys
ALG_NAME = sys.argv[1]
CONFIG, TRAINER = get_algorithm_by_name(ALG_NAME)
ENVIRONMENT = sys.argv[2]
EXPERIMENT = None if sys.argv[3].lower()=='none' else sys.argv[3]
TEST_EVERY_N_STEP = int(float(sys.argv[4]))
STOP_TRAINING_AFTER_N_STEP = int(float(sys.argv[5]))
EPISODES_PER_TEST = int(float(sys.argv[6]))
SAVE_GIFS = sys.argv[7].lower() == 'true'
CENTRALISED_TRAINING = sys.argv[8].lower() == 'true'
NUM_AGENTS = int(float(sys.argv[9]))
OPTIONS = {}
if len(sys.argv) > 10:
	print('Updating options..')
	OPTIONS = json.loads(' '.join(sys.argv[10:]))
	print('New options:', json.dumps(OPTIONS, indent=4))
OPTIONS["callbacks"] = CustomEnvironmentCallbacks

for k,v in get_model_catalog_dict(ALG_NAME, OPTIONS.get("framework","tf")).items():
	ModelCatalog.register_custom_model(k, v)

# Setup MARL training strategy: centralised or decentralised
if not CENTRALISED_TRAINING:
	policy_graphs = {
		f'agent-{i}'
		for i in range(number_of_agents)
	}
	policy_mapping_fn = lambda agent_id: f'agent-{agent_id}'
else:
	# policy_graphs = {DEFAULT_POLICY_ID: (None, obs_space, act_space, CONFIG)}
	policy_graphs = {DEFAULT_POLICY_ID}
	policy_mapping_fn = lambda agent_id: DEFAULT_POLICY_ID

OPTIONS["multiagent"] = {
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
print('Config:', OPTIONS)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(
	ignore_reinit_error=False, 
	include_dashboard=False, 
	log_to_driver=False, 
	num_cpus=1,
	_temp_dir='/var/tmp',
)

train(
	TRAINER, 
	CONFIG, 
	OPTIONS,
	ENVIRONMENT, 
	experiment=EXPERIMENT, 
	test_every_n_step=TEST_EVERY_N_STEP, 
	stop_training_after_n_step=STOP_TRAINING_AFTER_N_STEP,
	save_gif=SAVE_GIFS, delete_screens_after_making_gif=True, compress_gif=True, n_episodes=EPISODES_PER_TEST, with_log=False
)
