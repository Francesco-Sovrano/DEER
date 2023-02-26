"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class CustomEnvironmentCallbacks(DefaultCallbacks):
	def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, env_index: int, **kwargs):
		# This requires environment-level horizons. Setting a global horizon won't trigger this function
		agent_id = max(episode.get_agents(), key=lambda x: len(episode._agent_reward_history[x]))
		last_info = episode.last_info_for(agent_id)
		if isinstance(last_info,dict) and "stats_dict" in last_info:
			for k,v in last_info["stats_dict"].items():
				episode.custom_metrics[k] = v

	def on_train_result(self, *, algorithm, result, **kwargs):
		# you can mutate the result dict to add new fields to return
		result["callback_ok"] = True 
