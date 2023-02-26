import gym
import numpy as np
import logging
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

from .Env_Builder import *
from .od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from .od_mstar3 import od_mstar
# from .GroupLock import Lock
from .Primal2Observer import Primal2Observer
from .Primal2Env import Primal2Env
from .Map_Generator import *

class GridPlanning(MultiAgentEnv):
	metadata = Primal2Env.metadata

	@staticmethod
	def preprocess_observation_dict(obs_dict):
		# for k,(state,vector) in obs_dict.items():
		# 	print(state.shape)
		# 	print(np.array(vector).shape)
		return {
			k: {
				'map': np.array(state, dtype=np.float32),
				'goal': np.array(vector, dtype=np.float32),
			}
			for k,(state,vector) in obs_dict.items()
		}

	def __init__(self, config):
		# super().__init__()
		self.env_config = config
		self.observation_size = config.get('observation_size',3)
		self.observer = Primal2Observer(
			observation_size=self.observation_size, 
			num_future_steps=config.get('num_future_steps',3),
		)
		self.map_generator = maze_generator(
			env_size = config.get('env_size',(10, 30)), 
			wall_components = config.get('wall_components',(3, 8)),
			obstacle_density = config.get('obstacle_density',(0.5, 0.7)),
		)
		self.num_agents = config.get('num_agents',1)
		self.IsDiagonal = config.get('IsDiagonal',False)
		self.frozen_steps = config.get('frozen_steps',0)
		self.isOneShot = config.get('isOneShot',False)
		self.time_limit = config.get('time_limit',5)
		
		self._env = Primal2Env(observer=self.observer, map_generator=self.map_generator, num_agents=self.num_agents, IsDiagonal=self.IsDiagonal, frozen_steps=self.frozen_steps, isOneShot=self.isOneShot)
		self._agent_ids = set(range(1, self.num_agents + 1))

		self.observation_space = gym.spaces.Dict({
			'map': gym.spaces.Box(low=-255, high=255, shape=(11, self.observation_size, self.observation_size), dtype=np.float32),
			'goal': gym.spaces.Box(low=-255, high=255, shape=(3,), dtype=np.float32),
		})
		self.action_space = gym.spaces.Discrete(9 if self.IsDiagonal else 5)
	
	def reset(self):
		self._env._reset()
		obs = self._env._observe()
		# print(obs[1][0].shape, obs[1][1].shape)
		self.last_obs = obs
		self._step_count = 1
		return self.preprocess_observation_dict(obs)

	def get_why_explanation(self, new_pos, old_astar_pos, is_valid_action=True):
		explanation_list = []
		if not is_valid_action:
			explanation_list.append('invalid_action')
		# print(new_pos, old_astar_pos, new_pos == old_astar_pos)
		if new_pos == old_astar_pos:
			explanation_list.append('acting_as_A*')
		if old_astar_pos is None:
			explanation_list.append('wrong_path')
		# if new_pos == old_mstar_pos:
		# 	explanation_list.append('acting_as_M*')
		if not explanation_list:
			explanation_list = ['acting_differently']
		return explanation_list

	def get_reward(self, standing_on_goal, is_valid_action):
		if standing_on_goal:
			return 1
		# if not is_valid_action:
		# 	return -0.01
		return 0

	# Executes an action by an agent
	def step(self, action_dict):
		self._step_count += 1
		# print(list(action_dict.keys()), list(self.last_obs.keys()))
		living_agents = list(action_dict.keys())
		valid_action_dict = {
			k: action_dict[k] in self._env.listValidActions(k, self.last_obs[k])
			for k in living_agents
		}
		# print(action_dict[1])
		astar_path_iter = (self._env.expert_until_first_goal(agent_ids=[i]) for i in living_agents)
		astar_pos_dict = {
			i: path[0] if path is not None else None
			for i,path in zip(living_agents,astar_path_iter)
		}
		# path_list = self._env.expert_until_first_goal(agent_ids=living_agents, time_limit=self.time_limit)
		# if path_list and len(path_list) == len(living_agents):
		# 	mstar_pos_dict = {
		# 		k: path_list[k][0]
		# 		for k in living_agents
		# 	}
		# else:
		# 	mstar_pos_dict = {
		# 		k: None
		# 		for k in living_agents
		# 	}

		_obs,rew = self._env.step_all(action_dict)
		obs = self.preprocess_observation_dict(
			_obs 
			if len(living_agents) == len(self._agent_ids) else 
			{k:_obs[k] for k in living_agents}
		)

		done = {
			k: self._env.isStandingOnGoal[k]
			for k in living_agents
		}

		positions = self._env.getPositions()
		rew = {
			k: self.get_reward(self._env.isStandingOnGoal[k], valid_action_dict[k])
			for k in living_agents
		}
		throughput = sum((1 if self._env.isStandingOnGoal[k] else 0 for k in self._agent_ids))#/len(self._agent_ids)
		info = {
			k: {
				'explanation': {
					'why': self.get_why_explanation(
						positions[k], 
						# mstar_pos_dict[k], 
						astar_pos_dict[k], 
						is_valid_action=valid_action_dict[k],
					)
				},
				'stats_dict': {
					"throughput": throughput,
					"living_agents": len(living_agents)
				}
			}
			for k in living_agents
		}
		
		# print(info)
		done["__all__"] = all(done.values()) or self._step_count == self.env_config.get('horizon',float('inf'))
		# rew["__all__"] = np.sum([r for r in step_r.reward.values()])
		self.last_obs = _obs
		return obs, rew, done, info

	def render(self,mode='human'):
		return self._env._render(self._env_config.get('render'))
