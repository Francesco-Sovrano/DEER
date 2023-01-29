# -*- coding: utf-8 -*-
from .full_world_all_agents_env import *


class FullWorldSomeAgents_Agent(FullWorldAllAgents_Agent):

	def __init__(self, n_of_other_agents, culture, env_config):
		super().__init__(n_of_other_agents, culture, env_config)

		state_dict = {
			"fc_junctions-64": gym.spaces.Box( # Junction properties and roads'
				low= float('-inf'),
				high= float('inf'),
				shape= (
					self.max_n_junctions_in_view,
					self.junction_feature_size + self.road_feature_size*self.env_config['max_roads_per_junction'], # junction.pos + junction.is_target + junction.is_source + junction.normalized_target_food + junction.normalized_source_food + (road.end + road.af_features)*max_roads_per_junction
				),
				dtype=np.float32
			),
			"fc_this_agent-8": gym.spaces.Box( # Agent features
				low= 0,
				high= 1,
				shape= (
					self.agent_state_size,
				),
				dtype=np.float32
			),
		}
		self.observation_space = gym.spaces.Dict(state_dict)
		# self._visible_road_network_junctions = None

	def get_visible_junctions(self, source_point, source_orientation):
		relative_jpos_vector = shift_and_rotate_vector([j.pos for j in self.road_network.junctions], source_point, source_orientation) #/ self.max_relative_coordinate
		return sorted(zip(relative_jpos_vector.tolist(),self.road_network.junctions), key=lambda x: x[0])

	def get_state(self, car_point=None, car_orientation=None):
		if car_point is None:
			car_point=self.car_point
		if car_orientation is None:
			car_orientation=self.car_orientation

		sorted_junctions = self.get_visible_junctions(car_point, car_orientation)
		roads_view_list = self.get_roads_view_list(sorted_junctions, car_point, car_orientation, self.max_n_junctions_in_view)
		roads_view = np.array(roads_view_list, dtype=np.float32)
		roads_view = np.reshape(roads_view_list, (-1, self.env_config['max_roads_per_junction']*self.road_feature_size))
		junctions_view_list = self.get_junction_view_list(sorted_junctions, car_point, car_orientation, self.max_n_junctions_in_view)
		junctions_view = np.array(junctions_view_list, dtype=np.float32)
		junctions_view = np.concatenate((junctions_view,roads_view), axis=-1)
		return {
			"fc_junctions-64": junctions_view,
			"fc_this_agent-8": np.array(self.get_agent_feature_list(), dtype=np.float32),
		}

	# def can_see(self, p):
	# 	if self.source_junction and p == self.source_junction.pos:
	# 		return True
	# 	if self.goal_junction and p == self.goal_junction.pos:
	# 		return True
	# 	return euclidean_distance(p, self.car_point) <= self.env_config['visibility_radius']


class FullWorldSomeAgents_GraphDelivery(FullWorldAllAgents_GraphDelivery):

	def __init__(self, config=None):
		super().__init__(config)
		self.agent_list = [
			FullWorldSomeAgents_Agent(self.num_agents-1, self.culture, self.env_config)
			for _ in range(self.num_agents)
		]
		self.action_space = self.agent_list[0].action_space
		base_space = self.agent_list[0].observation_space
		self.observation_space = gym.spaces.Dict({
			'all_agents_absolute_position_vector': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.num_agents,2), dtype=np.float32),
			'all_agents_absolute_orientation_vector': gym.spaces.Box(low=0, high=two_pi, shape=(self.num_agents,1), dtype=np.float32),
			'all_agents_relative_features_list': gym.spaces.Tuple(
				[base_space]*self.num_agents
			),
			# 'this_agent_id_mask': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.uint8),
			'this_agent_id': gym.spaces.Box(low=0, high=len(self.agent_list)-1, shape=(1,), dtype=np.uint8),
		})
		self.invisible_position_vec = np.array((float('inf'),float('inf')), dtype=np.float32)
		self.empty_agent_features = self.get_empty_state_recursively(base_space)
		self.agent_id_mask_dict = {}
		self.seed(config.get('seed',21))

	@staticmethod
	def get_empty_state_recursively(_obs_space):
		if isinstance(_obs_space, gym.spaces.Dict):
			return {
				k: FullWorldSomeAgents_GraphDelivery.get_empty_state_recursively(v)
				for k,v in _obs_space.spaces.items()
			}
		elif isinstance(_obs_space, gym.spaces.Tuple):
			return list(map(FullWorldSomeAgents_GraphDelivery.get_empty_state_recursively, _obs_space.spaces))
		return np.full(_obs_space.shape, EMPTY_FEATURE_PLACEHOLDER, dtype=_obs_space.dtype)

	def get_this_agent_id_mask(self, this_agent_id):
		agent_id_mask = self.agent_id_mask_dict.get(this_agent_id,None)
		if agent_id_mask is None:
			agent_id_mask = np.zeros((self.num_agents,), dtype=np.uint8)
			agent_id_mask[this_agent_id] = 1
			self.agent_id_mask_dict[this_agent_id] = agent_id_mask
		return agent_id_mask

	def build_state_with_comm(self, state_dict):
		if not state_dict:
			return state_dict

		all_agents_absolute_position_vector = np.array([
			self.agent_list[that_agent_id].car_point if that_agent_id in state_dict else self.invisible_position_vec
			for that_agent_id in range(self.num_agents)
		], dtype=np.float32)
		all_agents_absolute_orientation_vector = np.array([
			self.agent_list[that_agent_id].car_orientation if that_agent_id in state_dict else 0.
			for that_agent_id in range(self.num_agents)
		], dtype=np.float32)[:, np.newaxis]
		all_agents_relative_features_list = [
			state_dict.get(that_agent_id,self.empty_agent_features)
			for that_agent_id in range(self.num_agents)
		]
		return {
			this_agent_id: {
				'all_agents_absolute_position_vector': all_agents_absolute_position_vector,
				'all_agents_absolute_orientation_vector': all_agents_absolute_orientation_vector,
				'all_agents_relative_features_list': all_agents_relative_features_list,
				# 'this_agent_id_mask': self.get_this_agent_id_mask(this_agent_id),
				'this_agent_id': np.array([this_agent_id], dtype=np.uint8)
			}
			for this_agent_id in state_dict.keys()
		}

	def reset(self):
		return self.build_state_with_comm(super().reset())

	def step(self, action_dict):
		state_dict, reward_dict, terminal_dict, info_dict = super().step(action_dict)
		# assert not any(terminal_dict.values())
		return self.build_state_with_comm(state_dict), reward_dict, terminal_dict, info_dict
