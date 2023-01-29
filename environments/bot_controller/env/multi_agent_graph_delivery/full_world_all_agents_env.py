# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import numpy as np
import json
from more_itertools import unique_everseen
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import cv2
from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.text import Text
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from ...utils.geometry import *
from .lib.multi_agent_road_network import MultiAgentRoadNetwork
from .lib.multi_agent_road_cultures import *

import logging
logger = logging.getLogger(__name__)

# import time

normalize_delivery_count = lambda value, max_value: np.clip(value, 0, max_value)/max_value
is_source_junction = lambda j: j.is_available_source
is_target_junction = lambda j: j.is_available_target
EMPTY_FEATURE_PLACEHOLDER = 0

class FullWorldAllAgents_Agent:

	def seed(self, seed=None):
		# logger.warning(f"Setting random seed to: {seed}")
		self.np_random = seeding.np_random(seed)[0]
		return [seed]

	def __init__(self, n_of_other_agents, culture, env_config):
		# super().__init__()
		
		self.culture = culture
		self.n_of_other_agents = n_of_other_agents
		self.env_config = env_config
		self.build_action_list = self.env_config.get('build_action_list', False) or self.env_config.get('build_joint_action_list', False)
		self.max_n_junctions_in_view = self.env_config['junctions_number']
		self.terminate_if_wrong_behaviour = self.env_config.get('terminate_if_wrong_behaviour',False)
		self.max_depth_searching_for_closest_target = self.env_config.get('max_depth_searching_for_closest_target',3)
		# self.max_relative_coordinate = 2*self.env_config['max_dimension']
		self.reward_fn = eval(f'self.{self.env_config.get("reward_fn","frequent")}_reward_default')
		self.fairness_reward_fn = eval(f'self.{self.env_config["fairness_reward_fn"]}_fairness_reward') if self.env_config.get("fairness_reward_fn",None) else lambda x: 0
		self.fairness_type_fn = eval(f'self.{self.env_config["fairness_type_fn"]}_fairness_type') if self.env_config.get("fairness_type_fn",None) else lambda: 'unknown'
		
		self.obs_road_features = len(culture.properties) if culture else 0  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(culture.agent_properties) if culture else 0  # Number of binary CAR features in Hard Culture (excluded speed)
		# Spaces
		self.discrete_action_space = self.env_config.get('n_discrete_actions',None)
		self.decides_speed = False
		if self.discrete_action_space:
			self.allowed_orientations = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			if not self.decides_speed:
				self.allowed_speeds = [1]
			else:
				self.allowed_speeds = np.linspace(-1, 1, self.env_config['n_discrete_actions']).tolist()
			self.action_space = gym.spaces.Discrete(len(self.allowed_orientations)*len(self.allowed_speeds))
		else:
			if self.decides_speed:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1+1,), dtype=np.float32)
			else:
				self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
		self.junction_feature_size = 2 + 1 + 1 + 1 + 1 # junction.pos + junction.is_target + junction.is_source + junction.normalized_target_food + junction.normalized_source_food 
		self.road_feature_size = 2 + self.obs_road_features # road.end + road.af_features
		state_dict = {
			"fc_junctions-64": gym.spaces.Box( # Junction properties and roads'
				low= float('-inf'),
				high= float('inf'),
				shape= (
					self.max_n_junctions_in_view,
					self.junction_feature_size + self.road_feature_size*self.env_config['max_roads_per_junction'],
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
		if self.n_of_other_agents > 0:
			state_dict["fc_other_agents-16"] = gym.spaces.Box( # permutation invariant
				low= -1,
				high= 1,
				shape= (
					self.n_of_other_agents,
					2 + 1 + self.agent_state_size,
				), # agent.position + agent.orientation + agent.features
				dtype=np.float32
			)
		self.observation_space = gym.spaces.Dict(state_dict)

		self._empty_junction = np.full(self.junction_feature_size, EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		self._empty_road = np.full(self.road_feature_size, EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		self._empty_junction_roads = np.full((self.env_config['max_roads_per_junction'], self.road_feature_size), EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)
		if self.n_of_other_agents > 0:
			self._empty_agent = np.full(self.observation_space['fc_other_agents-16'].shape[1:], EMPTY_FEATURE_PLACEHOLDER, dtype=np.float32)

	def reset(self, car_point, agent_id, road_network, other_agent_list):
		self.agent_id = agent_id
		self.road_network = road_network
		self.other_agent_list = other_agent_list
		# car position
		self.car_point = car_point
		self.car_orientation = (self.np_random.random()*two_pi) % two_pi # in [0,2*pi)
		self.previous_closest_junction = None
		self.closest_junction = self.road_network.junction_dict[car_point]
		self.closest_road = None
		# speed
		self.car_speed = self.env_config['min_speed'] if self.decides_speed else self.env_config['max_speed']
		#####
		self.last_closest_road = None
		self.last_closest_junction = None
		self.source_junction = None
		self.goal_junction = None
		# init concat variables
		self.last_action_mask = None
		self.is_dead = False
		self.has_food = is_source_junction(self.closest_junction)
		# self.steps_in_junction = 0
		self.step = 1
		# self.idle = False
		self.last_reward = None
		if self.build_action_list:
			self.action_list = []

		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False
		self.stuck_in_junction = False
		self.steps_stuck_in_junction = 0

	def can_see(self, p):
		return True

	@property
	def agent_state_size(self):
		agent_state_size = 4
		if self.decides_speed:
			agent_state_size += 1
		if self.culture:
			agent_state_size += self.obs_car_features
		return agent_state_size

	@property
	def task_completion(self):
		deliveries_to_do = self.env_config['target_junctions_number']*self.env_config['max_deliveries_per_target']
		return self.road_network.deliveries/deliveries_to_do

	def get_agent_feature_list(self):
		agent_state = [
			self.is_in_junction(self.car_point),
			self.has_food,
			self.is_dead,
			self.task_completion,
			# self.step_gain, # in (0,1]
		]
		if self.decides_speed:
			agent_state.append(self.car_speed/self.env_config['max_speed']) # normalised speed # in [0,1]
		if self.culture:
			agent_state.extend(self.agent_id.binary_features(as_tuple=True))
		assert len(agent_state)==self.agent_state_size
		return agent_state

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
		state_dict = {
			"fc_junctions-64": junctions_view,
			"fc_this_agent-8": np.array(self.get_agent_feature_list(), dtype=np.float32),
		}
		if self.n_of_other_agents > 0:
			agent_neighbourhood_view = self.get_neighbourhood_view(car_point, car_orientation)
			state_dict["fc_other_agents-16"] = np.array(agent_neighbourhood_view, dtype=np.float32)
		return state_dict

	# @property
	# def step_gain(self):
	# 	return 1/max(1,np.log(self.step)) # in (0,1]

	@property
	def step_seconds(self):
		return self.np_random.exponential(scale=self.env_config['mean_seconds_per_step']) if self.env_config['random_seconds_per_step'] else self.env_config['mean_seconds_per_step']
	
	def get_junction_roads(self, j, source_point, source_orientation):
		relative_road_pos_vector = shift_and_rotate_vector(
			[
				road.start.pos if j.pos!=road.start.pos else road.end.pos
				for road in j.roads_connected
			], 
			source_point, 
			source_orientation
		) #/ self.max_relative_coordinate
		# relative_road_pos_vector = relative_road_pos_vector.reshape((-1, 4))
		if self.culture:
			road_feature_vector = np.array(
				[
					road.binary_features(as_tuple=True) # in [0,1]
					for road in j.roads_connected
				], 
				dtype=np.float32
			)
			junction_road_list = np.concatenate(
				[
					relative_road_pos_vector,
					road_feature_vector
				], 
				axis=-1
			).tolist()
		else:
			junction_road_list = relative_road_pos_vector.tolist()
		junction_road_list.sort(key=lambda x: x[:2])
		
		missing_roads = [self._empty_road]*(self.env_config['max_roads_per_junction']-len(j.roads_connected))
		return junction_road_list + missing_roads

	def get_roads_view_list(self, sorted_junctions, source_point, source_orientation, n_junctions):
		return [
			self.get_junction_roads(sorted_junctions[i][1], source_point, source_orientation) 
			if i < len(sorted_junctions) else 
			self._empty_junction_roads
			for i in range(n_junctions)
		]

	def get_junction_view_list(self, sorted_junctions, source_point, source_orientation, n_junctions):
		return [
			np.array(
				(
					*sorted_junctions[i][0], 
					sorted_junctions[i][1].is_source, 
					normalize_delivery_count(
						sorted_junctions[i][1].refills, 
						self.env_config['max_refills_per_source']
					) if sorted_junctions[i][1].is_source else -1,
					sorted_junctions[i][1].is_target, 
					normalize_delivery_count(
						sorted_junctions[i][1].deliveries, 
						self.env_config['max_deliveries_per_target']
					) if sorted_junctions[i][1].is_target else -1,
				), 
				dtype=np.float32
			)
			if i < len(sorted_junctions) else 
			self._empty_junction
			for i in range(n_junctions)
		]

	def get_visible_junctions(self, source_point, source_orientation):
		relative_jpos_vector = shift_and_rotate_vector(
			[j.pos for j in self.road_network.junctions], 
			source_point, 
			source_orientation
		) #/ self.max_relative_coordinate
		return sorted(zip(relative_jpos_vector.tolist(),self.road_network.junctions), key=lambda x: x[0])

	def get_neighbourhood_view(self, source_point, source_orientation):
		if self.other_agent_list:
			alive_agent = [x for x in self.other_agent_list if not x.is_dead]
			sorted_alive_agents = sorted(
				(
					(
						(
							shift_and_rotate_vector(agent.car_point, source_point, source_orientation) #/ self.max_relative_coordinate
						).tolist(),
						agent.car_orientation/two_pi,
						agent.get_agent_feature_list(), 
					)
					for agent in alive_agent
				), 
				key=lambda x: x[0]
			)
			sorted_alive_agents = [
				(*agent_point, agent_orientation, *agent_state)
				for agent_point, agent_orientation, agent_state in sorted_alive_agents
			]
			agents_view_list = [
				np.array(sorted_alive_agents[i], dtype=np.float32) 
				if i < len(sorted_alive_agents) else 
				self._empty_agent
				for i in range(len(self.other_agent_list))
			]
		else:
			agents_view_list = None
		return agents_view_list

	def is_in_junction(self, point, radius=None):
		if radius is None:
			radius = self.env_config['junction_radius']
		return euclidean_distance(self.closest_junction.pos, point) <= radius

	def is_on_road(self, point, max_distance=None):
		if max_distance is None:
			max_distance = self.env_config['max_distance_to_path']
		return point_to_line_dist(point, self.closest_road.edge) <= max_distance

	def move_car(self, max_space):
		x,y = self.car_point
		dx,dy = get_heading_vector(
			angle=self.car_orientation, 
			space=min(self.car_speed*self.step_seconds, max_space)
		)
		return (x+dx, y+dy)

	@property
	def neighbouring_junctions_iter(self):
		j_pos_set_iter = unique_everseen((
			j_pos
			for road in self.closest_junction.roads_connected 
			for j_pos in road.edge
		))
		return (
			self.road_network.junction_dict[j_pos]
			for j_pos in j_pos_set_iter
		)

	def start_step(self, action_vector):
		was_in_junction = self.is_in_junction(self.car_point) # This is correct because during reset cars are always spawn in a junction
		##################################
		## Get actions
		##################################
		if self.discrete_action_space:
			action_vector = (
				self.allowed_orientations[action_vector//len(self.allowed_speeds)],
				self.allowed_speeds[(action_vector%len(self.allowed_speeds))]
			)
		else:
			a_low = self.action_space.low[0]
			a_high = self.action_space.high[0]
			action_vector = (action_vector-a_low)%(a_high-a_low+1) + a_low
			# action_vector = np.clip(action_vector, self.action_space.low[0], self.action_space.high[0])
		if self.build_action_list:
			self.action_list.append(action_vector)
		##################################
		## Compute new orientation
		##################################
		orientation_action = (action_vector[0]+1)*pi
		# Optimal orientation on road
		if self.goal_junction: # is on road
			road_edge = self.closest_road.edge if self.closest_road.edge[-1] == self.goal_junction.pos else self.closest_road.edge[::-1]
			self.car_orientation = get_slope_radians(*road_edge)%two_pi # in [0, 2*pi)
			if orientation_action > pi/2 and orientation_action < 3*pi/2:
				self.car_orientation += pi
				self.car_orientation %= two_pi # in [0, 2*pi)
				tmp = self.goal_junction
				self.goal_junction = self.source_junction
				self.source_junction = tmp
		else: # is in junction
			self.car_orientation = (self.car_orientation+orientation_action)%two_pi
		##################################
		## Compute new speed
		##################################
		if self.decides_speed:
			speed_action = action_vector[1]
			self.car_speed = np.clip((speed_action+1)/2, self.env_config['min_speed'], self.env_config['max_speed'])
		##################################
		## Move car
		##################################
		distance_to_goal = euclidean_distance(self.car_point, self.goal_junction.pos) if self.goal_junction else float('inf')
		self.car_point = self.move_car(max_space=distance_to_goal)
		##################################
		## Get closest junction and road
		##################################
		old_previous_closest_junction = self.previous_closest_junction
		old_closest_junction = self.closest_junction
		is_in_junction = self.is_in_junction(self.car_point) # This is correct because during reset cars are always spawn in a junction
		if not is_in_junction:
			if not self.closest_road:
				road_set = self.closest_junction.roads_connected if not self.env_config['random_seconds_per_step'] else unique_everseen((r for j in self.neighbouring_junctions_iter for r in j.roads_connected), key=lambda x:x.edge) # self.closest_junction.roads_connected is correct because we are asserting that self.env_config['max_speed']*self.env_config['mean_seconds_per_step'] < self.env_config['min_junction_distance']
				_,self.closest_road = self.road_network.get_closest_road_by_point(self.car_point, road_set)
		else:
			self.closest_road = None
		if self.closest_road:
			junction_set = (self.road_network.junction_dict[self.closest_road.edge[0]],self.road_network.junction_dict[self.closest_road.edge[1]])
			_,self.closest_junction = self.road_network.get_closest_junction_by_point(self.car_point, junction_set)
			if old_closest_junction != self.closest_junction:
				self.previous_closest_junction = old_closest_junction
		# else:
		# 	_,self.closest_junction = self.road_network.get_closest_junction_by_point(self.car_point, self.neighbouring_junctions_iter)
		##################################
		## Adjust Car Position: Correct Errors
		##################################
		if self.culture:
			self.following_regulation = True
		self.stuck_in_junction = False
		if not is_in_junction:
			adjust_car_position = False
			if not self.is_on_road(self.car_point): # Force car to stay on a road or a junction; go back
				adjust_car_position = True
			elif self.culture:
				self.following_regulation, self.explanation_list = self.road_network.run_dialogue(self.closest_road, self.agent_id, explanation_type="compact")
				if not self.following_regulation:
					adjust_car_position = True
			if adjust_car_position:
				self.stuck_in_junction = True
				self.previous_closest_junction = old_previous_closest_junction
				self.closest_junction = old_closest_junction
				self.closest_road = None
				self.car_point = self.closest_junction.pos
				is_in_junction = True
		##################################
		## Update the environment
		##################################
		self.visiting_new_road = False
		self.visiting_new_junction = False
		self.has_just_taken_food = False
		self.has_just_delivered_food = False
		if is_in_junction:
			# self.steps_in_junction += 1
			self.visiting_new_junction = self.closest_junction != self.last_closest_junction
			if self.visiting_new_junction: # visiting a new junction
				if self.last_closest_road is not None: # if closest_road is not the first visited road
					self.last_closest_road.is_visited_by(self.agent_id, True) # set the old road as visited
				self.closest_junction.is_visited_by(self.agent_id, True) # set the current junction as visited
				#########
				self.source_junction = None
				self.goal_junction = None
				self.last_closest_road = None
				self.last_closest_junction = self.closest_junction
				#########
				if self.has_food:
					if is_target_junction(self.closest_junction) and self.road_network.deliver_food(self.closest_junction):
						self.has_food = False
						self.has_just_delivered_food = True
				else:
					if is_source_junction(self.closest_junction) and self.road_network.acquire_food(self.closest_junction):
						self.has_food = True
						self.has_just_taken_food = True
		else:
			self.visiting_new_road = self.last_closest_road != self.closest_road		
			if self.visiting_new_road: # not in junction and visiting a new road
				self.last_closest_junction = None
				self.last_closest_road = self.closest_road # keep track of the current road
				self.goal_junction = self.road_network.junction_dict[self.closest_road.edge[0] if self.closest_road.edge[1] == self.closest_junction.pos else self.closest_road.edge[1]]
				self.source_junction = self.closest_junction

	def end_step(self):
		reward, dead, reward_type = self.reward_fn()
		how_fair = self.fairness_type_fn()
		reward += self.fairness_reward_fn(how_fair)

		state = self.get_state()
		if self.stuck_in_junction:
			self.steps_stuck_in_junction += 1
		info_dict = {
			'explanation':{
				'why': reward_type,
				'how_fair': how_fair,
			},
			# 'discard': self.idle and not reward,
		}
		if self.culture:
			info_dict['explanation']['who'] = (self.agent_id.binary_features(as_tuple=True),)
		if self.env_config.get('build_action_list', False):
			info_dict["action_list"] = self.action_list

		self.is_dead = dead
		self.step += 1
		self.last_reward = reward
		return [state, reward, dead, info_dict]
			
	def get_info(self):
		return f"speed={self.car_speed}, orientation={self.car_orientation}"

	def unitary_sparse_reward_default(self):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return unitary_reward(is_positive=True, is_terminal=False, label='has_just_delivered_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			return null_reward(is_terminal=False, label='has_just_taken_from_source')

		#######################################
		# "Follow regulation" rule. # Do this before checking if agent is_stuck_in_junction
		if self.culture:
			if not self.following_regulation:
				return null_reward(is_terminal=self.terminate_if_wrong_behaviour, label=explanation_list_with_label('not_following_regulation', self.explanation_list))

		#######################################
		# "Is stuck in junction" rule
		if self.stuck_in_junction:
			return null_reward(is_terminal=self.terminate_if_wrong_behaviour, label='is_stuck_in_junction')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')
		
		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def unitary_frequent_reward_default(self):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return unitary_reward(is_positive=True, is_terminal=False, label='has_just_delivered_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			return unitary_reward(is_positive=True, is_terminal=False, label='has_just_taken_from_source')

		#######################################
		# "Follow regulation" rule. # Do this before checking if agent is_stuck_in_junction
		if self.culture:
			if not self.following_regulation:
				return unitary_reward(is_positive=False, is_terminal=self.terminate_if_wrong_behaviour, label=explanation_list_with_label('not_following_regulation', self.explanation_list))

		#######################################
		# "Is stuck in junction" rule
		if self.stuck_in_junction:
			return unitary_reward(is_positive=False, is_terminal=self.terminate_if_wrong_behaviour, label='is_stuck_in_junction')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')

		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def unitary_more_sparse_reward_default(self):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Mission completed" rule
		if self.road_network.min_deliveries == self.env_config['max_deliveries_per_target']:
			return unitary_reward(is_positive=True, is_terminal=True, label='mission_completed')

		#######################################
		# "Has delivered food to target" rule
		if self.has_just_delivered_food:
			return null_reward(is_terminal=False, label='has_just_delivered_to_target')

		#######################################
		# "Has taken food from source" rule
		if self.has_just_taken_food:
			return null_reward(is_terminal=False, label='has_just_taken_from_source')

		#######################################
		# "Follow regulation" rule. # Do this before checking if agent is_stuck_in_junction
		if self.culture:
			if not self.following_regulation:
				return null_reward(is_terminal=self.terminate_if_wrong_behaviour, label=explanation_list_with_label('not_following_regulation', self.explanation_list))

		#######################################
		# "Is stuck in junction" rule
		if self.stuck_in_junction:
			return null_reward(is_terminal=self.terminate_if_wrong_behaviour, label='is_stuck_in_junction')

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(self.car_point):
			return null_reward(is_terminal=False, label='is_in_junction')

		#######################################
		# "Move forward" rule
		return null_reward(is_terminal=False, label='moving_forward')

	def simple_fairness_type(self):
		####### Facts
		if self.has_just_delivered_food: 
			just_delivered_to_worst_target = self.closest_junction.deliveries == self.road_network.min_deliveries or self.closest_junction.deliveries-1 == self.road_network.min_deliveries
			return 'has_fairly_pursued_a_poor_target' if just_delivered_to_worst_target else 'has_pursued_a_rich_target'
		return 'unknown'

	def simple_fairness_reward(self, how_fair):
		return 1 if 'has_fairly_pursued_a_poor_target' == how_fair else 0

	def engineered_fairness_type(self):
		how_fair = self.simple_fairness_type()
		if how_fair != 'unknown':
			return how_fair
		if self.visiting_new_junction and self.has_food:
			# is_visible_target = lambda x: x.is_available_target and (self.can_see(x.pos) or any((a.can_see(x.pos) for a in self.other_agent_list)))
			self.taget_distance_dict = self.road_network.get_closest_target_type(
				self.closest_junction, 
				max_depth=self.max_depth_searching_for_closest_target,
				# is_target_fn=is_visible_target,
			)
			if self.previous_closest_junction:
				self.old_taget_distance_dict = self.road_network.get_closest_target_type(
					self.previous_closest_junction, 
					max_depth=self.max_depth_searching_for_closest_target,
					# is_target_fn=is_visible_target,
				)
				if self.taget_distance_dict['poor_target_distance'] < self.old_taget_distance_dict['poor_target_distance']:
					return 'is_likely_to_fairly_pursue_a_poor_target'
				if self.taget_distance_dict['poor_target_distance'] > self.old_taget_distance_dict['poor_target_distance']:
					return 'is_unlikely_to_fairly_pursue_a_poor_target'
				if self.taget_distance_dict['rich_target_distance'] < self.old_taget_distance_dict['rich_target_distance']:
					return 'is_likely_to_pursue_a_rich_target'
		return 'unknown'

	def engineered_fairness_reward(self, how_fair):
		if 'has_fairly_pursued_a_poor_target' == how_fair:
			return 1
		if 'is_likely_to_fairly_pursue_a_poor_target' == how_fair:
			return 1/(self.taget_distance_dict['poor_target_distance']+1)
		if 'is_unlikely_to_fairly_pursue_a_poor_target' == how_fair:
			return -1/(self.old_taget_distance_dict['poor_target_distance']+1)
		return 0

	def unitary_engineered_fairness_reward(self, how_fair):
		if 'has_fairly_pursued_a_poor_target' == how_fair:
			return 1
		if 'is_likely_to_fairly_pursue_a_poor_target' == how_fair:
			return 1
		if 'is_unlikely_to_fairly_pursue_a_poor_target' == how_fair:
			return -1
		return 0

class FullWorldAllAgents_GraphDelivery(MultiAgentEnv):
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		for i,a in enumerate(self.agent_list):
			a.seed(seed+i)
		self._seed = seed-1
		self.np_random = seeding.np_random(self._seed)[0]
		# if self.culture:
		# 	self.culture.np_random = self.np_random
		return [self._seed]

	def __init__(self, config=None):
		self.env_config = config
		self.num_agents = config.get('num_agents',1)
		self.viewer = None

		self.env_config['min_junction_distance'] = 2.5*self.env_config['junction_radius']

		assert self.env_config['min_junction_distance'] > 2*self.env_config['junction_radius'], f"min_junction_distance has to be greater than {2*self.env_config['junction_radius']} but it is {self.env_config['min_junction_distance']}"
		assert self.env_config['max_speed']*self.env_config['mean_seconds_per_step'] < self.env_config['min_junction_distance'], f"max_speed*mean_seconds_per_step has to be lower than {self.env_config['min_junction_distance']} but it is {self.env_config['max_speed']*self.env_config['mean_seconds_per_step']}"

		logger.warning(f'Setting environment with reward_fn <{self.env_config["reward_fn"]}>, culture <{self.env_config["culture"]}>, fairness_type_fn <{self.env_config["fairness_type_fn"]}> and fairness_reward_fn <{self.env_config["fairness_reward_fn"]}>')
		self.culture = eval(f'{self.env_config["culture"]}Culture')(
			# road_options={
			# 	'require_priority': 1/8,
			# 	'accident': 1/8,
			# 	'require_fee': 1/8,
			# }, agent_options={
			# 	'emergency_vehicle': 1/5,
			# 	'has_priority': 1/2,
			# 	'can_pay_fee': 1/2,
			# }
		) if self.env_config["culture"] else None

		self.agent_list = [
			FullWorldAllAgents_Agent(self.num_agents-1, self.culture, self.env_config)
			for _ in range(self.num_agents)
		]
		self.action_space = self.agent_list[0].action_space
		self.observation_space = self.agent_list[0].observation_space
		self._agent_ids = set(range(self.num_agents))
		if self.env_config.get('build_joint_action_list', False):
			self._empty_action_vector = np.zeros((self.action_space.shape[0],), dtype=np.float32) if not self.env_config.get('n_discrete_actions',None) else np.full((1,), -1, dtype=np.float32)
		self.seed(config.get('seed',21))

	def reset(self):
		###########################
		self.road_network = MultiAgentRoadNetwork(
			self.culture, 
			self.np_random,
			map_size=(self.env_config['max_dimension'], self.env_config['max_dimension']), 
			min_junction_distance=self.env_config['min_junction_distance'],
			max_roads_per_junction=self.env_config['max_roads_per_junction'],
			number_of_agents=self.num_agents,
			junctions_number=self.env_config['junctions_number'],
			target_junctions_number=self.env_config['target_junctions_number'],
			source_junctions_number=self.env_config['source_junctions_number'],
			max_refills_per_source=self.env_config['max_refills_per_source'], 
			max_deliveries_per_target=self.env_config['max_deliveries_per_target'],
		)
		starting_point_list = self.road_network.get_random_starting_point_list(n=self.num_agents, source_only=self.env_config['spawn_on_sources_only'])
		for uid,agent in enumerate(self.agent_list):
			agent.reset(
				starting_point_list[uid], 
				self.road_network.agent_list[uid], 
				self.road_network, 
				self.agent_list[:uid]+self.agent_list[uid+1:]
			)
		# get_state is gonna use information about all agents, so initialize them first
		initial_state_dict = {
			uid: agent.get_state()
			for uid,agent in enumerate(self.agent_list)
		}
		self._step_count = 1
		return initial_state_dict

	def step(self, action_dict):
		###################################
		## Shuffle actions order
		action_dict_items = list(action_dict.items())
		self.np_random.shuffle(action_dict_items)
		###################################
		## Change environment
		# end_step uses information about all agents, this requires all agents to act first and compute rewards and states after everybody acted
		state_dict, reward_dict, terminal_dict, info_dict = {}, {}, {}, {}
		for uid,action in action_dict_items:
			self.agent_list[uid].start_step(action)
		for uid in action_dict.keys():
			state_dict[uid], reward_dict[uid], terminal_dict[uid], info_dict[uid] = self.agent_list[uid].end_step()
		terminal_dict['__all__'] = is_terminal = all(terminal_dict.values()) or self.road_network.min_deliveries == self.env_config['max_deliveries_per_target'] or self._step_count == self.env_config.get('horizon',float('inf'))
		###################################
		## Build stats dict
		if is_terminal:
			avg_steps_stuck_in_junction = sum((a.steps_stuck_in_junction for a in self.agent_list))/len(self.agent_list)
			dead_bots = sum((1 if a.is_dead else 0 for a in self.agent_list))
			for uid in action_dict.keys():
				info_dict[uid]["stats_dict"] = {
					"deliveries": self.road_network.deliveries,
					"fair_deliveries": self.road_network.fair_deliveries,
					"refills": self.road_network.refills,
					"avg_steps_stuck_in_junction": avg_steps_stuck_in_junction,
					"dead_bots": dead_bots,
				}
		###################################
		## Build the joint actions list
		if self.env_config.get('build_joint_action_list', False) and action_dict:
			action_list_size = len(self.agent_list[uid].action_list)
			joint_action_list = list(map(sorted, zip(*[
				a.action_list+[self._empty_action_vector]*(action_list_size-len(a.action_list)) 
				for a in self.agent_list
			])))
			for uid in action_dict.keys():
				info_dict[uid]['joint_action_list'] = joint_action_list
		self._step_count += 1
		return state_dict, reward_dict, terminal_dict, info_dict
			
	def get_info(self):
		return json.dumps({
			uid: agent.get_info()
			for uid,agent in enumerate(self.agent_list)
		}, indent=4)
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5), tight_layout=True)
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		handles = []

		def get_car_color(a):
			if a.is_dead:
				return 'grey'
			if a.has_food:
				return 'green'
			return 'red'
		handles += [
			ax.scatter(*self.road_network.source_junctions[0].pos, marker='s', facecolor='green', edgecolor='black', lw=1, alpha=1, label='Loaded Bot'),
			ax.scatter(*self.road_network.target_junctions[0].pos, marker='s', facecolor='red', edgecolor='black', lw=1, alpha=1, label='Unloaded Bot')
		]
		if self.culture:
			handles.append(ax.scatter(*self.road_network.source_junctions[0].pos, marker='s', color='grey', alpha=1, label='Dead Bot'))

		goal_junction_set = set((agent.goal_junction.pos for agent in self.agent_list if agent.goal_junction))
		def get_junction_color(j):
			if is_target_junction(j):
				return 'red'
			if is_source_junction(j):
				return 'green'
			# if j.pos in goal_junction_set:
			# 	return 'blue'
			# if j.pos in closest_junction_set:
			# 	return 'blue'
			for agent in self.agent_list:
				if not agent.is_dead and agent.can_see(j.pos):
					return 'orange'
			return 'grey'
		handles += [
			ax.scatter(*self.road_network.target_junctions[0].pos, marker='o', color='red', alpha=1, label='Target Node'),
			ax.scatter(*self.road_network.source_junctions[0].pos, marker='o', color='green', alpha=1, label='Source Node'),
			ax.scatter(*self.road_network.target_junctions[0].pos, marker='o', color='orange', alpha=1, label='Visible Node'),
		]
		
		# [Car]
		#######################
		car_len = self.env_config['max_dimension']/16
		car_view = [ # [Vehicle]
			Rectangle(
				xy=(agent.car_point[0]-car_len/2,agent.car_point[1]-car_len/2), 
				width=car_len,
				height=car_len, 
				facecolor=get_car_color(agent), 
				edgecolor='black',
				lw=1,
				alpha=1 if not agent.is_dead else 0.25,
			)
			for uid,agent in enumerate(self.agent_list)
		]
		ax.add_collection(PatchCollection(car_view, match_original=True))
		#######################
		for uid,agent in enumerate(self.agent_list): # [Heading Vector]
			car_x, car_y = agent.car_point
			dir_x, dir_y = get_heading_vector(angle=agent.car_orientation, space=1.5*car_len)
			heading_vector_handle = ax.plot(
				[car_x, car_x+dir_x],[car_y, car_y+dir_y], 
				color=get_car_color(agent), 
				alpha=1, 
				# label='Heading Vector'
			)
		#######################
		visibility_radius = self.env_config.get('visibility_radius',None)
		if visibility_radius: # [Visibility]
			visibility_view = [
				Circle(
					agent.car_point, 
					visibility_radius, 
					color='blue', 
					alpha=0.05,
				)
				for uid,agent in enumerate(self.agent_list)
				if not agent.is_dead
			]
			ax.add_collection(PatchCollection(visibility_view, match_original=True))
		#######################
		for uid,agent in enumerate(self.agent_list): # [Rewards]
			if not agent.last_reward:
				continue
			ax.text(
				x=agent.car_point[0]+2,
				y=agent.car_point[1]+2,
				s=f"{agent.last_reward:.2f}",
			)
			if agent.is_dead:
				agent.last_reward = None
		#######################
		if self.env_config.get('print_debug_info',True):
			for uid,agent in enumerate(self.agent_list): # [Debug info]
				if agent.is_dead:
					continue
				if agent.visiting_new_road:
					ax.text(
						x=agent.car_point[0],
						y=agent.car_point[1],
						s='R', 
					)
				if agent.visiting_new_junction:
					ax.text(
						x=agent.car_point[0],
						y=agent.car_point[1],
						s='J', 
					)
		#######################
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [
				Circle(
					junction.pos, 
					self.env_config['junction_radius'], 
					color=get_junction_color(junction), 
					alpha=.5,
					label='Target Node' if is_target_junction(junction) else ('Source Node' if is_source_junction(junction) else 'Normal Node')
				)
				for junction in self.road_network.junctions
			]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)
			for junction in self.road_network.target_junctions:
				ax.annotate(
					junction.deliveries, 
					(junction.pos[0],junction.pos[1]+0.5), 
					color='black', 
					# weight='bold', 
					fontsize=12, 
					ha='center', 
					va='center'
				)
			closest_junction_set = unique_everseen((agent.closest_junction for agent in self.agent_list if not agent.is_dead), key=lambda x:x.pos)
			#######################
			if self.env_config.get('print_debug_info',True):
				for junction in filter(lambda x: not x.is_target, closest_junction_set): # [Debug info]
					ax.annotate(
						'#', 
						junction.pos, 
						color='black', 
						# weight='bold', 
						fontsize=12, 
						ha='center', 
						va='center'
					)

		# [Roads]
		closest_road_set = set((agent.closest_road.edge for agent in self.agent_list if agent.closest_road and not agent.is_dead))
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			line_style = '--' if road.edge in closest_road_set else '-'
			path_handle = ax.plot(
				road_pos[0], road_pos[1], 
				color='black', 
				ls=line_style, 
				lw=2, 
				alpha=.5, 
				label="Road"
			)

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		ax.legend(handles=handles)
		# figure.tight_layout()
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		figure.clear()
		return data # RGB array

	def render(self, mode='human'):
		img = self.get_screen()
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			cv2.imshow('GraphDelivery rendering',img)
			cv2.waitKey(100)
			return True
