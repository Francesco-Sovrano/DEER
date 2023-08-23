# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import numpy as np

import copy
from matplotlib import use as matplotlib_use, patches
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from environment.car_controller.grid_drive.lib.road_grid import RoadGrid
from environment.car_controller.grid_drive.lib.road_cultures import *

import logging
logger = logging.getLogger(__name__)

class GridDrive(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array']}
	GRID_DIMENSION				= 15
	MAX_SPEED 					= 120
	SPEED_GAP					= 10
	MAX_GAPPED_SPEED			= MAX_SPEED//SPEED_GAP
	MAX_STEP					= 2**5
	DIRECTIONS					= 4 # N,S,W,E
	VISITED_CELL_GRID_IDX		= -2
	AGENT_CELL_GRID_IDX			= -1

	def get_state(self):
		state_dict = {
			"cnn_grid-128": self.grid_view,
			"fc_neighbours-64": self.grid.neighbour_features(),
		}
		if self.obs_car_features > 0:
			state_dict["fc_agent_extra_properties-64"] = self.grid.agent.binary_features()
		return state_dict

	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def __init__(self, config=None):
		logger.warning(f'Setting environment with reward_fn <{config["reward_fn"]}> and culture_level <{config["culture_level"]}>')
		self.reward_fn = eval(f'self.{config["reward_fn"]}')
		self.culture = eval(f'{config["culture_level"]}RoadCulture')(road_options={
			'motorway': 1/2,
			'stop_sign': 1/2,
			'school': 1/2,
			'single_lane': 1/2,
			'town_road': 1/2,
			'roadworks': 1/8,
			'accident': 1/8,
			'heavy_rain': 1/2,
			'congestion_charge': 1/8,
		}, agent_options={
			'emergency_vehicle': 1/5,
			'heavy_vehicle': 1/4,
			'worker_vehicle': 1/3,
			'tasked': 1/2,
			'paid_charge': 1/2,
			'speed': self.MAX_SPEED,
		})
		self.obs_road_features = len(self.culture.properties)  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(self.culture.agent_properties)-1  # Number of binary CAR features in Hard Culture (excluded speed)

		# Direction (N, S, W, E) + Speed [0-MAX_SPEED]
		# self.action_space	   = gym.spaces.MultiDiscrete([self.DIRECTIONS, self.MAX_GAPPED_SPEED])
		self.action_space	   = gym.spaces.Discrete(self.DIRECTIONS*self.MAX_GAPPED_SPEED)

		state_dict = {
			"cnn_grid-128": gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION, self.obs_road_features+2]), # Features representing the grid + visited cells + current position
			"fc_neighbours-64": gym.spaces.MultiBinary(self.obs_road_features * self.DIRECTIONS), # Neighbourhood view
		}
		if self.obs_car_features > 0:
			state_dict["fc_agent_extra_properties-64"] = gym.spaces.MultiBinary(self.obs_car_features) # Car features

		self.observation_space = gym.spaces.Dict(state_dict)
		self.step_counter = 0

	def reset(self):
		self.is_over = False
		self.culture.np_random = self.np_random
		self.viewer = None
		self.step_counter = 0
		self.cumulated_return = 0
		self.sum_speed = 0

		self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION, self.culture)
		self.grid_features = np.array(self.grid.get_features(), ndmin=3, dtype=np.int8)
		self.grid_view = np.concatenate([
			self.grid_features,
			np.zeros((self.GRID_DIMENSION, self.GRID_DIMENSION, 2), dtype=np.int8), # current position + visited cells
		], -1)

		x,y = self.grid.agent_position
		self.grid_view[x][y][self.AGENT_CELL_GRID_IDX] = 1 # set new position
		self.grid_view[x][y][self.VISITED_CELL_GRID_IDX] = 1 # set current cell as visited
		self.visited_cells = 1
		self.speed = self.grid.agent["Speed"]
		return self.get_state()

	def step(self, action_vector):
		self.step_counter += 1
		self.direction = action_vector//self.MAX_GAPPED_SPEED
		self.speed = (action_vector%self.MAX_GAPPED_SPEED)*self.SPEED_GAP
		self.sum_speed += self.speed
		# direction, gapped_speed = action_vector
		old_x, old_y = self.grid.agent_position # get this before moving the agent
		reward, dead, explanatory_labels = self.reward_fn(*self.grid.move_agent(self.direction, self.speed))
		self.cumulated_return += reward
		if not self.visiting_old_cell:
			self.visited_cells += 1 # increase it before setting the current position as visited, otherwise visiting_old_cell will always be true
		new_x, new_y = self.grid.agent_position # get this after moving the agent
		# do the following aftwer moving the agent and checking positions with get_reward
		self.grid_view[old_x][old_y][self.AGENT_CELL_GRID_IDX] = 0 # remove old position
		self.grid_view[new_x][new_y][self.AGENT_CELL_GRID_IDX] = 1 # set new position
		self.grid_view[new_x][new_y][self.VISITED_CELL_GRID_IDX] = 1 # set current cell as visited
		info_dict = {'explanation': explanatory_labels}
		out_of_time = self.step_counter >= self.MAX_STEP
		terminated_episode = dead or out_of_time
		if terminated_episode: # populate statistics
			self.is_over = True
			info_dict["stats_dict"] = {
				"avg_speed": self.sum_speed/self.step_counter,
				"out_of_time": 1 if out_of_time else 0,
				"visited_cells": self.visited_cells,
			}
		return [
			self.get_state(), # observation
			reward, 
			terminated_episode,
			info_dict,
		]

	def get_screen(self):  # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(10, 10), tight_layout=True)
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111)  # nrows=1, ncols=1, index=1

		cell_side = 20

		# Compute speed limits for all cells.
		road_limits = {}
		columns = self.grid.width
		rows = self.grid.height
		temp_agent = copy.deepcopy(self.grid.agent)
		for x in range(columns):
			for y in range(rows):
				road = self.grid.cells[x][y]
				road_limits[road] = self.grid.road_culture.get_speed_limits(road, self.grid.agent) # (None,None) if road is unfeasible
		# Draw cells
		shapes = []
		for x in range(columns):
			for y in range(rows):
				road = self.grid.cells[x][y]
				# Draw rectangle
				left = x * cell_side
				right = left + cell_side
				bottom = y * cell_side
				top = bottom + cell_side
				if self.grid_view[x][y][self.VISITED_CELL_GRID_IDX] > 0:  # Already visited cell
					cell_handle = Rectangle((left, bottom), cell_side, cell_side, color='gray', alpha=0.25)
				elif road_limits[road] == (None, None):  # Unfeasible road
					cell_handle = Rectangle((left, bottom), cell_side, cell_side, color='red', alpha=0.25)
				else: # new road with possible speed limits
					cell_handle = Rectangle((left, bottom), cell_side, cell_side, fill=False)
				shapes.append(cell_handle)

				# Do not add label if agent is on top of cell.
				if (x, y) == self.grid.agent_position:
					continue
				# Add speed limit label
				min_speed, max_speed = road_limits[road]
				label = f'{min_speed}-{max_speed}' if min_speed is not None else 'N/A'
				ax.text(0.5*(left + right), 0.5*(bottom + top), label,
							horizontalalignment='center', verticalalignment='center', size=18)



		# Draw agent and agent label
		agent_x, agent_y = self.grid.agent_position
		left = agent_x * cell_side
		right = left + cell_side
		bottom = agent_y * cell_side
		top = agent_y * cell_side
		agent_circle = Circle((left + (cell_side/2), bottom + (cell_side/2)), cell_side/2, color='b', alpha=0.5)
		shapes.append(agent_circle)

		patch_collection = PatchCollection(shapes, match_original=True)
		ax.add_collection(patch_collection)

		# Adjust view around agent
		zoom_factor = 3
		left_view = agent_x - zoom_factor
		right_view = agent_x + zoom_factor
		bottom_view = agent_y - zoom_factor
		top_view = agent_y + zoom_factor
		if agent_x > (columns - zoom_factor):  # Too far right
			left_view -= (agent_x + zoom_factor) - columns
		elif agent_x < zoom_factor:  		   # Too far left
			right_view += (zoom_factor - agent_x)
		if agent_y > (rows - zoom_factor):     # Too far up
			bottom_view -= (agent_y + zoom_factor) - rows
		elif agent_y < zoom_factor: 		   # Too far down
			top_view += (zoom_factor - agent_y)
		ax.set_xlim([max(0, left_view * cell_side),
					 min(columns * cell_side, right_view * cell_side)])
		ax.set_ylim([max(0, bottom_view * cell_side),
					 min(rows * cell_side, top_view * cell_side)])
		# ax.set_ylim([0, rows * cell_side])

		# Draw agent commanded speed on top of circle
		label = str(self.grid.agent["Speed"])
		ax.text(left + (cell_side/2), bottom + (cell_side/2), label,
					horizontalalignment='center', verticalalignment='center', size=18)

		# # Adjust ax limits in order to get the same scale factor on both x and y
		# a, b = ax.get_xlim()
		# c, d = ax.get_ylim()
		# max_length = max(d - c, b - a)
		# ax.set_xlim([a, a + max_length])
		# ax.set_ylim([c, c + max_length])

		# figure.tight_layout()
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		figure.clear()
		return data  # RGB array

	def render(self, mode='human'):
		img = self.get_screen()
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

	@property
	def visiting_old_cell(self):
		x, y = self.grid.agent_position
		return self.grid_view[x][y][self.VISITED_CELL_GRID_IDX] > 0
	
	def frequent_reward_default(self, following_regulation, explanation_list):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = (self.speed+1)/self.MAX_SPEED # in (0,1]
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.visiting_old_cell: # already visited cell
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')

	def frequent_reward_explanation_engineering_v1(self, following_regulation, explanation_list):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = (self.speed+1)/self.MAX_SPEED # in (0,1]
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.visiting_old_cell: # already visited cell
			return null_reward(is_terminal=False, label=explanation_list_with_label('not_visiting_new_roads',explanation_list))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('moving_forward',explanation_list))

	def frequent_reward_explanation_engineering_v2(self, following_regulation, explanation_list):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = (self.speed+1)/self.MAX_SPEED # in (0,1]
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.visiting_old_cell: # already visited cell
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('moving_forward',explanation_list))

	def frequent_reward_step_multiplied_by_junctions(self, following_regulation, explanation_list):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = (self.speed+1)/self.MAX_SPEED # in (0,1]
			reward *= self.visited_cells
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.visiting_old_cell: # already visited cell
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')

	def frequent_reward_full_step(self, following_regulation, explanation_list):
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			reward = (self.speed+1)/self.MAX_SPEED # in (0,1]
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		if not following_regulation:
			return step_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.visiting_old_cell: # already visited cell
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')
