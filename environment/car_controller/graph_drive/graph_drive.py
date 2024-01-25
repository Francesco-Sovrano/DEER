# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from PIL import Image

from environment.car_controller.utils import *
from environment.car_controller.graph_drive.lib.roads import *
from environment.car_controller.grid_drive.lib.road_cultures import *

import logging
logger = logging.getLogger(__name__)

class GraphDrive(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array']}
	random_seconds_per_step = False # whether to sample seconds_per_step from an exponential distribution
	mean_seconds_per_step = 0.25 # in average, a step every n seconds
	# track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	wheelbase = 0.35 # meters # https://en.wikipedia.org/wiki/Wheelbase
	# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
	min_speed = 0.2 # m/s
	max_speed = 1.2 # m/s
	# the fastest car has max_acceleration 9.25 m/s^2 (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
	# the slowest car has max_acceleration 0.7 m/s^2 (http://automdb.com/max_acceleration)
	max_acceleration = 1 # m/s^2
	# the best car has max_deceleration 29.43 m/s^2 (https://www.quora.com/What-can-be-the-maximum-deceleration-during-braking-a-car?share=1)
	# a normal car has max_deceleration 7.1 m/s^2 (http://www.batesville.k12.in.us/Physics/PhyNet/Mechanics/Kinematics/BrakingDistData.html)
	max_deceleration = 7 # m/s^2
	max_steering_degree = 45
	max_step = 2**9
	max_distance_to_path = 0.5 # meters
	# min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	# max_speed_noise = 0.25 # m/s
	# max_steering_noise_degree = 2
	max_speed_noise = 0 # m/s
	max_steering_noise_degree = 0
	# multi-road related stuff
	max_dimension = 25
	map_size = (max_dimension, max_dimension)
	junction_number = 16
	max_roads_per_junction = 4
	junction_radius = 1
	min_junction_distance = 2.5*junction_radius
	MAX_NORMALISED_SPEED = 120

	assert min_junction_distance > 2*junction_radius, f"min_junction_distance has to be greater than {2*junction_radius} but it is {min_junction_distance}"
	assert max_speed*mean_seconds_per_step < min_junction_distance, f"max_speed*mean_seconds_per_step has to be lower than {min_junction_distance} but it is {max_speed*mean_seconds_per_step}"

	@property
	def normalised_speed(self):
		# return (self.speed-self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
		return self.speed/self.max_speed # in (0,1]

	def get_state(self, car_point, car_orientation):
		road_view, junction_view = self.get_view(car_point, car_orientation)
		return {
			"fc": {
				"road_view": road_view,
				"junction_view": junction_view,
				"agent_features": np.concatenate([
					self.get_agent_state(),
					self.road_network.agent.binary_features(), 
				], axis=-1),
			}
		}

	def get_agent_state_size(self):
		return 5 # normalised steering angle + normalised speed
		
	def get_agent_state(self):
		return np.array((
			self.steering_angle/self.max_steering_angle, # normalised steering angle
			self.speed/self.max_speed, # normalised speed
			min(1, self.distance_to_closest_road/self.max_distance_to_path),
			self.is_in_junction(self.car_point),
			len(self.visited_junctions)/self.junction_number,
		), dtype=np.float32)

	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		self.np_random, _ = seeding.np_random(seed)
		return [seed]

	def __init__(self, config=None):
		self.config = config
		self.viewer = None
		self.max_steering_angle = np.deg2rad(self.max_steering_degree)
		self.max_steering_noise_angle = np.deg2rad(self.max_steering_noise_degree)

		logger.warning(f'Setting environment with reward_fn <{self.config["reward_fn"]}> and culture_level <{self.config["culture_level"]}>')
		self.reward_fn = eval(f'self.{self.config["reward_fn"]}')
		self.culture = eval(f'{self.config["culture_level"]}RoadCulture')(road_options={
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
			'speed': self.MAX_NORMALISED_SPEED,
		})
		self.obs_road_features = len(self.culture.properties)  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(self.culture.agent_properties) - 1  # Number of binary CAR features in Hard Culture (excluded speed)
		# Spaces
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # steering angle and speed
		self.observation_space = gym.spaces.Dict({
			"fc": gym.spaces.Dict({
				"road_view": gym.spaces.Box( # Closest road to the agent (the one it's driving on), sorted by relative position
					low= -1,
					high= 1,
					shape= (
						2 + 2 + self.obs_road_features + 1, # road properties: road.start.pos + road.end.pos + road.af_features + road.is_new_road
					),
					dtype=np.float32
				),
				"junction_view": gym.spaces.Box( # Roads directly connected to the closest road to the agent (the one it's driving on), sorted by relative position
					low= -1,
					high= 1,
					shape= ( # closest junctions view
						2, # junctions attached to the current road
						self.max_roads_per_junction, # maximum number of roads per junction
						2 + 2 + self.obs_road_features + 1,  # road properties: road.start.pos + road.end.pos + road.af_features + road.is_new_road
					),
					dtype=np.float32
				),
				"agent_features": gym.spaces.Box( # Agent features
					low= -1,
					high= 1,
					shape= (self.get_agent_state_size() + self.obs_car_features,),
					dtype=np.float32
				),
			}),
		})

	@staticmethod
	def normalize_point(p):
		return (np.clip(p[0]/GraphDrive.map_size[0],-1,1), np.clip(p[1]/GraphDrive.map_size[1],-1,1))

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_x, source_y = source_point
		shift_rotate_normalise_point = lambda x: self.normalize_point(shift_and_rotate(*x, -source_x, -source_y, -source_orientation))
		j1, j2 = self.closest_junction_list
		# Get road view
		road_points = ( # 2x2
			j1.pos,
			j2.pos,
		)
		relative_road_points = tuple(map(shift_rotate_normalise_point, road_points))
		road_view = sum(sorted(relative_road_points),()) + self.closest_road.binary_features(as_tuple=True) + (1 if self.closest_road.is_visited else 0,)
		road_view = np.array(road_view, dtype=np.float32)
		# Get junction view
		junction_view = np.array([ # 2 x self.max_roads_per_junction x (1+1)
			sorted([
				(
					*shift_rotate_normalise_point(road.start.pos),
					*shift_rotate_normalise_point(road.end.pos),
					*road.binary_features(as_tuple=True), # in [0,1]
					1 if road.is_visited else 0, # whether road has been previously visited
				) if euclidean_distance(road.start.pos,j.pos) < euclidean_distance(road.end.pos,j.pos) else (
					*shift_rotate_normalise_point(road.end.pos),
					*shift_rotate_normalise_point(road.start.pos),
					*road.binary_features(as_tuple=True), # in [0,1]
					1 if road.is_visited else 0, # whether road has been previously visited
				)
				for road in j.roads_connected
			], key=lambda x:(x[0:4])) + [ # placeholders for unavailable roads
				(
					-1,-1,-1,-1,
					*[-1]*self.obs_road_features,
					-1,
				)
			]*(self.max_roads_per_junction-len(j.roads_connected))
			for j,_ in sorted(zip((j1,j2),relative_road_points),key=lambda x:x[1])
		], dtype=np.float32)
		# print(junction_view.shape)
		return road_view, junction_view
	
	def reset(self):
		self.culture.np_random = self.np_random
		# print(0, self.np_random.random())
		self.is_over = False
		self._step = 0
		self.seconds_per_step = self.get_step_seconds()
		###########################
		self.road_network = RoadNetwork(
			self.culture, 
			map_size=self.map_size, 
			min_junction_distance=self.min_junction_distance,
			max_roads_per_junction=self.max_roads_per_junction,
		)
		# car position
		self.car_point = self.road_network.set(self.junction_number)
		self.car_orientation = (2*self.np_random.random()-1)*np.pi # in [-pi,pi]
		self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point)
		self.closest_junction = self.get_closest_junction(self.closest_junction_list, self.car_point)
		self.visited_junctions = [self.closest_junction]

		self.last_closest_road = None
		self.goal_junction = None
		self.current_road_speed_list = []
		# steering angle & speed
		self.steering_angle = 0
		self.speed = self.min_speed #+ (self.max_speed-self.min_speed)*self.np_random.random() # in [min_speed,max_speed]
		# self.speed = self.min_speed+(self.max_speed-self.min_speed)*(70/120) # for testing
		self.road_network.agent.assign_property_value("Speed", self.road_network.normalise_speed(self.min_speed, self.max_speed, self.speed))
		# init concat variables
		self.last_reward = 0
		self.last_reward_type = 'move_forward'
		self.last_action_mask = None
		self.last_state = self.get_state(car_point=self.car_point, car_orientation=self.car_orientation)
		# init log variables
		self.cumulative_reward = 0
		self.sum_speed = 0
		return self.last_state

	@staticmethod
	def get_closest_junction(junction_list, point):
		return min(junction_list, key=lambda x: euclidean_distance(x.pos,point))

	@staticmethod
	def get_furthest_junction(junction_list, point):
		return max(junction_list, key=lambda x: euclidean_distance(x.pos,point))

	def move(self, point, orientation, steering_angle, speed, add_noise=False):
		# https://towardsdatascience.com/how-self-driving-cars-steer-c8e4b5b55d7f?gi=90391432aad7
		# Add noise
		if add_noise:
			steering_angle += (2*self.np_random.random()-1)*self.max_steering_noise_angle
			steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle) # |steering_angle| <= max_steering_angle, ALWAYS
			speed += (2*self.np_random.random()-1)*self.max_speed_noise
		#### Ackerman Steering: Forward Kinematic for Car-Like vehicles #### https://www.xarg.org/book/kinematics/ackerman-steering/
		turning_radius = self.wheelbase/np.tan(steering_angle)
		# Max taylor approximation error of the tangent simplification is about 3° at 30° steering lock
		# turning_radius = self.wheelbase/steering_angle
		angular_velocity = speed/turning_radius
		# get normalized new orientation
		new_orientation = np.mod(orientation + angular_velocity*self.seconds_per_step, 2*np.pi) # in [0,2*pi)
		# Move point
		x, y = point
		dir_x, dir_y = get_heading_vector(angle=new_orientation, space=speed*self.seconds_per_step)
		return (x+dir_x, y+dir_y), new_orientation

	def get_steering_angle_from_action(self, action): # action is in [-1,1]
		return action*self.max_steering_angle # in [-max_steering_angle, max_steering_angle]
		
	def get_acceleration_from_action(self, action): # action is in [-1,1]
		return action*(self.max_acceleration if action >= 0 else self.max_deceleration) # in [-max_deceleration, max_acceleration]
		
	def accelerate(self, speed, acceleration):
		# use seconds_per_step instead of mean_seconds_per_step, because this way the algorithm is able to explore more states and train better
		# return np.clip(speed + acceleration*self.mean_seconds_per_step, self.min_speed, self.max_speed)
		return np.clip(speed + acceleration*self.seconds_per_step, self.min_speed, self.max_speed)
		
	def get_step_seconds(self):
		return self.np_random.exponential(scale=self.mean_seconds_per_step) if self.random_seconds_per_step is True else self.mean_seconds_per_step

	def is_in_junction(self, car_point, radius=None):
		if radius is None:
			radius = self.junction_radius
		return euclidean_distance(self.closest_junction.pos, car_point) <= radius

	def step(self, action_vector):
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		# compute new acceleration
		self.acceleration = self.get_acceleration_from_action(action=action_vector[1])
		# compute new speed
		self.speed = self.accelerate(speed=self.speed, acceleration=self.acceleration)
		self.road_network.agent.assign_property_value("Speed", self.road_network.normalise_speed(self.min_speed, self.max_speed, self.speed))
		# move car
		old_car_point = self.car_point
		old_goal_junction = self.goal_junction
		visiting_new_road = False
		self.car_point, self.car_orientation = self.move(
			point=self.car_point, 
			orientation=self.car_orientation, 
			steering_angle=self.steering_angle, 
			speed=self.speed, 
			add_noise=True
		)
		if self.goal_junction is None:
			self.distance_to_closest_road, self.closest_road, self.closest_junction_list = self.road_network.get_closest_road_and_junctions(self.car_point, self.closest_junction_list)
		else:
			self.distance_to_closest_road = point_to_line_dist(self.car_point, self.closest_road.edge)
		self.closest_junction = self.get_closest_junction(self.closest_junction_list, self.car_point)
		# if a new road is visited, add the old one to the set of visited ones
		self.acquired_junction = False
		if self.is_in_junction(self.car_point):
			if self.closest_junction not in self.visited_junctions:
				self.visited_junctions.append(self.closest_junction)
				self.acquired_junction = True
			self.goal_junction = None
			if self.last_closest_road is not None: # if closest_road is not the first visited road
				self.last_closest_road.is_visited = True # set the old road as visited
		elif self.last_closest_road != self.closest_road: # not in junction and visiting a new road
			visiting_new_road = True
			self.last_closest_road = self.closest_road # keep track of the current road
			self.goal_junction = self.get_furthest_junction(self.closest_junction_list, self.car_point)
			self.current_road_speed_list = []
		self.current_road_speed_list.append(self.speed)
		# compute perceived reward
		reward, dead, reward_type = self.reward_fn(visiting_new_road, old_goal_junction, old_car_point)
		# compute new state (after updating progress)
		state = self.get_state(
			car_point=self.car_point, 
			car_orientation=self.car_orientation,
		)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = reward
		self.last_reward_type = reward_type
		# update cumulative reward
		self.cumulative_reward += reward
		self.sum_speed += self.speed
		# update step
		self._step += 1
		out_of_time = self._step >= self.max_step
		terminal = dead or out_of_time
		info_dict = {'explanation':reward_type}
		if terminal: # populate statistics
			self.is_over = True
			info_dict["stats_dict"] = {
				"avg_speed": self.sum_speed/self._step,
				"out_of_time": 1 if out_of_time else 0,
				"visited_junctions": len(self.visited_junctions),
			}
		return [state, reward, terminal, info_dict]
			
	def get_info(self):
		return f"speed={self.speed}, steering_angle={self.steering_angle}, orientation={self.car_orientation}\n"
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5), tight_layout=True)
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [Circle(junction.pos, self.junction_radius, color='y', alpha=0.25) for junction in self.road_network.junctions]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)

		# [Car]
		car_x, car_y = self.car_point
		car_handle = ax.scatter(car_x, car_y, marker='o', color='g', label='Car')
		# [Heading Vector]
		dir_x, dir_y = get_heading_vector(angle=self.car_orientation, space=self.max_dimension/16)
		heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')

		# [Roads]
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			# print("Drawing road {} {}".format(road[0], road[1]))
			if road.colour is None:
				min_speed = self.road_network.road_culture.get_minimum_speed(road, self.road_network.agent) # None if road is unfeasible
				can_move = min_speed is not None
				road.colour = "Green" if can_move else "Red"
			road_colour = road.colour
			if road_colour == "Green":
				self.road_network.agent.assign_property_value("Speed", self.road_network.normalise_speed(self.min_speed, self.max_speed, self.speed))
				correct_properties, _ = self.road_network.run_dialogue(road, self.road_network.agent, explanation_type="compact")
				if not correct_properties:
					road_colour = "Gold"
			line_style = '-.' if road.is_visited else ('--' if road==self.closest_road else '-')
			path_handle, = ax.plot(road_pos[0], road_pos[1], color=colour_to_hex(road_colour), ls=line_style, lw=2, alpha=0.5, label="Road")
			# ax.fill_between(road_pos[0], np.array(road_pos[1])+self.max_distance_to_path, np.array(road_pos[1])-self.max_distance_to_path, alpha=0.1, color=colour_to_hex(road_colour))
			# ax.fill_between(road_pos[1], np.array(road_pos[0])+self.max_distance_to_path, np.array(road_pos[0])-self.max_distance_to_path, alpha=0.1, color=colour_to_hex(road_colour))

		path1_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Green"), lw=2, label="OK")
		path2_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Red"), lw=2, label="Unfeasible")
		path3_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Gold"), lw=2, label="Wrong Speed")
		path4_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Black"), ls='--', lw=2, label="Closest Road")
		path5_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Black"), ls='-.', lw=2, label="Old Road")
		# junction_handle = ax.scatter(0, 0, marker='o', color='y', label='Junction')

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car_handle, path1_handle, path2_handle, path3_handle, path4_handle, path5_handle]
		ax.legend(handles=handles)
		# Draw plot
		figure.suptitle(' '.join([
			f'[Angle]{np.rad2deg(self.steering_angle):.2f}°', 
			f'[Orient.]{np.rad2deg(self.car_orientation):.2f}°', 
			f'[Speed]{self.speed:.2f} m/s', 
			'\n',
			f'[Step]{self._step}', 
			f'[Old]{self.closest_road.is_visited}', 
			f'[Car]{self.road_network.agent.binary_features()}', 
			f'[Reward]{self.last_reward:.2f}',
		]))
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
		# elif mode == 'human':
		# 	from gym.envs.classic_control import rendering
		# 	if self.viewer is None:
		# 		self.viewer = rendering.SimpleImageViewer()
		# 	self.viewer.imshow(img)
		# 	return self.viewer.isopen
		elif mode == 'human':
			viewer = Image.fromarray(img)
			viewer.show()
			viewer.close()
			return True

	def frequent_reward_default(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			# reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')
				
	def frequent_reward_explanation_engineering_v1(self, visiting_new_road, old_goal_junction, old_car_point): # GOOD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			# reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('moving_forward', explanation_list))

	def frequent_reward_explanation_engineering_v2(self, visiting_new_road, old_goal_junction, old_car_point): # GOOD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			# reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label=explanation_list_with_label('not_visiting_new_roads', explanation_list))
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('moving_forward', explanation_list))

	def frequent_reward_explanation_engineering_v3(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			# reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return step_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')

	def frequent_reward_step_multiplied_by_junctions(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')

	def frequent_reward_full_step(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = np.mean(self.current_road_speed_list)
			# reward = self.speed
			reward = self.normalised_speed # in (0,1]
			# reward *= len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		#######################################
		# "Is in junction" rule
		if is_in_junction:
			return null_reward(is_terminal=False, label='is_in_junction')
		#######################################
		# "No U-Turning outside junction" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return step_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return step_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return step_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation', explanation_list))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(is_terminal=False, label='not_visiting_new_roads')
		# #######################################
		# # "Explore new roads" rule
		# if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
		# 	return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return step_reward(is_positive=True, is_terminal=False, label='moving_forward')

	def sparse_reward_default(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)-1
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label='is_in_new_junction')
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label='moving_forward')

	def sparse_reward_explanation_engineering_v1(self, visiting_new_road, old_goal_junction, old_car_point): # GOOD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)-1
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label='is_in_new_junction')
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label=explanation_list_with_label('moving_forward',explanation_list))

	def sparse_reward_explanation_engineering_v2(self, visiting_new_road, old_goal_junction, old_car_point): # GOOD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)-1
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction',self.last_explanation_list))
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label=explanation_list_with_label('moving_forward',explanation_list))

	def sparse_reward_explanation_engineering_v3(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)-1
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction',self.last_explanation_list))
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label='moving_forward')

	def sparse_reward_step_multiplied_by_junctions(self, visiting_new_road, old_goal_junction, old_car_point): # BAD
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)-1
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				return step_reward(is_positive=True, is_terminal=False, label='is_in_new_junction')
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label='moving_forward')
