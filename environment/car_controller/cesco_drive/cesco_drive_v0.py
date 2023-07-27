# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import numpy as np

from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from scipy import optimize
from environment.car_controller.utils import *

import logging
logger = logging.getLogger(__name__)

class CescoDriveV0(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array']}
	mean_seconds_per_step = 0.1 # in average, a step every n seconds
	horizon_distance = 3 # meters
	track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	wheelbase = 0.9 # meters # https://en.wikipedia.org/wiki/Wheelbase
	# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
	min_speed = 0.4 # m/s
	max_speed = 1.6 # m/s
	# the fastest car has max_acceleration 9.25 m/s^2 (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
	# the slowest car has max_acceleration 0.7 m/s^2 (http://automdb.com/max_acceleration)
	max_acceleration = 0.7 # m/s^2
	# the best car has max_deceleration 29.43 m/s^2 (https://www.quora.com/What-can-be-the-maximum-deceleration-during-braking-a-car?share=1)
	# a normal car has max_deceleration 7.1 m/s^2 (http://www.batesville.k12.in.us/Physics/PhyNet/Mechanics/Kinematics/BrakingDistData.html)
	max_deceleration = 7.1 # m/s^2
	max_steering_degree = 30
	max_step_per_spline = 75
	control_points_per_step = 10
	max_distance_to_path = 0.3 # meters
	# obstacles related stuff
	max_obstacle_count = 6
	min_obstacle_radius = 0.1 # meters
	max_obstacle_radius = 0.3 # meters
	min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	max_speed_noise = 0.25 # m/s
	max_steering_noise_degree = 2
	# splines related stuff
	spline_number = 2
	control_points_per_spline = 50

	def get_concatenation_size(self):
		return 3
		
	def get_concatenation(self):
		return np.array([self.steering_angle/self.max_steering_angle, self.speed/self.max_speed, self.speed/self.speed_upper_limit], dtype=np.float32)

	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def __init__(self, config=None):
		self.viewer = None
		self.max_step = self.max_step_per_spline*self.spline_number
		self.speed_lower_limit = max(self.min_speed_lower_limit,self.min_speed)
		self.meters_per_step = 2*self.max_speed*self.mean_seconds_per_step
		self.max_steering_angle = np.deg2rad(self.max_steering_degree)
		self.max_steering_noise_angle = np.deg2rad(self.max_steering_noise_degree)
		# Spaces
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # steering angle, continuous control without softmax
		# There are 2 types of objects (obstacles and lines), each object has 3 numbers (x, y and size)
		# if no obstacles are considered, then there is no need for representing the line size because it is always set to 0
		cnn_dict = {
			"points": gym.spaces.Box(
				low=-25,
				high=25,
				shape=(1,self.control_points_per_step,2)
			),
		}
		if self.max_obstacle_count > 0:
			cnn_dict["obstacles"] = gym.spaces.Box(
				low=-25,
				high=25,
				shape=(1,self.max_obstacle_count,3)
			)
		self.observation_space = gym.spaces.Dict({
			"cnn": gym.spaces.Dict(cnn_dict),
			"fc": gym.spaces.Dict({
				"extra": gym.spaces.Box(
					low=-1,
					high=self.max_speed/self.speed_lower_limit,
					shape=(self.get_concatenation_size(),)
				),
			}),
		})
	
	def reset(self):
		self.is_over = False
		self.episode_statistics = {}
		self._step = 0
		###########################
		self.seconds_per_step = self.get_step_seconds()
		self.path = self.build_random_path()
		# car position
		self.car_point = (0,0) # car point and orientation are always expressed with respect to the initial point and orientation of the road fragment
		self.car_progress = 0 # self.find_closest_position(point=self.car_point)
		self.car_orientation = self.get_angle_from_position(self.car_progress)
		# speed limit
		self.speed_upper_limit = self.speed_lower_limit + (self.max_speed-self.speed_lower_limit)*self.np_random.random() # in [speed_lower_limit,max_speed]
		# steering angle & speed
		self.speed = self.min_speed + (self.max_speed-self.min_speed)*self.np_random.random() # in [min_speed,max_speed]
		self.steering_angle = 0
		# get obstacles
		self.obstacles = self.get_new_obstacles()
		# init concat variables
		self.last_reward = 0
		self.last_reward_type = 'move_forward'
		self.last_action_mask = None
		self.last_state = self.get_state(car_point=self.car_point, car_orientation=self.car_orientation, car_progress=self.car_progress, obstacles=self.obstacles)
		# init log variables
		self.cumulative_reward = 0
		self.avg_speed_per_steps = 0
		return self.last_state
			
	def get_new_obstacles(self):
		if self.max_obstacle_count <= 0:
			return []
		def build_random_obstacle():
			radius = self.min_obstacle_radius + (self.max_obstacle_radius-self.min_obstacle_radius)*self.np_random.random() # in [min_obstacle_radius,max_obstacle_radius]
			# add track/2 to obstacle_radius because car_point is the centre of the front of the car
			radius += self.track/2
			# get a point on the road
			x,y = self.get_point_from_position(self.spline_number*self.np_random.random())
			# the obstacle must be on the road but it does not necessarily to have its centre on the road
			x += radius*(2*self.np_random.random()-1)
			y += radius*(2*self.np_random.random()-1)
			centre = (x,y)
			return (centre,radius)
		return [
			build_random_obstacle()
			for _ in range(self.np_random.randint(self.max_obstacle_count))
		]
		
	def get_closest_obstacle(self, point, obstacles):
		if len(obstacles) == 0:
			return None
		obstacle_distances_from_point = map(lambda obstacle: (obstacle, euclidean_distance(obstacle[0], point)-obstacle[1]), obstacles)
		return min(obstacle_distances_from_point, key=lambda tup: tup[1])[0]
		
	def get_spline_by_position(self, position):
		return np.clip(int(np.ceil(position)-1),0,self.spline_number-1)
		
	def get_horizon_around(self, position): # get horizon around
		spline = self.get_spline_by_position(position)
		length = self.lengths[spline]
		return self.horizon_distance/length
		
	def get_position_around(self, position): # this function measures the max possible position distance, using: road_length, max_car_speed, seconds_per_step
		spline = self.get_spline_by_position(position)
		length = self.lengths[spline]
		return self.meters_per_step/length
		
	def get_point_from_position(self, position):
		spline = self.get_spline_by_position(position)
		relative_position = position-spline
		origin = self.origins[spline]
		orientation = self.orientations[spline]
		U = self.Us[spline]
		V = self.Vs[spline]
		return rotate_and_shift(poly(relative_position,U), poly(relative_position,V), origin[0], origin[1], orientation)
		
	def get_angle_from_position(self, position):
		spline = self.get_spline_by_position(position)
		return angle(position-spline, self.Us[spline], self.Vs[spline])+self.orientations[spline]
		
	def build_random_path(self):
		# setup environment
		self.Us = [] # Us are x-generator splines
		self.Vs = [] # Vs are y-generator splines
		self.lengths = [] # length of each spline
		self.origins = [(0,0)] # origin of each spline
		self.orientations = [0] # orientation angle of each spline
		xy = []
		for i in range(self.spline_number):
			U, V = generate_random_polynomial(self.np_random)
			self.Us.append(U)
			self.Vs.append(V)
			self.orientations.append(angle(1, U, V))
			self.origins.append(self.get_point_from_position(i+1))
			# get spline length
			self.lengths.append(get_poly_length(spline=(U,V), integration_range=(0,1)))
			# get control points
			control_points = np.linspace(start=i, stop=i+1, num=self.control_points_per_spline)
			xy += map(self.get_point_from_position, control_points)
		return list(zip(*xy))

	def is_terminal_position(self, position):
		return self.get_goal(position) >= self.spline_number*0.9
		
	def find_closest_spline_position(self, point, start_position=0, end_position=1):
		return optimize.minimize_scalar(lambda pos: euclidean_distance(point, self.get_point_from_position(pos)), method='bounded', bounds=(start_position,end_position)).x
			
	def find_closest_position(self, point, previous_position=None): # Find the closest spline
		if previous_position is not None:
			position_around = max(self.get_position_around(previous_position),1) # usually 1 is big enough, but sometimes the road length may be too small
			# N.B. properly tuning position_around may significantly speed up the search for the closest_spline_position
			return self.find_closest_spline_position(point=point, start_position=previous_position, end_position=previous_position+position_around)
		elif self.spline_number == 1:
			return self.find_closest_spline_position(point=point)
		else:
			# road is non-convex function, thus we have to find separately closest position in all splines
			closest_position_generator = (self.find_closest_spline_position(point=point, start_position=spline, end_position=spline+1) for spline in range(self.spline_number))
			position_distance_generator = [(pos, euclidean_distance(point,self.get_point_from_position(pos))) for pos in closest_position_generator]
			return min(position_distance_generator, key=lambda t: t[1])[0]

	def find_closest_point(self, point, previous_position=None):
		closest_position = self.find_closest_position(point, previous_position)
		return self.get_point_from_position(closest_position)

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
		return self.np_random.exponential(scale=self.mean_seconds_per_step)

	def step(self, action_vector):
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		# compute new acceleration
		self.acceleration = self.get_acceleration_from_action(action=action_vector[1])
		# compute new speed
		self.speed = self.accelerate(speed=self.speed, acceleration=self.acceleration)
		# move car
		old_car_point = self.car_point
		self.car_point, self.car_orientation = self.move(point=self.car_point, orientation=self.car_orientation, steering_angle=self.steering_angle, speed=self.speed, add_noise=True)
		# update position and direction
		car_position = self.find_closest_position(point=self.car_point, previous_position=self.car_progress)
		# compute perceived reward
		reward, dead, reward_type = self.get_reward(car_speed=self.speed, car_point=self.car_point, old_car_point=old_car_point, car_progress=self.car_progress, car_position=car_position, obstacles=self.obstacles)
		if car_position > self.car_progress: # is moving toward next position
			self.car_progress = car_position # progress update
		# compute new state (after updating progress)
		state = self.get_state(car_point=self.car_point, car_orientation=self.car_orientation, car_progress=self.car_progress, obstacles=self.obstacles)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = reward
		self.last_reward_type = reward_type
		# update cumulative reward
		self.cumulative_reward += reward
		self.avg_speed_per_steps += self.speed
		# update step
		self._step += 1
		completed_track = self.is_terminal_position(car_position)
		out_of_time = self._step >= self.max_step
		terminal = dead or completed_track or out_of_time
		info_dict = {'explanation':reward_type}
		if terminal: # populate statistics
			self.is_over = True
			stats = {
				"avg_speed": self.avg_speed_per_steps/self._step,
				"completed_track": 1 if completed_track else 0,
				"out_of_time": 1 if out_of_time else 0,
			}
			if self.max_obstacle_count > 0:
				stats["avoid_collision"] = 0 if dead else 1
			info_dict["stats_dict"] = self.episode_statistics = stats
		return [state, reward, terminal, info_dict]
		
	def has_collided_obstacle(self, old_car_point, car_point, obstacle):
		return segment_collide_circle(circle=obstacle, segment=(old_car_point, car_point))
		
	def get_reward(self, car_speed, car_point, old_car_point, car_progress, car_position, obstacles):
		max_distance_to_path = self.max_distance_to_path
		car_projection_point = self.get_point_from_position(car_position)
		closest_obstacle = self.get_closest_obstacle(point=car_projection_point, obstacles=obstacles)
		if closest_obstacle is not None:
			obstacle_point, obstacle_radius = closest_obstacle
			if self.has_collided_obstacle(obstacle=closest_obstacle, old_car_point=old_car_point, car_point=car_point): # collision
				return (-1, True, 'collision') # terminate episode
			if euclidean_distance(obstacle_point, car_projection_point) <= obstacle_radius: # could collide obstacle
				max_distance_to_path += obstacle_radius
		if car_position > car_progress: # is moving toward next position
			distance = euclidean_distance(car_point, car_projection_point)
			distance_ratio = np.clip(distance/max_distance_to_path, 0,1) # always in [0,1]
			inverse_distance_ratio = 1 - distance_ratio
			# the more car_speed > self.speed_upper_limit, the bigger the malus
			malus = self.speed_upper_limit*max(0,car_speed/self.speed_upper_limit-1)*self.seconds_per_step
			# smaller distances to path give higher rewards
			bonus = min(car_speed,self.speed_upper_limit)*self.seconds_per_step*inverse_distance_ratio
			return (bonus-malus, False, 'move_forward') # do not terminate episode
		#else is NOT moving toward next position
		return (-0.1, False, 'move_backward') # do not terminate episode
		
	def get_control_points(self, source_point, source_orientation, source_position): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_goal = self.get_goal(source_position)
		# print(source_position, source_goal)
		control_points = np.zeros((1,self.control_points_per_step,2), dtype=np.float32)
		source_x, source_y = source_point
		control_distance = (source_goal-source_position)/self.control_points_per_step
		# add control points
		for i in range(self.control_points_per_step):
			cp_x, cp_y = self.get_point_from_position(source_position + (i+1)*control_distance)
			cp_x, cp_y = shift_and_rotate(cp_x, cp_y, -source_x, -source_y, -source_orientation) # get control point with coordinates relative to source point
			control_points[0][i][0] = cp_x
			control_points[0][i][1] = cp_y
		return control_points
		
	def get_control_obstacles(self, car_point, car_orientation, obstacles):
		car_x, car_y = car_point
		control_obstacles = []
		for (j, obstacle) in enumerate(obstacles):
			obstacle_point, obstacle_radius = obstacle
			if euclidean_distance(obstacle_point,car_point) <= self.horizon_distance+obstacle_radius:
				ro_x, ro_y = shift_and_rotate(obstacle_point[0], obstacle_point[1], -car_x, -car_y, -car_orientation) # get control point with coordinates relative to car point
				control_obstacles.append((ro_x, ro_y, obstacle_radius))
		# sort obstacles by euclidean distance from closer to most distant
		if len(control_obstacles) > 0:
			# sort obstacles by euclidean distance from closer to most distant
			control_obstacles.sort(key=lambda t: np.absolute(euclidean_distance((t[0],t[1]),car_point)-t[2]))
		for _ in range(self.max_obstacle_count-len(control_obstacles)):
			control_obstacles.append((0.,0.,0.))
		return np.array([control_obstacles], dtype=np.float32)
		
	def get_state(self, car_point, car_orientation, car_progress, obstacles):
		state = {
			"cnn": {
				"points": self.get_control_points(car_point, car_orientation, car_progress),
			},
			"fc": {
				"extra": self.get_concatenation(), 
			},
		}
		if self.max_obstacle_count > 0:
			state["cnn"]["obstacles"] = self.get_control_obstacles(car_point, car_orientation, obstacles)
		return state
		
	def get_goal(self, position):
		return position + self.get_horizon_around(position)
	
	def get_info(self):
		return "speed={}, steering_angle={}, orientation={}\n".format(self.speed, self.steering_angle, self.car_orientation)
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5))
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		# [Obstacles]
		if len(self.obstacles) > 0:
			circles = [Circle(point,radius,color='b') for (point,radius) in self.obstacles]
			patch_collection = PatchCollection(circles, match_original=True)
			ax.add_collection(patch_collection)
		# [Car]
		car_x, car_y = self.car_point
		car_handle = ax.scatter(car_x, car_y, marker='o', color='g', label='Car')
		# [Heading Vector]
		dir_x, dir_y = get_heading_vector(angle=self.car_orientation)
		heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')
		# [Goal]
		waypoint_x, waypoint_y = self.get_point_from_position(self.get_goal(self.car_progress))
		goal_handle = ax.scatter(waypoint_x, waypoint_y, marker='o', color='r', label='Horizon')
		# [Path]
		path_handle, = ax.plot(self.path[0], self.path[1], lw=2, alpha=0.5, label='Path')
		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car_handle,heading_vector_handle,goal_handle,path_handle]
		if len(self.obstacles) > 0:
			# https://stackoverflow.com/questions/11423369/matplotlib-legend-circle-markers
			handles.append(Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue", label='Obstacle'))
		ax.legend(handles=handles)
		# Draw plot
		figure.suptitle('[Angle]{1:.2f}° [Orientation]{4:.2f}° \n [Speed]{0:.2f} m/s [Limit]{3:.2f} m/s [Step]{2}'.format(self.speed, np.rad2deg(self.steering_angle), self._step, self.speed_upper_limit, np.rad2deg(self.car_orientation)))
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
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen
				
	def get_statistics(self):
		return self.episode_statistics
