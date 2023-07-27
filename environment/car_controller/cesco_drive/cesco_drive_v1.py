# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
from environment.car_controller.cesco_drive.cesco_drive_v0 import CescoDriveV0
from environment.car_controller.utils import *

class CescoDriveV1(CescoDriveV0):

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
		reward, finished, reward_type = self.get_reward(car_speed=self.speed, car_point=self.car_point, old_car_point=old_car_point, car_progress=self.car_progress, car_position=car_position, obstacles=self.obstacles)
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
		terminal = finished or completed_track or out_of_time
		if terminal: # populate statistics
			self.is_over = True
			stats = {
				"avg_speed": self.avg_speed_per_steps/self._step,
				"completed_track": 1 if completed_track else 0,
				"out_of_time": 1 if out_of_time else 0,
			}
			for rule in ['move_forward','avoid_collision','follow_lane','respect_speed_limit','go_fast']:
				stats[rule] = 1 if reward_type!=rule else 0
			self.episode_statistics = stats
		return [state, reward, terminal, {'explanation':reward_type}]
	
	def get_reward(self, car_speed, car_point, old_car_point, car_progress, car_position, obstacles):
		# "Move forward" rule
		if car_position <= car_progress: # car is not moving toward next position
			return (-1., True, 'move_forward') # terminate episode
		# "Avoid collisions" rule
		max_distance_to_path = self.max_distance_to_path
		car_projection_point = self.get_point_from_position(car_position)
		closest_obstacle = self.get_closest_obstacle(point=car_projection_point, obstacles=obstacles)
		if closest_obstacle is not None:
			obstacle_point, obstacle_radius = closest_obstacle
			if self.has_collided_obstacle(obstacle=closest_obstacle, old_car_point=old_car_point, car_point=car_point): # collision
				return (-1., True, 'avoid_collision') # terminate episode
			if euclidean_distance(obstacle_point, car_projection_point) <= obstacle_radius: # could collide obstacle
				max_distance_to_path += obstacle_radius
		space_gained = car_speed*self.seconds_per_step
		# "Follow the lane" rule
		distance = euclidean_distance(car_point, car_projection_point)
		if distance > max_distance_to_path:
			return (-space_gained, False, 'follow_lane') # terminate episode
		# "Respect the speed limit" rule
		if car_speed > self.speed_upper_limit:
			return (-space_gained, False, 'respect_speed_limit') # terminate episode
		# "Default" rule
		# if self.is_terminal_position(car_position): # avoid this reward if goal is not visible
		# 	return (1., True, 'reach_goal') # terminate episode
		return (space_gained, False, 'none') # do NOT terminate episode
