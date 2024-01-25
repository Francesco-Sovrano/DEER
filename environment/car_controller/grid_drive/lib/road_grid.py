# import numpy as np
from environment.car_controller.grid_drive.lib.road_cell import RoadCell
from environment.car_controller.grid_drive.lib.road_agent import RoadAgent
import numpy as np

NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3

class RoadGrid:
	def __init__(self, x_dim, y_dim, culture):
		self.agent = RoadAgent()
		self.agent_position = (0, 0)
		self.width = x_dim
		self.height = y_dim

		self.road_culture = culture
		self.agent.set_culture(self.road_culture)
		self.road_culture.initialise_random_agent(self.agent)
		self.cells = self.initialise_random_grid()
		self.set_random_position()

	def set_random_position(self):
		x, y = self.road_culture.np_random.integers(0,self.width), self.road_culture.np_random.integers(0,self.height)
		self.agent_position = (x,y)
		self.road_culture.initialise_feasible_road(self.cells[x][y])

	def within_bounds(self, coord):
		"""
		Checks if a given coordinate exists in the 2D grid.
		:param coord: Position to check.
		:returns True if within bounds. False otherwise.
		"""
		return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

	def neighbours_of(self, coord, neighbourhood_type='von_neumann'):
		"""
		Returns immediate neighbourhood of given coordinate.
		:param coord: Position to return neighbourhood.
		:param neighbourhood_type: Currently supports 'von_neumann' or 'moore' neighbourhoods.
		:return: List of neighbours. False if arguments are invalid..
		"""
		if not self.within_bounds(coord):
			print("RoadGrid::neighbours_of: Position out of bounds.")
			return None
		x, y = coord
		neighbours = []
		if neighbourhood_type == 'von_neumann':
			# if self.within_bounds((x - 1, y)): neighbours.append((x - 1, y))  # Left
			# if self.within_bounds((x + 1, y)): neighbours.append((x + 1, y))  # Right
			# if self.within_bounds((x, y - 1)): neighbours.append((x, y - 1))  # Up
			# if self.within_bounds((x, y + 1)): neighbours.append((x, y + 1))  # Down
			neighbours += [
				((x - 1)%self.height, y), # Left
				((x + 1)%self.height, y), # Right
				(x, (y - 1)%self.width), # Up
				(x, (y + 1)%self.width), # Down
			]
		elif neighbourhood_type == 'moore':
			for i in range(-1, 2):
				for j in range(-1, 2):
					if i == j == 0: continue
					# if self.within_bounds((x + i, y + j)): neighbours.append((x + i, y + j))
					neighbours.append(((x + i)%self.height, (y + j)%self.width))
		else:
			print("RoadGrid::neighbours_of: This neighbourhood type is not supported.")
			return None
		return neighbours

	def neighbour_features(self):
		# Start with order NORTH, SOUTH, EAST, WEST.
		x, y = self.agent_position
		north_features = self.cells[x][(y + 1)%self.width].binary_features() #if self.within_bounds((x, y + 1)) else self.inaccessible
		south_features = self.cells[x][(y - 1)%self.width].binary_features() #if self.within_bounds((x, y - 1)) else self.inaccessible
		east_features  = self.cells[(x + 1)%self.height][y].binary_features() #if self.within_bounds((x + 1, y)) else self.inaccessible
		west_features  = self.cells[(x - 1)%self.height][y].binary_features() #if self.within_bounds((x - 1, y)) else self.inaccessible

		return np.concatenate([north_features, south_features, east_features, west_features], -1)

	def get_features(self):
		return tuple(
			tuple(
				e.binary_features()
				for e in row
			)
			for row in self.cells
		)

	def initialise_random_grid(self):
		"""
		Fills a grid with random RoadCells, each initialised by the current culture.
		:return:
		"""
		def build_road(i,j):
			road = RoadCell(i, j)
			road.set_culture(self.road_culture)
			self.road_culture.initialise_random_road(road)
			return road
		return tuple(
			tuple(
				build_road(i,j)
				for j in range(self.height)	
			)
			for i in range(self.width)	
		)

	def run_dialogue(self, road, agent, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			road: RoadCell corresponding to destination cell.
			agent: RoadAgent corresponding to agent.
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# Game starts with proponent using argument 0 ("I will not get a ticket").
		return self.road_culture.run_default_dialogue(road, agent, explanation_type=explanation_type)

	def move_agent(self, direction, speed):
		"""
		Attempts to move an agent to a neighbouring cell.
		:param speed: commanded speed to traverse next cell
		:param direction: 0 == NORTH, 1 == SOUTH, 2 == EAST, 3 == WEST
		:return: False if move is illegal. Integer-valued reward if move is valid.
		"""
		# if self.agent_position is False:
		# 	print("RoadGrid::move_agent: Agent not found!")
		# 	return False

		dest_x, dest_y = self.agent_position
		if   direction == NORTH:
			dest_y += 1
		elif direction == SOUTH:
			dest_y -= 1
		elif direction == EAST:
			dest_x += 1
		elif direction == WEST:
			dest_x -= 1
		dest_x %= self.width # infinite grid
		dest_y %= self.height # infinite grid

		# if not self.within_bounds((dest_x, dest_y)):
		# 	return "FAIL: Out of bounds!"

		self.agent_position = (dest_x, dest_y)
		self.agent.assign_property_value("Speed", speed)

		can_move, explanation_list = self.run_dialogue(self.cells[dest_x][dest_y], self.agent, explanation_type="compact")
		return can_move, explanation_list

