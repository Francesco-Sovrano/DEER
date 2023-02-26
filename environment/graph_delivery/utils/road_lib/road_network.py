import numpy as np
import sys
from .road_cultures import *
from .road_cell import RoadCell
from .road_agent import RoadAgent
from ..geometry import *
from ..random_planar_graph.GenerateGraph import get_random_planar_graph

class Junction:
	def __init__(self, pos):
		self.pos = pos
		self.roads_connected = []
		self.visiting_dict = {}

	def __eq__(self, other):
		if not isinstance(other,Junction):
			return False
		return self.pos == other.pos

	def __len__(self):
		return len(self.roads_connected)

	def is_visited_by(self, agent, set_to=None):
		if set_to is not None:
			self.visiting_dict[agent] = set_to
		else:
			return self.visiting_dict.get(agent,False)

	@property
	def is_visited(self):
		return len(self.visiting_dict) > 0 and any(self.visiting_dict.values())

	def connect(self, road):
		if road not in self.roads_connected:
			self.roads_connected.append(road)
		return True

class Road(RoadCell):
	def __init__(self, start: Junction, end: Junction, connect=False):
		# arbitrary sorting by x-coordinate to avoid mirrored duplicates
		super().__init__()
		self.start = start
		self.end = end
		self.edge = (start.pos, end.pos)
		self.normalised_slope = (get_slope_radians(*self.edge)%two_pi)/two_pi  # in [0,1)
		self.is_connected = False
		self.visiting_dict = {}
		self.colour = None
		if connect:
			self.connect_to_junctions()
			self.is_connected = True

	def is_visited_by(self, agent, set_to=None):
		if set_to is not None:
			self.visiting_dict[agent] = set_to
		return self.visiting_dict.get(agent,False)

	@property
	def is_visited(self):
		return len(self.visiting_dict) > 0 and any(self.visiting_dict.values())
	
	def __eq__(self, other):
		if not isinstance(other,Road):
			return False
		return self.start == other.start and self.end == other.end

	@property
	def id(self):
		return (self.start.pos, self.end.pos)

	def connect_to_junctions(self):
		self.start.connect(self)
		self.end.connect(self)

class RoadNetwork:

	def __init__(self, culture, np_random, map_size=(50, 50), min_junction_distance=None, max_roads_per_junction=8):
		self.np_random = np_random
		self.junctions = []
		self.roads = []
		self.map_size = map_size
		self.max_roads_per_junction = max_roads_per_junction
		self.agent = RoadAgent()
		if min_junction_distance is None:
			self.min_junction_distance = map_size[0]/8
		else:
			self.min_junction_distance = min_junction_distance
		self.road_culture = culture
		if self.road_culture:
			self.agent.set_culture(culture)
			self.road_culture.initialise_random_agent(self.agent, self.np_random)

	@staticmethod
	def get_closest_junction(junction_list, point):
		return min(junction_list, key=lambda x: euclidean_distance(x.pos,point))

	@staticmethod
	def get_furthermost_junction(junction_list, point):
		return max(junction_list, key=lambda x: euclidean_distance(x.pos,point))

	def run_dialogue(self, road, agent, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			road: RoadCell corresponding to destination cell.
			agent: RoadAgent corresponding to agent.
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# print("@@@@@@@@@@@@@ NEW DIALOGUE @@@@@@@@@@@@@")
		# Game starts with proponent using argument 0 ("I will not get a ticket").
		return self.road_culture.run_default_dialogue(road, agent, explanation_type=explanation_type)

	def normalise_speed(self, min_, max_, current):
		"""
		Normalises speed from Euclidean m/s to nominal speeds used in the culture rules (0-100)
		Args:
			min: min speed in m/s
			max: max speed in m/s
			current: current speed in m/s
		Returns: speed normalised to range (0-120)
		"""
		return self.road_culture.agent_options.get('speed',120) * ((current - min_) / (max_ - min_))

	def add_junction(self, junction):
		if junction not in self.junctions:
			self.junctions.append(junction)

	def add_road(self, road):
		if road not in self.roads:
			self.roads.append(road)

	def get_visible_junctions_by_point(self, source_point, horizon_distance):
		return [
			junction
			for junction in self.junctions
			if euclidean_distance(source_point, junction.pos) <= horizon_distance
		]

	def get_closest_road_and_junctions(self, point, closest_junctions=None):
		# the following lines of code are correct because the graph is planar
		if not closest_junctions:
			distance_to_closest_road, closest_road = self.get_closest_road_by_point(point)
		else:
			distance_to_closest_road, closest_road = min(
				(
					(
						point_to_line_dist(point, r.edge),
						r
					)
					for j in closest_junctions
					for r in j.roads_connected
				), key=lambda x:x[0]
			)
		road_start, road_end = closest_road.edge
		closest_junctions = [self.junction_dict[road_start],self.junction_dict[road_end]]
		return distance_to_closest_road, closest_road, closest_junctions

	def get_closest_junction_by_point(self, source_point, junction_set=None):
		if junction_set is None:
			junction_set = self.junctions
		return min(
			(
				(
					euclidean_distance(junction.pos,source_point),
					junction
				)
				for junction in junction_set
			), key=lambda x:x[0]
		)
	
	def get_closest_road_by_point(self, source_point, road_set=None):
		if road_set is None:
			road_set = self.roads
		return min(
			(
				(
					point_to_line_dist(source_point,road.edge),
					road
				)
				for road in road_set
			), key=lambda x:x[0]
		)

	def set(self, nodes_amount):
		self.junctions = []
		self.roads = []
		random_planar_graph = get_random_planar_graph({
			"width": self.map_size[0], # "Width of the field on which to place points.  neato might choose a different width for the output image."
			"height": self.map_size[1], # "Height of the field on which to place points.  As above, neato might choose a different size."
			"nodes": nodes_amount, # "Number of nodes to place."
			"edges": 2*nodes_amount, # "Number of edges to use for connections.  Double edges aren't counted."
			"radius": self.min_junction_distance, # "Nodes will not be placed within this distance of each other."
			"double": 0, # "Probability of an edge being doubled."
			"hair": 0, # "Adjustment factor to favour dead-end nodes.  Ranges from 0.00 (least hairy) to 1.00 (most hairy).  Some dead-ends may exist even with a low hair factor."
			"seed": self.np_random.randint(0,sys.maxsize), # "Seed for the random number generator."
			"debug_trimode": 'conform', # ['pyhull', 'triangle', 'conform'], "Triangulation mode to generate the initial triangular graph.  Default is conform.")
			"debug_tris": None, # "If a filename is specified here, the initial triangular graph will be saved as a graph for inspection."
			"debug_span": None, # "If a filename is specified here, the spanning tree will be saved as a graph for inspection."
		})
		self.junction_dict = dict(zip(random_planar_graph['nodes'], map(Junction, random_planar_graph['nodes'])))
		self.junctions = tuple(self.junction_dict.values())
		spanning_tree_set = set(random_planar_graph['spanning_tree'])
		# print('edges', random_planar_graph['edges'])
		for edge in random_planar_graph['edges']:
			p1,p2 = edge
			j1 = self.junction_dict[p1]
			j2 = self.junction_dict[p2]
			if len(j1) < self.max_roads_per_junction and len(j2) < self.max_roads_per_junction:
				road = Road(j1, j2, connect=True)
				self.roads.append(road)
				if self.road_culture:
					road.set_culture(self.road_culture)
					self.road_culture.initialise_random_road(road, self.np_random)
		self.junctions = list(filter(lambda x: x.roads_connected, self.junctions))
		starting_index = self.np_random.choice(len(random_planar_graph['spanning_tree']), 1)[0]
		starting_point = random_planar_graph['spanning_tree'][starting_index][0]
		return starting_point
