from .road_lib.road_network import RoadNetwork
from .road_lib.road_agent import RoadAgent

class MultiAgentRoadNetwork(RoadNetwork):

	def __init__(self, culture, np_random, map_size=(50, 50), min_junction_distance=None, max_roads_per_junction=8, number_of_agents=5, junctions_number=10, target_junctions_number=5, source_junctions_number=5, max_refills_per_source=float('inf'), max_deliveries_per_target=10):
		assert junctions_number-target_junctions_number-source_junctions_number >= 0
		super().__init__(culture, np_random, map_size=map_size, min_junction_distance=min_junction_distance, max_roads_per_junction=max_roads_per_junction)
		### Agent
		del self.agent
		self.agent_list = [
			RoadAgent()
			for _ in range(number_of_agents)
		]
		if culture:
			for agent in self.agent_list:
				agent.set_culture(culture)
				culture.initialise_random_agent(agent, self.np_random)
		### Junction
		self.set(junctions_number)
		for j in self.junctions:
			j.is_source=j.is_available_source=False
			j.is_target=j.is_available_target=False
		self.target_junctions = []
		for j in self.np_random.choice(self.junctions, size=target_junctions_number, replace=False):
			j.is_target=j.is_available_target=True
			j.deliveries = 0
			self.target_junctions.append(j)
		non_target_junctions = [x for x in self.junctions if not x.is_target]
		self.source_junctions = []
		for j in self.np_random.choice(non_target_junctions, size=source_junctions_number, replace=False):
			j.is_source=j.is_available_source=True
			j.refills = 0
			self.source_junctions.append(j)
		### Constants
		self.max_refills_per_source = max_refills_per_source
		self.max_deliveries_per_target = max_deliveries_per_target
		### Deliveries
		self.min_deliveries = 0
		self.deliveries = 0
		self.fair_deliveries = 0
		self.refills = 0
		self.junction_dict = {
			x.pos: x
			for x in self.junctions
		}

	def acquire_food(self, j):
		if not j.is_available_source:
			return False
		j.refills += 1
		self.refills += 1
		if j.refills == self.max_refills_per_source:
			j.is_available_source = False
		return True

	def deliver_food(self, j):
		if not j.is_available_target:
			return False
		if self.min_deliveries == j.deliveries:
			self.fair_deliveries += 1
		j.deliveries += 1
		self.deliveries += 1
		self.min_deliveries = min(map(lambda x: x.deliveries, self.target_junctions))
		if j.deliveries == self.max_deliveries_per_target:
			j.is_available_target = False
		return True

	def get_target_type(self, j, is_target_fn):
		if not is_target_fn(j):
			return None
		if j.deliveries == self.min_deliveries:
			return 'poor'
		return 'rich'

	def get_closest_target_type(self, start_junction, max_depth=float('inf'), is_target_fn=None):
		if not is_target_fn:
			is_target_fn = lambda x: x.is_available_target
		# target_type = self.get_target_type(start_junction, is_target_fn)
		# if target_type:
		# 	return target_type, 0
		visited_junction_set = set()
		junction_list = [start_junction]
		depth = 1
		while junction_list and depth < max_depth:
			visited_junction_set.update(map(lambda x: x.pos, junction_list))
			other_junction_list = []
			for junction in junction_list:
				least_advantaged_target_list = []
				advantaged_target_list = []
				for road in junction.roads_connected:
					other_junction = self.junction_dict[road.end.pos if road.start.pos == junction.pos else road.start.pos]
					target_type = self.get_target_type(other_junction, is_target_fn)
					if target_type == 'poor':
						least_advantaged_target_list.append(other_junction)
					elif target_type == 'rich':
						advantaged_target_list.append(other_junction)
					elif other_junction.pos not in visited_junction_set:
						other_junction_list.append(other_junction)
				if least_advantaged_target_list and advantaged_target_list:
					return {
						'poor_target_distance': depth,
						'poor_target': least_advantaged_target_list[0],
						'rich_target_distance': depth,
						'rich_target': advantaged_target_list[0],
					}
				if least_advantaged_target_list:
					return {
						'poor_target_distance': depth,
						'poor_target': least_advantaged_target_list[0],
						'rich_target_distance': max_depth+1,
						'rich_target': None,
					}
				if advantaged_target_list:
					return {
						'poor_target_distance': max_depth+1,
						'poor_target': None,
						'rich_target_distance': depth,
						'rich_target': advantaged_target_list[0],
					}
			junction_list = other_junction_list
			depth += 1
		return {
			'poor_target_distance': max_depth+1,
			'poor_target': None,
			'rich_target_distance': max_depth+1,
			'rich_target': None,
		}

	def get_random_starting_point_list(self, n=1, source_only=True):
		return [
			j.pos
			for j in self.np_random.choice(self.source_junctions if source_only else self.junctions, size=n)
		]		
