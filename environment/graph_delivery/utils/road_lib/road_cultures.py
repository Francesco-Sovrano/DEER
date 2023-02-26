from ..culture_lib.culture import Culture, Argument
from .road_cell import RoadCell
from .road_agent import RoadAgent
import numpy as np
import copy

#####################
# EASY ROAD CULTURE #
#####################

class RoadCulture(Culture):
	starting_argument_id = 0

	def __init__(self):
		super().__init__()

	def initialise_random_agent(self, agent: RoadAgent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		agent.assign_property_value("Speed", 0)

	def initialise_feasible_road(self, road: RoadCell):
		for p in self.properties.keys():
			road.assign_property_value(p, False)

	def run_default_dialogue(self, road, agent, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			road: RoadCell corresponding to destination cell.
			agent: RoadAgent corresponding to agent.
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# Game starts with proponent using argument 0 ("I will not get a ticket").
		return super().run_dialogue(road, agent, starting_argument_id=self.starting_argument_id, explanation_type=explanation_type)

	def get_minimum_speed(self, road, agent):
		agent = copy.deepcopy(agent)
		for speed in [0,10,20,30,40]:
			agent.assign_property_value("Speed", speed)
			can_move, _ = self.run_default_dialogue(road, agent, explanation_type="compact")
			if can_move:
				return speed
		return None

	def get_speed_limits(self, road, agent):
		agent = copy.deepcopy(agent)
		min_speed = self.get_minimum_speed(road, agent)
		if min_speed is None:
			return (None,None) # (None,None) if road is unfeasible
		max_speed = None
		step = 10
		for speed in range(min_speed+step, self.agent_options.get('speed',120)+1, step):
			agent.assign_property_value("Speed", speed)
			can_move, _ = self.run_default_dialogue(road, agent, explanation_type="compact")
			if can_move:
				if max_speed is None or speed > max_speed:
					max_speed = speed
		if max_speed is None:
			max_speed = min_speed
		return (min_speed, max_speed)

class EasyRoadCulture(RoadCulture):
	def __init__(self, road_options=None, agent_options=None):
		if road_options is None: road_options = {}
		if agent_options is None: agent_options = {}
		self.road_options = road_options
		self.agent_options = agent_options
		self.ids = {}
		super().__init__()
		self.name = "Easy Road Culture"
		# Properties of the culture with their default values go in self.properties.
		self.properties = {"Motorway": False,
						   "Stop Sign": False}

		self.agent_properties = {"Speed": 0}

	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		args = []

		_id = 0
		motion = Argument(_id, "I will not get a ticket.")
		self.ids["no_ticket"] = _id
		motion.set_verifier(lambda *gen: True)  # Propositional arguments are always valid.
		args.append(motion)

		_id += 1
		arg1 = Argument(_id, "This is a motorway.")
		self.ids["is_motorway"] = _id
		def arg1_verifier(road: RoadCell, agent: RoadAgent):
			return road["Motorway"] is True
		arg1.set_verifier(arg1_verifier)
		args.append(arg1)

		_id += 1
		arg2 = Argument(_id, "There is a stop sign.")
		self.ids["has_stop_sign"] = _id
		def arg2_verifier(road: RoadCell, agent: RoadAgent):
			return road["Stop Sign"] is True
		arg2.set_verifier(arg2_verifier)
		args.append(arg2)

		_id += 1
		speed0 = Argument(_id, "My speed is 0.")
		self.ids["speed==0"] = _id
		def speed0_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 0
		speed0.set_verifier(speed0_verifier)
		args.append(speed0)

		_id += 1
		speed70 = Argument(_id, "My speed is 70 or less.")
		self.ids["speed<=70"] = _id
		def speed70_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 70
		speed70.set_verifier(speed70_verifier)
		args.append(speed70)

		self.AF.add_arguments(args)

	def initialise_random_road(self, road: RoadCell, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		motorway = True if np_random.random() <= self.road_options.get('motorway',1/2) else False
		road.assign_property_value("Motorway", motorway)

		if motorway:
			road.assign_property_value("Stop Sign", False)
		else:
			stop_sign = True if np_random.random() <= self.road_options.get('stop_sign',1/2) else False
			road.assign_property_value("Stop Sign", stop_sign)

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		"""
		ID = self.ids

		self.AF.add_attack(ID["is_motorway"], ID["no_ticket"])
		self.AF.add_attack(ID["has_stop_sign"], ID["no_ticket"])
		self.AF.add_attack(ID["has_stop_sign"], ID["is_motorway"])
		self.AF.add_attack(ID["speed==0"], ID["has_stop_sign"])
		self.AF.add_attack(ID["speed<=70"], ID["is_motorway"])

#######################
# MEDIUM ROAD CULTURE #
#######################

class MediumRoadCulture(RoadCulture):
	def __init__(self, road_options=None, agent_options=None):
		if road_options is None: road_options = {}
		if agent_options is None: agent_options = {}
		self.road_options = road_options
		self.agent_options = agent_options
		self.ids = {}
		super().__init__()
		self.name = "Medium Road Culture"
		# Properties of the culture with their default values go in self.properties.
		self.properties = {"Motorway": False,
						   "Stop Sign": False,
						   "School": False,
						   "Single Lane": False,
						   "Town Road": False
						   }

		self.agent_properties = {"Speed": 0,
								 "Emergency Vehicle": False}

	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		args = []

		_id = 0
		motion = Argument(_id, "I will not get a ticket.")
		self.ids["no_ticket"] = _id
		motion.set_verifier(lambda *gen: True)  # Propositional arguments are always valid.
		args.append(motion)

		_id += 1
		arg1 = Argument(_id, "This is a motorway.")
		self.ids["is_motorway"] = _id
		def arg1_verifier(road: RoadCell, agent: RoadAgent):
			return road["Motorway"] is True
		arg1.set_verifier(arg1_verifier)
		args.append(arg1)

		_id += 1
		arg2 = Argument(_id, "There is a stop sign.")
		self.ids["has_stop_sign"] = _id
		def arg2_verifier(road: RoadCell, agent: RoadAgent):
			return road["Stop Sign"] is True
		arg2.set_verifier(arg2_verifier)
		args.append(arg2)

		_id += 1
		arg3 = Argument(_id, "There is a school nearby.")
		self.ids["has_school"] = _id
		def arg3_verifier(road: RoadCell, agent: RoadAgent):
			return road["School"] is True
		arg3.set_verifier(arg3_verifier)
		args.append(arg3)

		_id += 1
		arg4 = Argument(_id, "This is a single lane road.")
		self.ids["single_lane"] = _id
		def arg4_verifier(road: RoadCell, agent: RoadAgent):
			return road["Single Lane"] is True
		arg4.set_verifier(arg4_verifier)
		args.append(arg4)

		_id += 1
		arg5 = Argument(_id, "This is a town road.")
		self.ids["town_road"] = _id
		def arg5_verifier(road: RoadCell, agent: RoadAgent):
			return road["Town Road"] is True
		arg5.set_verifier(arg5_verifier)
		args.append(arg5)

		_id += 1
		speed0 = Argument(_id, "My speed is 0.")
		self.ids["speed==0"] = _id
		def speed0_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 0
		speed0.set_verifier(speed0_verifier)
		args.append(speed0)

		_id += 1
		speed20 = Argument(_id, "My speed is 20 or less.")
		self.ids["speed<=20"] = _id
		def speed20_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 20
		speed20.set_verifier(speed20_verifier)
		args.append(speed20)

		_id += 1
		speed30 = Argument(_id, "My speed is 30 or less.")
		self.ids["speed<=30"] = _id
		def speed30_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 30
		speed30.set_verifier(speed30_verifier)
		args.append(speed30)

		_id += 1
		speed60 = Argument(_id, "My speed is 60 or less.")
		self.ids["speed<=60"] = _id
		def speed60_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 60
		speed60.set_verifier(speed60_verifier)
		args.append(speed60)

		_id += 1
		speed70 = Argument(_id, "My speed is 70 or less.")
		self.ids["speed<=70"] = _id
		def speed70_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Speed"] <= 70
		speed70.set_verifier(speed70_verifier)
		args.append(speed70)

		_id += 1
		emergency = Argument(_id, "I am an emergency vehicle.")
		self.ids["emergency_vehicle"] = _id
		def emergency_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Emergency Vehicle"] is True
		emergency.set_verifier(emergency_verifier)
		args.append(emergency)

		self.AF.add_arguments(args)

	def initialise_random_road(self, road: RoadCell, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		motorway = True if np_random.random() <= self.road_options.get('motorway',1/2) else False
		road.assign_property_value("Motorway", motorway)

		if motorway:
			road.assign_property_value("School", False)
			road.assign_property_value("Town Road", False)
			road.assign_property_value("Stop Sign", False)
		else:
			stop_sign = True if np_random.random() <= self.road_options.get('stop_sign',1/2) else False
			road.assign_property_value("Stop Sign", stop_sign)

			school = True if np_random.random() <= self.road_options.get('school',1/2) else False
			road.assign_property_value("School", school)

			town_road = True if np_random.random() <= self.road_options.get('town_road',1/2) else False
			road.assign_property_value("Town Road", town_road)

		single_lane = True if np_random.random() <= self.road_options.get('single_lane',1/2) else False
		road.assign_property_value("Single Lane", single_lane)

	def initialise_random_agent(self, agent: RoadAgent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		emergency_vehicle = True if np_random.random() <= self.agent_options.get('emergency_vehicle',1/5) else False
		agent.assign_property_value("Emergency Vehicle", emergency_vehicle)

		super().initialise_random_agent(agent, np_random)

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		"""
		ID = self.ids

		# Base conditions for ticket.
		self.AF.add_attack(ID["is_motorway"], ID["no_ticket"])
		self.AF.add_attack(ID["has_stop_sign"], ID["no_ticket"])
		self.AF.add_attack(ID["has_school"], ID["no_ticket"])
		self.AF.add_attack(ID["single_lane"], ID["no_ticket"])
		self.AF.add_attack(ID["town_road"], ID["no_ticket"])

		# Speed checks and mitigatory rules.
		self.AF.add_attack(ID["speed==0"], ID["has_stop_sign"])
		self.AF.add_attack(ID["speed<=20"], ID["has_school"])
		self.AF.add_attack(ID["speed<=30"], ID["town_road"])
		self.AF.add_attack(ID["speed<=60"], ID["single_lane"])
		self.AF.add_attack(ID["speed<=70"], ID["is_motorway"])

		self.AF.add_attack(ID["emergency_vehicle"], ID["has_stop_sign"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["has_school"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["town_road"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["single_lane"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["is_motorway"])

#####################
# HARD ROAD CULTURE #
#####################

class HardRoadCulture(RoadCulture):
	def __init__(self, road_options=None, agent_options=None):
		if road_options is None: road_options = {}
		if agent_options is None: agent_options = {}
		self.road_options = road_options
		self.agent_options = agent_options
		self.ids = {}
		super().__init__()
		self.name = "Hard Road Culture"
		# Properties of the culture with their default values go in self.properties.
		self.properties = {"Motorway": False,
						   "Stop Sign": False,
						   "School": False,
						   "Single Lane": False,
						   "Town Road": False,
						   "Roadworks": False,
						   "Accident": False,
						   "Heavy Rain": False,
						   "Congestion Charge": False
						   }

		self.agent_properties = {"Speed": 0,
								 "Emergency Vehicle": False,
								 "Heavy Vehicle": False,
								 "Worker Vehicle": False,
								 "Tasked": False,
								 "Paid Charge": False}

	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		args = []

		_id = 0
		motion = Argument(_id, "You will not get a ticket.")
		self.ids["no_ticket"] = _id
		motion.set_verifier(lambda *gen: True)  # Propositional arguments are always valid.
		args.append(motion)

		_id += 1
		arg1 = Argument(_id, "You are driving on a motorway with speed above 70.")
		self.ids["motorway_above_70"] = _id
		def arg1_verifier(road: RoadCell, agent: RoadAgent):
			return road["Motorway"] is True and agent["Speed"] > 70
		arg1.set_verifier(arg1_verifier)
		args.append(arg1)

		_id += 1
		agent11 = Argument(_id, "You are an emergency vehicle.")
		self.ids["emergency_vehicle"] = _id
		def agent11_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Emergency Vehicle"] is True
		agent11.set_verifier(agent11_verifier)
		args.append(agent11)

		_id += 1
		agent1 = Argument(_id, "You are a tasked emergency vehicle.")
		self.ids["tasked_emergency_vehicle"] = _id
		def agent1_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Emergency Vehicle"] is True and agent["Tasked"] is True
		agent1.set_verifier(agent1_verifier)
		args.append(agent1)

		_id += 1
		arg3 = Argument(_id, "You are driving on a motorway with speed below 30.")
		self.ids["motorway_below_30"] = _id
		def arg3_verifier(road: RoadCell, agent: RoadAgent):
			return road["Motorway"] is True and agent["Speed"] <= 30
		arg3.set_verifier(arg3_verifier)
		args.append(arg3)

		_id += 1
		arg7 = Argument(_id, "There is an accident ahead.")
		self.ids["accident"] = _id
		def arg7_verifier(road: RoadCell, agent: RoadAgent):
			return road["Accident"] is True
		arg7.set_verifier(arg7_verifier)
		args.append(arg7)

		_id += 1
		arg71 = Argument(_id, "There is a stop sign ahead.")
		self.ids["stop_sign"] = _id
		def arg71_verifier(road: RoadCell, agent: RoadAgent):
			return road["Accident"] is True
		arg71.set_verifier(arg71_verifier)
		args.append(arg71)

		_id += 1
		arg4 = Argument(_id, "You are driving on a single lane road with speed above 60.")
		self.ids["single_lane_above_60"] = _id
		def arg4_verifier(road: RoadCell, agent: RoadAgent):
			return road["Single Lane"] is True and agent["Speed"] > 60
		arg4.set_verifier(arg4_verifier)
		args.append(arg4)

		_id += 1
		arg5 = Argument(_id, "You are driving on a town road with speed above 30.")
		self.ids["town_road_above_30"] = _id
		def arg5_verifier(road: RoadCell, agent: RoadAgent):
			return road["Town Road"] is True and agent["Speed"] > 30
		arg5.set_verifier(arg5_verifier)
		args.append(arg5)

		_id += 1
		arg51 = Argument(_id, "You are driving on a school road with speed above 20.")
		self.ids["school_road_above_20"] = _id
		def arg51_verifier(road: RoadCell, agent: RoadAgent):
			return road["School"] is True and agent["Speed"] > 20
		arg51.set_verifier(arg51_verifier)
		args.append(arg51)

		_id += 1
		arg6 = Argument(_id, "You drove into roadworks.")
		self.ids["roadworks"] = _id
		def arg6_verifier(road: RoadCell, agent: RoadAgent):
			return road["Roadworks"] is True
		arg6.set_verifier(arg6_verifier)
		args.append(arg6)

		_id += 1
		arg61 = Argument(_id, "You are a worker vehicle driving with speed below 30.")
		self.ids["worker_below_30"] = _id
		def arg61_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Worker Vehicle"] is True and agent["Speed"] <= 30 and agent["Tasked"] is True
		arg61.set_verifier(arg61_verifier)
		args.append(arg61)

		_id += 1
		arg62 = Argument(_id, "There is a stop sign and your speed is above 0.")
		self.ids["stop_sign_above_0"] = _id
		def arg62_verifier(road: RoadCell, agent: RoadAgent):
			return road["Stop Sign"] is True and agent["Speed"] > 0
		arg62.set_verifier(arg62_verifier)
		args.append(arg62)

		_id += 1
		arg63 = Argument(_id, "There is an accident and your speed is below 20.")
		self.ids["accident_below_20"] = _id
		def arg63_verifier(road: RoadCell, agent: RoadAgent):
			return road["Accident"] is True and agent["Speed"] <= 20
		arg63.set_verifier(arg63_verifier)
		args.append(arg63)

		_id += 1
		arg64 = Argument(_id, "You are driving a heavy vehicle at speed above 50.")
		self.ids["heavy_above_50"] = _id
		def arg64_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Heavy Vehicle"] is True and agent["Speed"] > 50
		arg64.set_verifier(arg64_verifier)
		args.append(arg64)

		_id += 1
		arg644 = Argument(_id, "It is raining heavily and your speed is above 60.")
		self.ids["rain_above_60"] = _id
		def arg644_verifier(road: RoadCell, agent: RoadAgent):
			return road["Heavy Rain"] is True and agent["Speed"] > 60
		arg644.set_verifier(arg644_verifier)
		args.append(arg644)

		_id += 1
		arg65 = Argument(_id, "There is a congestion charge which hasn't been paid.")
		self.ids["congestion_charge_not_paid"] = _id
		def arg65_verifier(road: RoadCell, agent: RoadAgent):
			return road["Congestion Charge"] is True and agent["Paid Charge"] is False
		arg65.set_verifier(arg65_verifier)
		args.append(arg65)

		_id += 1
		arg8 = Argument(_id, "It is raining heavily.")
		self.ids["heavy_rain"] = _id
		def arg8_verifier(road: RoadCell, agent: RoadAgent):
			return road["Heavy Rain"] is True
		arg8.set_verifier(arg8_verifier)
		args.append(arg8)

		_id += 1
		agent2 = Argument(_id, "I am a heavy vehicle.")
		self.ids["heavy_vehicle"] = _id
		def agent2_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Heavy Vehicle"] is True
		agent2.set_verifier(agent2_verifier)
		args.append(agent2)
		
		_id += 1
		agent3 = Argument(_id, "I am a worker vehicle.")
		self.ids["worker_vehicle"] = _id
		def agent3_verifier(road: RoadCell, agent: RoadAgent):
			return agent["Worker Vehicle"] is True
		agent3.set_verifier(agent3_verifier)
		args.append(agent3)

		self.AF.add_arguments(args)

	def initialise_random_road(self, road: RoadCell, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		motorway = True if np_random.random() <= self.road_options.get('motorway',1/2) else False
		road.assign_property_value("Motorway", motorway)

		if motorway:
			road.assign_property_value("School", False)
			road.assign_property_value("Town Road", False)
			road.assign_property_value("Stop Sign", False)
		else:
			school = True if np_random.random() <= self.road_options.get('school',1/2) else False
			road.assign_property_value("School", school)

			town_road = True if np_random.random() <= self.road_options.get('town_road',1/2) else False
			road.assign_property_value("Town Road", town_road)

			stop_sign = True if np_random.random() <= self.road_options.get('stop_sign',1/2) else False
			road.assign_property_value("Stop Sign", stop_sign)

		single_lane = True if np_random.random() <= self.road_options.get('single_lane',1/2) else False
		road.assign_property_value("Single Lane", single_lane)

		roadworks = True if np_random.random() <= self.road_options.get('roadworks',1/2) else False
		road.assign_property_value("Roadworks", roadworks)

		accident = True if np_random.random() <= self.road_options.get('accident',1/8) else False
		road.assign_property_value("Accident", accident)

		heavy_rain = True if np_random.random() <= self.road_options.get('heavy_rain',1/2) else False
		road.assign_property_value("Heavy Rain", heavy_rain)

		congestion_charge = True if np_random.random() <= self.road_options.get('congestion_charge',1/2) else False
		road.assign_property_value("Congestion Charge", congestion_charge)

	def initialise_random_agent(self, agent: RoadAgent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		emergency_vehicle = True if np_random.random() <= self.agent_options.get('emergency_vehicle',1/5) else False
		agent.assign_property_value("Emergency Vehicle", emergency_vehicle)

		heavy_vehicle = True if np_random.random() <= self.agent_options.get('heavy_vehicle',1/4) else False
		agent.assign_property_value("Heavy Vehicle", heavy_vehicle)

		worker_vehicle = True if np_random.random() <= self.agent_options.get('worker_vehicle',1/3) else False
		agent.assign_property_value("Worker Vehicle", worker_vehicle)

		tasked = True if np_random.random() <= self.agent_options.get('tasked',1/2) else False
		agent.assign_property_value("Tasked", tasked)

		paid_charge = True if np_random.random() <= self.agent_options.get('paid_charge',1/2) else False
		agent.assign_property_value("Paid Charge", paid_charge)

		super().initialise_random_agent(agent, np_random)

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		Culture can be seen here:
		https://docs.google.com/document/d/1O7LCeRVVyCFnP-_8PVcfNrEdVEN5itGxcH1Ku6GN5MQ/edit?usp=sharing
		"""
		ID = self.ids

		# 1
		self.AF.add_attack(ID["motorway_above_70"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["motorway_above_70"])

		# 2
		self.AF.add_attack(ID["motorway_below_30"], ID["no_ticket"])
		self.AF.add_attack(ID["heavy_vehicle"], ID["motorway_below_30"])
		self.AF.add_attack(ID["accident"], ID["motorway_below_30"])
		self.AF.add_attack(ID["stop_sign"], ID["motorway_below_30"])

		# 3
		self.AF.add_attack(ID["single_lane_above_60"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["single_lane_above_60"])

		# 4
		self.AF.add_attack(ID["town_road_above_30"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["town_road_above_30"])

		# 5
		self.AF.add_attack(ID["school_road_above_20"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["school_road_above_20"])

		# 6
		self.AF.add_attack(ID["stop_sign_above_0"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["stop_sign_above_0"])

		# 7
		self.AF.add_attack(ID["accident_below_20"], ID["no_ticket"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["accident_below_20"])
		self.AF.add_attack(ID["stop_sign_above_0"], ID["accident_below_20"])
		
		# 8
		self.AF.add_attack(ID["heavy_above_50"], ID["no_ticket"])
		
		# 9
		self.AF.add_attack(ID["rain_above_60"], ID["no_ticket"])

		# 10
		self.AF.add_attack(ID["congestion_charge_not_paid"], ID["no_ticket"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["congestion_charge_not_paid"])
		self.AF.add_attack(ID["worker_vehicle"], ID["congestion_charge_not_paid"])
		self.AF.add_attack(ID["heavy_rain"], ID["congestion_charge_not_paid"])

		# 11
		self.AF.add_attack(ID["roadworks"], ID["no_ticket"])
		self.AF.add_attack(ID["worker_below_30"], ID["roadworks"])
		self.AF.add_attack(ID["tasked_emergency_vehicle"], ID["roadworks"])

