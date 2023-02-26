from .culture_lib.culture import Culture, Argument
# from .road_lib.road_cell import RoadCell
# from .road_lib.road_agent import RoadAgent

class HeterogeneityCulture(Culture):
	starting_argument_id = 0

	def __init__(self, road_options=None, agent_options=None):
		if road_options is None: road_options = {}
		if agent_options is None: agent_options = {}
		self.road_options = road_options
		self.agent_options = agent_options
		self.ids = {}
		super().__init__()
		self.name = "Heterogeneity Culture"
		# Properties of the culture with their default values go in self.env_properties.
		self.env_properties = {
			"Require Special Permission": False,
			"Accident": False,
			"Require Fee": False
		}

		self.agent_properties = {
			"Emergency Vehicle": False,
			"Has Special Permission": False,
			"Can Pay Fee": False
		}

	def initialise_feasible_road(self, road):
		for p in self.env_properties.keys():
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

	def initialise_random_road(self, road, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		road.assign_property_value("Require Special Permission", np_random.random() <= self.road_options.get('require_special_permission',1/2))
		road.assign_property_value("Accident", np_random.random() <= self.road_options.get('accident',1/8))
		road.assign_property_value("Require Fee", np_random.random() <= self.road_options.get('require_fee',1/2))

	def initialise_random_agent(self, agent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		agent.assign_property_value("Emergency Vehicle", np_random.random() <= self.agent_options.get('emergency_vehicle',1/5))
		agent.assign_property_value("Has Special Permission", np_random.random() <= self.agent_options.get('has_special_permission',1/2))
		agent.assign_property_value("Can Pay Fee", np_random.random() <= self.agent_options.get('can_pay_fee',1/2))
	
	def build_argument(self, _id, _label, _description, _verifier_fn):
		motion = Argument(_id, _description)
		self.ids[_label] = _id
		motion.set_verifier(_verifier_fn)  # Propositional arguments are always valid.
		return motion

	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		self.AF.add_arguments([
			self.build_argument(0, "ok", "Nothing wrong.", lambda *gen: True), # Propositional arguments are always valid.
			self.build_argument(1, "emergency_vehicle", "You are an emergency vehicle.", lambda road, agent: agent["Emergency Vehicle"] is True),
			self.build_argument(2, "has_special_permission", "You have the special permission.", lambda road, agent: agent["Has Special Permission"] is True),
			self.build_argument(3, "accident", "There is an accident ahead.", lambda road, agent: road["Accident"] is True),
			self.build_argument(4, "required_special_permission", "You drove into a road that requires a special permission.", lambda road, agent: road["Require Special Permission"] is True),
			self.build_argument(5, "required_fee", "You drove into a road that requires a fee.", lambda road, agent: road["Require Fee"] is True),
			self.build_argument(6, "can_pay_fee", "You can pay the fee.", lambda road, agent: agent["Can Pay Fee"] is True),
		])

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		Culture can be seen here:
		https://docs.google.com/document/d/1O7LCeRVVyCFnP-_8PVcfNrEdVEN5itGxcH1Ku6GN5MQ/edit?usp=sharing
		"""
		ID = self.ids

		# 1
		self.AF.add_attack(ID["required_fee"], ID["ok"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["required_fee"])
		self.AF.add_attack(ID["can_pay_fee"], ID["required_fee"])

		# 2
		self.AF.add_attack(ID["required_special_permission"], ID["ok"])
		self.AF.add_attack(ID["has_special_permission"], ID["required_special_permission"])

		# 3
		self.AF.add_attack(ID["accident"], ID["ok"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["accident"])

class ComplexHeterogeneityCulture(HeterogeneityCulture):

	def __init__(self, road_options=None, agent_options=None):
		super().__init__(road_options=road_options, agent_options=agent_options)
		self.name = "Complex Heterogeneity Culture"
		# Properties of the culture with their default values go in self.env_properties.
		self.env_properties.update({
			"Require Electric": False,
		})

		self.agent_properties.update({
			"Electric Vehicle": False,
			"Expired Car Tax": False,
		})

	def initialise_random_road(self, road, np_random):
		"""
		Receives an empty RoadCell and initialises properties with acceptable random values.
		:param road: uninitialised RoadCell.
		"""
		super().initialise_random_road(road, np_random)
		road.assign_property_value("Require Electric", np_random.random() <= self.road_options.get('require_electric',1/6))

	def initialise_random_agent(self, agent, np_random):
		"""
		Receives an empty RoadAgent and initialises properties with acceptable random values.
		:param agent: uninitialised RoadAgent.
		"""
		super().initialise_random_agent(agent, np_random)
		agent.assign_property_value("Electric Vehicle", np_random.random() <= self.agent_options.get('electric_vehicle',1/3))
		agent.assign_property_value("Expired Car Tax", np_random.random() <= self.agent_options.get('expired_car_tax',1/20))
	
	def create_arguments(self):
		"""
		Defines set of arguments present in the culture and their verifier functions.
		"""
		super().create_arguments()
		self.AF.add_arguments([
			self.build_argument(7, "is_electric", "You are an electric vehicle.", lambda road, agent: agent["Electric Vehicle"] is True),
			self.build_argument(8, "required_electric", "You drove into a road for electric vehicles only.", lambda road, agent: road["Require Electric"] is True),
			self.build_argument(9, "expired_car_tax", "Your car tax has expired.", lambda road, agent: agent["Expired Car Tax"] is True),
		])

	def define_attacks(self):
		"""
		Defines attack relationships present in the culture.
		"""
		super().define_attacks()
		ID = self.ids

		# 1
		self.AF.add_attack(ID["is_electric"], ID["required_fee"])

		# 2
		self.AF.add_attack(ID["expired_car_tax"], ID["has_special_permission"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["expired_car_tax"])

		# 4
		self.AF.add_attack(ID["required_electric"], ID["ok"])
		self.AF.add_attack(ID["is_electric"], ID["required_electric"])
		self.AF.add_attack(ID["emergency_vehicle"], ID["required_electric"])

