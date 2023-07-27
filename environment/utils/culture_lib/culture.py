try:
	from environment.utils.culture_lib.argument import Argument, ArgumentationFramework
except Exception as e:
	print('Warning: graph-tool not installed, using old Argumentation Framework.')
	from environment.utils.culture_lib.argument_old import Argument, ArgumentationFramework

class Culture:
	def __init__(self):
		self.AF = ArgumentationFramework()
		self.properties = {}
		self.name = None

		self.create_arguments()
		self.define_attacks()

	def create_arguments(self):
		pass


	def define_attacks(self):
		pass

	def run_dialogue(self, agent_1, agent_2, starting_argument_id=0, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# print("@@@@@@@@@@@@@ NEW DIALOGUE @@@@@@@@@@@@@")
		AF = self.AF
		verified = set()

		# Prune temporary AF out of unverified arguments
		to_remove = []
		for argument_id in AF.all_arguments:
			argument_obj = AF.argument(argument_id)
			if argument_obj.verify(agent_1, agent_2):
				verified.add(argument_id)

		# Game starts with proponent using argument 0 ("I will not get a ticket").
		used_arguments = {"opponent": set(), "proponent": {starting_argument_id}}
		last_argument = {"opponent": set(), "proponent": {starting_argument_id}}

		dialogue_history = []
		dialogue_history.append(last_argument["proponent"])

		# Odd turns: opponent. Even turns: proponent.
		turn = 1
		game_over = False
		winner = None
		while not game_over:
			# print("##### TURN {} #####".format(turn))
			if turn % 2:
				# Opponent's turn.
				current_player = "opponent"
				next_player = "proponent"
			else:
				# Proponent's turn.
				current_player = "proponent"
				next_player = "opponent"
			turn += 1
			# Remove previously used arguments.
			all_used_arguments = used_arguments["proponent"] | used_arguments["opponent"]
			forbidden_arguments = set(all_used_arguments)
			# Cannot pick argument that is attacked by previously used argument.
			forbidden_arguments.update(AF.arguments_attacked_by_list(list(all_used_arguments)))
			# print("All used arguments: {}".format(all_used_arguments))
			# print("Forbidden arguments: {}".format(forbidden_arguments))
			# Use all arguments as possible.
			all_viable_arguments = set(AF.arguments_that_attack(list(last_argument[next_player])))
			# print("Viable arguments: {}".format(all_viable_arguments))
			verified_attacks = verified.intersection(all_viable_arguments)
			# print("Verified attacks: {}".format(verified_attacks))
			targets = set(AF.arguments_attacked_by_list(list(verified_attacks)))
			if last_argument[next_player].issubset(targets):
				used_arguments[current_player].update(verified_attacks)
				last_argument[current_player] = verified_attacks
				# print("{} used arguments {}".format(current_player, verified_attacks))
				dialogue_history.append(verified_attacks)
			else:
				game_over = True
				winner = next_player
				# print("GAME OVER! {} wins".format(winner))

		motion_validated = True if winner == "proponent" else False

		# Building the explanation.

		if explanation_type == "verbose":
			turn = 0
			explanation_list = []
			for argument_list in dialogue_history:
				argument_explanation = "CON: " if turn % 2 else "PRO: "
				argument_explanation += ' / '.join(sorted((
					AF.argument(argument_id).descriptive_text
					for argument_id in argument_list
					if argument_id != starting_argument_id # motion_validated is already telling whether the ground_argument has been won or lost
				)))
				turn += 1
				explanation_list.append(argument_explanation)
		else:
			explanation_list = [
				AF.argument(argument_id).descriptive_text
				for argument_id in last_argument[winner]
				if argument_id != starting_argument_id # motion_validated is already telling whether the ground_argument has been won or lost
			]

		return motion_validated, explanation_list
