
class Argument:
    def __init__(self, arg_id, descriptive_text):
        self.arg_id = arg_id
        self.descriptive_text = descriptive_text
        self.framework = None
        self.evidence = []
        self.verifier_function = None

    def id(self):
        return self.arg_id

    def set_framework(self, framework):
        self.framework = framework

    def add_evidence(self, evidence):
        self.evidence.append(evidence)

    def attacks(self, attacked):
        if type(attacked) is Argument:
            self.framework.add_argument(self)
            self.framework.add_argument(attacked)
            attacked_id = attacked.id()
        elif type(attacked) is int:
            attacked_id = attacked
        else:
            print("Argument::attacks: Invalid type for argument!")
            return
        self.framework.add_attack(self.arg_id, attacked_id)

    def set_verifier(self, verifier):
        self.verifier_function = verifier
        pass

    def verifier(self):
        return self.verifier_function

    def verify(self, me, they):
        if self.verifier_function is not None:
            return self.verifier_function(me, they)


class ArgumentationFramework:
    def __init__(self):
        self.all_arguments = {}
        self.all_attacks = {}
        self.all_attacked_by = {}

    def add_arguments(self, arguments: list):
        for arg in arguments:
            self.add_argument(arg)

    def add_argument(self, argument):
        self.all_arguments[argument.id()] = argument
        argument.set_framework(self)

    def add_attack(self, attacker_id, attacked_id):
        if self.all_attacks.get(attacker_id, None) is None:
            self.all_attacks[attacker_id] = set()
        if self.all_attacked_by.get(attacked_id, None) is None:
            self.all_attacked_by[attacked_id] = set()
        self.all_attacks[attacker_id].add(attacked_id)
        self.all_attacked_by[attacked_id].add(attacker_id)

    def arguments_that_attack(self, argument):
        if isinstance(argument, list):
            return self.arguments_that_attack_list(argument)
        return self.all_attacked_by.get(argument, set())

    def arguments_that_attack_list(self, argument_list):
        result = set()
        for argument_id in argument_list:
            result.update(self.arguments_that_attack(argument_id))
        return result

    def arguments_attacked_by_list(self, argument_list):
        result = set()
        for argument_id in argument_list:
            result.update(self.arguments_attacked_by(argument_id))
        return result

    def arguments_attacked_by(self, argument):
        if isinstance(argument, list):
            return self.arguments_attacked_by_list(argument)
        return self.all_attacks.get(argument, set())

    def argument(self, argument_id):
        return self.all_arguments[argument_id]




