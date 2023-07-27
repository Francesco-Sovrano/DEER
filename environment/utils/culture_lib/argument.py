# import graph_tool.all as gt
import subprocess
import string
import re

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


class PrivateArgument(Argument):
    def __init__(self, arg_id, descriptive_text, privacy_cost):
        super(PrivateArgument, self).__init__(arg_id, descriptive_text)
        self.privacy_cost = privacy_cost


class ArgumentationFramework:
    def __init__(self):
        self.all_arguments = {}
        self.all_attacks = {}
        self.all_attacked_by = {}
        self.argument_strength = {}
        self.least_attacked = []
        self.strongest_attackers = []

    def add_arguments(self, arguments: list):
        for arg in arguments:
            self.add_argument(arg)

    def arguments(self):
        return self.all_arguments.values()

    def argument_ids(self):
        return self.all_arguments.keys()

    def attacks(self):
        return self.all_attacks

    def attacked_by(self):
        return self.all_attacked_by

    def remove_argument(self, argument_id):
        if argument_id in self.all_arguments.keys():
            del self.all_arguments[argument_id]
        if argument_id in self.all_attacks.keys():
            del self.all_attacks[argument_id]
        if argument_id in self.all_attacked_by.keys():
            del self.all_attacked_by[argument_id]
        for id, attacked_set in self.all_attacks.items():
            if argument_id in attacked_set:
                attacked_set.remove(argument_id)
        for id, attacker_set in self.all_attacked_by.items():
            if argument_id in attacker_set:
                attacker_set.remove(argument_id)

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

    def compute_rank_arguments_occurrence(self, semantics="EE-PR"):
        """
        Calls ConArg as an external process to compute extensions.
        Returns a normalised "argument strength" value denoted by occurrences/num_extensions.
        :param semantics: The type of semantics to be considered.
        :return: Argument strengths as percentage of occurrence.
        """
        result_string = self.run_solver(semantics)
        result_string = result_string.replace("[", "")
        result_string = result_string.replace("]", "")
        result_string = result_string.replace("\t", "")
        match = result_string.split("\n")
        occurrences = {}
        for argument_obj in self.arguments():
            occurrences[argument_obj.id()] = 0
        for m in match:
            for argument_obj in self.arguments():
                arg_id = argument_obj.id()
                # FIXME: Finding individual digits in strings, flawed counting
                if str(arg_id) in m.split(","):
                    occurrences[arg_id] += 1
        num_extensions = len(match)

        argument_strength = {}
        for id, count in occurrences.items():
            if num_extensions == 0:
                num_extensions = 1
            argument_strength[id] = count / num_extensions

        self.argument_strength = argument_strength

    def run_solver(self, semantics="EE-PR", arg_str=""):
        with open('sample.apx',  'w') as file:
            file.write(self.to_aspartix_id())

        # subprocess.run(["conarg_x64/conarg2", "-w dung", "-e admissible", "-c 4", "sample.apx"])
        if not arg_str:
            result = subprocess.run(["mu-toksia/mu-toksia", "-p", semantics, "-fo", "apx", "-f", "sample.apx"],
                                    capture_output=True, text=True)
        else:
            result = subprocess.run(["mu-toksia/mu-toksia", "-p", semantics, "-fo", "apx", "-f", "sample.apx",
                                     "-a", arg_str],
                                    capture_output=True, text=True)

        return result.stdout

    def rank_least_attacked_arguments(self):
        """
        :return: List of argument ids in ascending order of attacks received.
        """
        rank = {}
        for arg_id in self.all_arguments:
            rank[arg_id] = 0
        for arg_id, attackers in self.all_attacked_by.items():
            rank[arg_id] = len(attackers)
        self.least_attacked = sorted(rank, key=rank.get)

    def rank_strongest_attacker_arguments(self):
        """
        :return: List of argument ids in ascending order of attacks received.
        """
        rank = {}
        for arg_id in self.all_arguments:
            rank[arg_id] = 0
        for arg_id, attacks in self.all_attacks.items():
            rank[arg_id] = len(attacks)
        self.strongest_attackers = sorted(rank, key=rank.get, reverse=True)

    def to_aspartix_id(self):
        text = ""
        for argument in self.all_arguments:
            text += "arg({}).\n".format(argument)
        for attacker in self.all_attacks.keys():
            for attacked in self.all_attacks[attacker]:
                text += "att({},{}).\n".format(attacker, attacked)
        return text

    def to_aspartix_text(self):
        text = ""
        for argument_id in self.all_arguments:
            arg_text = self.argument(argument_id).descriptive_text()
            text += "arg({}).\n".format(arg_text)
        for attacker_id in self.all_attacks.keys():
            for attacked_id in self.all_attacks[attacker_id]:
                attacker_text = self.argument(attacker_id).descriptive_text()
                attacked_text = self.argument(attacked_id).descriptive_text()
                text += "att({},{}).\n".format(attacker_text, attacked_text)
        return text

    # def to_graph_tool(self):
    #     g = gt.Graph(directed=True)
    #     ref = {}
    #     g.vp.id = g.new_vertex_property("int")
    #     g.vp.privacy = g.new_vertex_property("int")
    #     g.vp.arg_obj = g.new_vertex_property("object")
    #     for argument_id, argument_obj in self.all_arguments.items():
    #         v = g.add_vertex()
    #         ref[argument_id] = v
    #         g.vp.id[v] = argument_id
    #         g.vp.privacy[v] = argument_obj.privacy_cost
    #         g.vp.arg_obj[v] = argument_obj
    #     for attacker, attacked_set in self.all_attacks.items():
    #         for attacked in attacked_set:
    #             source = ref[attacker]
    #             target = ref[attacked]
    #             g.add_edge(source, target)
    #     return g

    def from_graph_tool(self, g):
        self.all_arguments = {}
        self.all_attacks = {}
        self.all_attacked_by = {}
        ref = {}
        for v in g.vertices():
            id = g.vp.id[v]
            privacy = g.vp.privacy[v]
            obj = g.vp.arg_obj[v]
            new_arg = PrivateArgument(arg_id=id,
                                      descriptive_text=obj.descriptive_text(),
                                      privacy_cost=privacy)
            new_arg.set_verifier(obj.verifier_function)
            self.add_argument(new_arg)

        for e in g.edges():
            source_id = g.vp.id[e.source()]
            target_id = g.vp.id[e.target()]
            self.add_attack(source_id, target_id)

    # def make_largest_component(self):
    #     g = self.to_graph_tool()
    #     comp = gt.label_largest_component(g, directed=False)
    #     g = gt.GraphView(g, vfilt=comp)
    #     self.from_graph_tool()

    # def make_spanning_graph(self):
    #     spanning_graph = None
    #     while True:
    #         self.make_largest_component()
    #         g = self.to_graph_tool()
    #         motion_found = gt.find_vertex(g, g.vp.id, 0)
    #         if motion_found:
    #             spanning_graph = gt.random_spanning_tree(g, root=motion_found[0])
    #             break
    #         print("Failed to find root!")
    #     g = gt.GraphView(g, efilt=spanning_graph)
    #     self.from_graph_tool(g)

    # def stats(self):
    #     g = self.to_graph_tool()
    #     dist, ends = gt.pseudo_diameter(g)
    #     print("Diameter: {}".format(dist))
    #     num_v = len(g.get_vertices())
    #     num_e = len(g.get_edges())
    #     print("Edges per vertex: {}".format(num_e/num_v))


    def circuits(self):
        g = self.to_graph_tool()
        self.from_graph_tool(g)

