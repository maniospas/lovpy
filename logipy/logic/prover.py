import random
# import os

from logipy.logic.timestamps import RelativeTimestamp
from logipy.monitor.time_source import get_zero_locked_timesource


MAX_PROOF_PATH = 2  # Max number of theorems to be applied in order to prove a property.


class PropertyNotHoldsException(Exception):
    def __init__(self, property_text):
        message = "A property found not to hold:\n\t"
        message += property_text
        super().__init__(message)


def prove_set_of_properties(property_graphs, execution_graph):
    """A very simple and somewhat silly prover."""
    # Don't modify the original properties.
    property_graphs = [p.get_copy() for p in property_graphs]

    execution_graph.visualize("Execution Graph")

    theorems, properties_to_prove = split_into_theorems_and_properties_to_prove(property_graphs)

    for p in negate_conclusion_part_of_properties(properties_to_prove):
        execution_graph = execution_graph.get_copy()  # Modify a separate graph for each property.

        theorems_applied = []
        while len(theorems_applied) < MAX_PROOF_PATH:
            possible_theorems = find_possible_theorem_applications(execution_graph, theorems)
            if not possible_theorems:
                break

            next_theorem = select_next_theorem_application(execution_graph, possible_theorems,
                                                           p, theorems_applied)
            if not next_theorem:
                break
            next_theorem.implication_graph.visualize("Next theorem to apply.")
            apply_theorem(execution_graph, next_theorem)
            execution_graph.visualize("New execution graph.")
            theorems_applied.append(next_theorem)

        if execution_graph.contains_property_graph(p):
            execution_graph.visualize("Execution Graph where property not holds")
            p.visualize("Property that not holds")
            raise PropertyNotHoldsException(p.get_property_textual_representation())

    # # Try to apply all theorems, until no more theorem can be applied.
    # more_to_be_applied = True
    # properties_applied = 0
    # while more_to_be_applied and properties_applied < 2:
    #     # TODO: Do it only one time
    #     # TODO: Sort always proved properties based on the complexity of p.
    #     more_to_be_applied = False
    #
    #     for p in theorems:
    #         assumption, conclusion = p.get_top_level_implication_subgraphs()
    #         if execution_graph.contains_property_graph(assumption):
    #             execution_graph.replace_subgraph(assumption, conclusion)
    #             more_to_be_applied = True
    #             properties_applied += 1
    #             # p.visualize("Property Applied")
    #             # execution_graph.visualize("New Execution Graph")
    #
    # # Check that its not possible to prove the negation of all the rest properties.
    # for p in negate_conclusion_part_of_properties(properties_to_prove):
    #     if execution_graph.contains_property_graph(p):
    #         execution_graph.visualize("Execution Graph where property not holds")
    #         p.visualize("Property that not holds.")
    #         raise PropertyNotHoldsException(p.get_property_textual_representation())

    # # Visualization implementation.
    # base_path = "./test_runs/{}/".format(prove_set_of_properties.exported_counter)
    # property_id = 0
    # for p in properties_graphs:
    #     dir_path = base_path + "properties/"
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)
    #     p.export_to_graphml_file(dir_path+str(property_id)+".graphml")
    #     property_id += 1
    # execution_graph.export_to_graphml_file(base_path+"execution_graph.graphml")
    # prove_set_of_properties.exported_counter += 1


def negate_conclusion_part_of_properties(properties):
    """Returns a copy of given sequence of properties with a negated conclusion part."""
    negated_properties = []

    for p in properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        assumption = assumption.get_copy()
        conclusion = conclusion.get_copy()
        conclusion.logical_not()
        assumption.logical_and(conclusion)
        negated_properties.append(assumption)

    return negated_properties


def get_all_possible_modus_ponens(graph, properties):
    # TODO: Implemented in TimedPropertyGraph. Remove it from here.
    possible_modus_ponens = {}

    for p in properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        matching_cases, _, cases_timestamps, _ = graph.find_equivalent_subgraphs(assumption)
        if matching_cases:
            possible_modus_ponens[p] = matching_cases

    return possible_modus_ponens


def find_possible_theorem_applications(graph, theorems):
    possible_theorem_applications = []
    for theorem in theorems:
        possible_theorem_applications.extend(graph.find_all_possible_modus_ponens(theorem))
    return possible_theorem_applications


def select_next_theorem_application(graph, theorem_applications, goal, previous_applications):
    # TODO: Implement a better selector than the random one.
    # return random.choice(theorem_applications)
    used_base_theorems = {t.implication_graph for t in previous_applications}
    unused_base_applications = [t for t in theorem_applications
                                if t.implication_graph not in used_base_theorems]
    if unused_base_applications:
        return theorem_applications[0]
    else:
        return None


def apply_theorem(graph, theorem_application):
    return graph.apply_modus_ponens(theorem_application)


def split_into_theorems_and_properties_to_prove(properties):
    theorems = []
    properties_to_prove = []

    # All properties whose conclusion refers to a present moment are considered theorems.
    for p in properties:
        assumption, conclusion = p.get_top_level_implication_subgraphs()

        t = RelativeTimestamp(0)
        t.set_time_source(get_zero_locked_timesource())
        if conclusion.is_uniform_timestamped(timestamp=t):
            theorems.append(p)
        else:
            properties_to_prove.append(p)

    # In theorems, also add the parts of complex properties in which conclusion refers to the
    # same time moment as the assumption.
    for p in properties_to_prove:
        assumption, conclusion = p.get_top_level_implication_subgraphs()
        conclusion_present_part = conclusion.get_present_time_subgraph()
        if conclusion_present_part:
            theorem = assumption.get_copy()
            theorem.logical_implication(conclusion_present_part)
            theorems.append(theorem)
            p.remove_subgraph(conclusion_present_part)

    return theorems, properties_to_prove


# prove_set_of_properties.exported_counter = 0
