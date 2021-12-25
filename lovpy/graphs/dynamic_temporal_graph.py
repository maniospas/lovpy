from itertools import product
from copy import copy, deepcopy
import re

from .timed_property_graph import TimedPropertyGraph, PredicateNode


class DynamicGraph:
    """A dynamic graph that produces temporal graphs by dynamic code execution."""

    def __init__(self, graph: TimedPropertyGraph, mappings={}):
        self.temporal_graph = graph
        # Map between predicate nodes and list of the dynamic parts of each node.
        self.dynamic_mappings = mappings

    def evaluate(self, globs={}, locs={}):
        """Evaluates dynamic graph into each possible temporal graph.

        Each dynamic part that is evaluated into a list produces multiple
        temporal graphs, one for each item of the list.

        :param globs: Dictionary of global variables for dynamic execution.
        :param locs: Dictionary of local variables for dynamic execution.

        :return: A generator of all possible temporal graphs produced after dynamic
                parts evaluation.
        """
        evaluated_mappings = self._evaluate_mappings(globs, locs)

        for case in evaluated_mappings:
            yield self._generate_graph_from_evaluation(case)

    @staticmethod
    def to_dynamic(graph: TimedPropertyGraph):
        """Converts a temporal graph to a dynamic graph.

        :param graph: Temporal graph to be converted to a dynamic one.

        :return: A dynamic graph if given temporal graph contained dynamics predicates,
                else None.
        """
        mappings = dict()

        for n in graph.graph.nodes():
            if isinstance(n, PredicateNode):
                # Extract the dynamic parts of each predicate.
                dynamic_parts = re.findall(r"\$[^$]*\$", n.predicate)
                if dynamic_parts:
                    mappings[n] = dynamic_parts

        return DynamicGraph(graph, mappings) if mappings else None

    def _evaluate_mappings(self, globs, locs):
        """Computes all possible evaluated instances of mappings."""
        evaluated_cases = []  # [[(n, dyn_text, eval_tex), ...], ...]

        for n, dyn_parts in self.dynamic_mappings.items():
            for d in dyn_parts:
                part_evaluations = eval(str(d).strip("$"), globs, locs)
                if not isinstance(part_evaluations, list):
                    part_evaluations = [part_evaluations]

                if evaluated_cases:  # Copy old partial cases and expand them.
                    old_cases = evaluated_cases
                    evaluated_cases = []
                    for case, new in product(old_cases, part_evaluations):
                        case = copy(case)
                        case.append(tuple([n, d, new]))
                        evaluated_cases.append(case)
                else:  # Create the first partial cases.
                    for new in part_evaluations:
                        evaluated_cases.append([tuple([n, d, new])])

        return evaluated_cases

    def _generate_graph_from_evaluation(self, evaluated_mapping):
        """Generates a temporal graph according to given evaluation of dynamic nodes.

        :param evaluated_mapping: A list of tuples in the form of (node, dynamic_text,
                evaluated_text).

        :return: An evaluated `TimedPropertyGraph` object.
        """
        evaluated_graph = deepcopy(self.temporal_graph)
        replace_mappings = {}

        for n, dynamic_part, evaluated in evaluated_mapping:
            new_node = deepcopy(n)
            new_node.predicate = n.predicate.replace(dynamic_part, str(evaluated))
            replace_mappings[n] = new_node

        evaluated_graph.replace_nodes(replace_mappings)

        return evaluated_graph
