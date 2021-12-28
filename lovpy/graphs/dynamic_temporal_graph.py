from itertools import product
from copy import copy, deepcopy
from typing import Generator
import re

from .timed_property_graph import TimedPropertyGraph, PredicateNode


class DynamicGraph:
    """A dynamic graph that produces temporal graphs by dynamic code execution."""

    def __init__(self, graph: TimedPropertyGraph, mappings: dict = {}):
        self.temporal_graph = graph
        # Map between predicate nodes and list of the dynamic parts of each node.
        self.dynamic_mappings = mappings

    def evaluate(self, globs: dict = {}, locs: dict = {}) -> Generator['EvaluatedDynamicGraph',
                                                                       None, None]:
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
            yield EvaluatedDynamicGraph.convert_temporal_graph(
                self._generate_graph_from_evaluation(case), self)

    @staticmethod
    def to_dynamic(graph: TimedPropertyGraph) -> 'DynamicGraph':
        """Converts a temporal graph to a dynamic graph.

        :param graph: Temporal graph to be converted to a dynamic one.

        :return: A dynamic graph if given temporal graph contained dynamics predicates,
                else None.
        """
        mappings = dict()

        for n in graph.graph.nodes():
            dynamic_parts = []

            if isinstance(n, PredicateNode):
                # Extract the dynamic parts of each predicate.
                dynamic_parts = re.findall(r"\$[^$]*\$", n.predicate)
            elif isinstance(n, str):
                dynamic_parts = re.findall(r"\$[^$]*\$", n)

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
            if isinstance(n, PredicateNode):
                new_node = deepcopy(n)
                new_node.predicate = n.predicate.replace(dynamic_part, str(evaluated))
                replace_mappings[n] = new_node
            elif isinstance(n, str):
                new_node = n.replace(dynamic_part, str(evaluated))
                replace_mappings[n] = new_node

        evaluated_graph.replace_nodes(replace_mappings)

        return evaluated_graph


class EvaluatedDynamicGraph(TimedPropertyGraph):
    """A wrapper for adding dynamic origin info to `TimedPropertyGraph` objects."""

    def __init__(self, dynamic_graph: DynamicGraph = None):
        super().__init__()
        self.dynamic_graph = dynamic_graph

    @classmethod
    def convert_temporal_graph(cls, temporal_graph: TimedPropertyGraph,
                               dynamic_graph: DynamicGraph) -> 'EvaluatedDynamicGraph':
        """Converts a temporal graph to an evaluated one.

        Conversion is performed in-place, so given `TimedPropertyGraph` instance
        will be forever converted to an `EvaluatedDynamicGraph` instance.

        :param temporal_graph: Temporal graph to be permanently converted to an
                `EvaluatedDynamicGraph`.
        :param dynamic_graph: Dynamic graph from which given temporal graph
                `originates.
        :return: A reference to the converted temporal graph.
        """
        temporal_graph.__class__ = cls
        temporal_graph.dynamic_graph = dynamic_graph
        return temporal_graph
