from copy import deepcopy
from typing import Union, Iterable, List

from lovpy.graphs.timestamps import RelativeTimestamp
from lovpy.graphs.dynamic_temporal_graph import DynamicGraph, EvaluatedDynamicGraph
from lovpy.graphs.timed_property_graph import TimedPropertyGraph
from lovpy.monitor.time_source import get_global_time_source, get_zero_locked_timesource


global_rule_sets = list()


class LogipyPropertyException(Exception):
    def __init__(self, message):
        super().__init__(message)


class RuleSet:
    def __init__(self):
        self.rules = set()                # Raw rule graphs.
        self.properties = set()           # Properties of rules to be proved.
        self.negative_properties = set()  # Properties with negated conclusion part.
        self.theorems = set()             # Theorems of rules.
        self.neg_to_pos_property_mapping = {}  # Mapping of negative properties to positives.

    def add_rule(self, rule: TimedPropertyGraph) -> None:
        """Adds given rule to the set.

        Rule is required to be frozen in order to be successfully added to the set.
        """
        rule.set_time_source(get_global_time_source())
        self.rules.add(rule)

        theorems, properties = split_into_theorems_and_properties_to_prove([deepcopy(rule)])
        for t in theorems:
            self._add_theorem(t)
        for p in properties:
            self._add_property(p)

    def get_evaluated_theorems(self, globs: dict = None,
                               locs: dict = None) -> list[Union[TimedPropertyGraph,
                                                                EvaluatedDynamicGraph]]:
        """Computes all evaluations of the theorems in the set.

        :param globs: A mapping of available global variables during evaluation.
        :param locs: A mapping of available local variables during evaluation.
        :return: A list containing all
        """
        return RuleSet._evaluate_dynamic_graphs(self.theorems, globs, locs)

    def get_evaluated_properties(self, globs: dict = None, locs: dict = None,
                                 negatives: bool = False) -> list[Union[TimedPropertyGraph,
                                                                        EvaluatedDynamicGraph]]:
        """Computes all evaluations of the properties in the set.

        :param globs: A mapping of available global variables during evaluation.
        :param locs: A mapping of available local variables during evaluation.
        :param negatives: If set to True returns the evaluations of the properties
                with negated conclusion part.
        :return:
        """
        if negatives:
            return RuleSet._evaluate_dynamic_graphs(self.negative_properties, globs, locs)
        return RuleSet._evaluate_dynamic_graphs(self.properties, globs, locs)

    def _add_theorem(self, theorem: TimedPropertyGraph) -> None:
        theorem.freeze()
        dynamic = DynamicGraph.to_dynamic(theorem)
        self.theorems.add(dynamic) if dynamic else self.theorems.add(theorem)

    def _add_property(self, prop: TimedPropertyGraph) -> None:
        prop.freeze()
        dynamic = DynamicGraph.to_dynamic(prop)
        final_prop = dynamic if dynamic else prop
        self.properties.add(final_prop)

        negative = convert_implication_to_and(negate_implication_property(prop))
        negative.freeze()
        dynamic_neg = DynamicGraph.to_dynamic(prop)
        final_neg = dynamic_neg if dynamic_neg else negative
        self.negative_properties.add(final_neg)
        self.neg_to_pos_property_mapping[final_neg] = final_prop

    @staticmethod
    def _evaluate_dynamic_graphs(graphs: Iterable[Union[TimedPropertyGraph, DynamicGraph]],
                                 globs: dict,
                                 locs: dict) -> list[Union[TimedPropertyGraph,
                                                           EvaluatedDynamicGraph]]:
        """Evaluates a mixed list of dynamic and normal temporal graphs.

        :param graphs:
        :param globs:
        :param locs:

        :return:
            -evaluated: A list containing all possible evaluations of given graphs.
        """
        evaluated = []  # Evaluated temporal graphs.

        for g in graphs:
            if isinstance(g, DynamicGraph):
                evals = list(g.evaluate(globs, locs))
                evaluated.extend(evals)
            else:
                # If a graph is not dynamic, then it should be appended as is.
                evaluated.append(g)

        return evaluated


def add_global_rule_set(rule_set: RuleSet) -> None:
    """Registers a new rule set to the global ones."""
    global global_rule_sets
    global_rule_sets.append(rule_set)


def get_global_rule_sets() -> List[RuleSet]:
    """Returns all registered global rule sets."""
    return global_rule_sets


def split_into_theorems_and_properties_to_prove(properties):
    """Splits given sequence of properties into theorems and properties to prove.

    :return:
        -theorems: A sequence of theorems that can be in proving processes.
        -properties_to_prove: A sequence of properties that should be proved.
    """
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


def negate_conclusion_part_of_properties(properties):
    """Returns a copy of given sequence of properties with a negated conclusion part."""
    negated_properties = []

    for p in properties:
        negated_properties.append(convert_implication_to_and(negate_implication_property(p)))

    return negated_properties


def negate_implication_property(property_graph):
    """Returns a copy of given property with conclusion part negated."""
    assumption, conclusion = property_graph.get_top_level_implication_subgraphs()
    assumption = assumption.get_copy()
    conclusion = conclusion.get_copy()
    conclusion.logical_not()
    assumption.logical_implication(conclusion)
    return assumption


def convert_implication_to_and(property_graph):
    """Converts an implication TimedPropertyGraph to an AND form property.

    :param property_graph: An implication TimedPropertyGraph.

    :return: A new TimedPropertyGraph with top level implication operator converted to
            an AND operator.
    """
    if not property_graph.is_implication_graph():
        message = "Error in converting non-implication TimedPropertyGraph to AND form."
        raise RuntimeError(message)

    assumption, conclusion = property_graph.get_top_level_implication_subgraphs()
    assumption = assumption.get_copy()
    assumption.logical_and(conclusion)

    return assumption


#  ==== Deprecated functions. To be removed in next update. ====
def get_global_properties():
    return get_global_rule_sets()[0].rules


def get_global_theorems():
    return get_global_rule_sets()[0].theorems


def get_global_properties_to_prove():
    return get_global_rule_sets()[0].properties


def get_negative_properties_to_prove():
    return get_global_rule_sets()[0].negative_properties


def get_negative_to_positive_mapping():
    return get_global_rule_sets()[0].neg_to_pos_property_mapping


def empty_properties():
    return set()


def combine(property_set1, property_set2):
    """Extends the first set with the properties contained in second set."""
    for property in property_set2:
        property_set1.add(property)


def has_property(property_set, property):
    """"Checks whether given property set has the given property."""
    positive = True
    if property.startswith("NOT "):
        property = property[4:]
        positive = False
    if property == "TRUE":
        return True
    if property == "FALSE":
        return False
    return (property in property_set) == positive


def add_property(property_set, given_rules, properties):
    if given_rules is not None:
        for rule in given_rules.split(" AND "):
            if not has_property(property_set, rule):
                return

    for property in properties.split(" AND "):
        if property.startswith("SHOULD "):
            # For SHOULD it's enough to check that property already belongs to given set.
            if not has_property(property_set, property[len("SHOULD "):]):
                raise LogipyPropertyException(property)
        elif property.startswith("NOT "):
            # Adding a property preceded by NOT means removing it from the set.
            if property[4:] in property_set:
                property_set.remove(property[4:])
        elif property.startswith("ERROR"):
            raise LogipyPropertyException(property[5:])
        elif property.startswith("PRINT"):
            print(property[5:])
        else:
            property_set.add(property)
