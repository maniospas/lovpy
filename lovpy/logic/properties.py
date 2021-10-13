from lovpy.graphs.timestamps import RelativeTimestamp
from lovpy.monitor.time_source import get_global_time_source, get_zero_locked_timesource

global_properties = list()  # Storage for graph properties that apply everywhere.
global_theorems = set()
global_properties_to_prove = set()
global_negative_properties_to_prove = set()
negative_to_positive_mapping = dict()


class LogipyPropertyException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_global_properties():
    return global_properties


def get_global_theorems():
    return global_theorems


def get_global_properties_to_prove():
    return global_properties_to_prove


def get_negative_properties_to_prove():
    return global_negative_properties_to_prove


def get_negative_to_positive_mapping():
    return negative_to_positive_mapping


def add_global_property(property_graph):
    property_graph.set_time_source(get_global_time_source())
    global_properties.append(property_graph.get_copy())

    theorems, properties_to_prove = split_into_theorems_and_properties_to_prove([property_graph])
    for t in theorems:
        t.freeze()
    for p in properties_to_prove:
        p.freeze()
    global_theorems.update(theorems)
    global_properties_to_prove.update(properties_to_prove)

    negative_properties = negate_conclusion_part_of_properties(properties_to_prove)
    for p in negative_properties:
        p.freeze()
    global_negative_properties_to_prove.update(negative_properties)
    for pos, neg in zip(properties_to_prove, negative_properties):
        negative_to_positive_mapping[neg] = pos


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


# All code below, is deprecated.
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
