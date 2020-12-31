import os
import glob
import re

import logipy.logic.properties
from logipy.logic.monitored_predicate import MonitoredPredicate
from logipy.logic.timed_property_graph import *
from logipy.logic.timestamps import RelativeTimestamp, LesserThanRelativeTimestamp


def import_gherkin_path(root_path=""):
    """Imports the rules from all .gherkin files under root_path."""
    for path in glob.glob(root_path+"**/*.gherkin", recursive=True):
        if os.path.isfile(path):
            import_gherkin_file(path)


def import_gherkin_file(path):
    """Imports the rules of given .gherkin file."""
    if not path.endswith(".gherkin"):
        raise Exception("Can only import .gherkin files: "+path)

    lines = list()
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line and line[0] != "#":
                if line[-1] == "\n":
                    line[-1] = line[:-1]
                lines.append(line)
        file.close()

    import_gherkin_lines(lines)


def import_gherkin_lines(lines):
    """Imports all rules defined in given gherkin lines of code."""
    for rule in (" ".join(lines)).split("SCENARIO:"):
        rule = rule.strip()
        if rule:
            # print("\""+rule+"\"\n")
            graph = convert_specification_to_graph(rule)
            logipy.logic.properties.add_global_property(graph)
            # TODO: Remove debug code.
            # print("Nodes:\n")
            # for n in graph._get_graph().nodes():
            #     print(n)
            # print("Edges:\n")
            # for e in graph._get_graph().edges():
            #     print(e)


def convert_specification_to_graph(formula):
    """Converts a specification formula to a specification graph."""
    given_clause, when_clause, then_clause = get_fundamental_clauses(formula)

    when_property = convert_clause_to_graph(when_clause)
    when_property.set_timestamp(LesserThanRelativeTimestamp(-1))

    then_property = convert_clause_to_graph(then_clause)
    then_property.set_timestamp(RelativeTimestamp(0))

    final_property = when_property
    if given_clause:
        given_property = convert_clause_to_graph(given_clause)
        given_property.set_timestamp(LesserThanRelativeTimestamp(-1))
        final_property.logical_and(given_property)

    final_property.logical_implication(then_property)

    return final_property


def get_fundamental_clauses(formula):
    """Extracts the fundamental step subformulas out of a specification formula."""
    regex = re.compile(
        r"^(GIVEN (?P<given_clause>.*) )?(WHEN (?P<when_clause>.*) )(THEN (?P<then_clause>.*))")

    matches = regex.match(formula).groupdict()
    given_clause = matches['given_clause']
    when_clause = matches['when_clause']
    then_clause = matches['then_clause']

    if when_clause is None or then_clause is None:
        exc_text = "WHEN and THEN clauses are required in specifications syntax.\n"
        exc_text += "The following specifications is invalid:\n"
        exc_text += formula
        raise Exception()

    return given_clause, when_clause, then_clause


def convert_clause_to_graph(clause):
    """Converts a fundamental step clause, to property graph.

    A fundamental step clause, is the text tha follows GIVEN, WHEN, THEN steps.
    """
    subclauses = clause.split(' AND ')
    clause_graph = TimedPropertyGraph()

    for subclause in subclauses:
        subclause_graph = None

        print("\tConverting: \'"+subclause+"\'\n")  # TODO: Remove debug comment.

        # Initially, remove any preceding negation and parse the positive predicate.
        is_negated = subclause.startswith('NOT ')
        if is_negated:
            subclause = subclause.lstrip('NOT ')

        # Check if subclause is a defined function.
        monitored_predicate = MonitoredPredicate.find_text_matching_monitored_predicate(subclause)
        print("\t\tfound monitored predicate:"+str(monitored_predicate)+"\n")  # TODO: Remove debug comment.
        if monitored_predicate is None:
            subclause_graph = convert_predicate_to_graph(subclause)
        else:
            subclause_graph = monitored_predicate.convert_to_graph()

        # If original subclause was negated, negate the total graph of the subclause.
        if is_negated:
            subclause_graph.logical_not()
            print("\t\tapplied NOT")  # TODO: Remove debug comment.

        clause_graph.logical_and(subclause_graph)

    return clause_graph


def convert_predicate_to_graph(predicate):
    """Converts a predicate to a graph representation

    Predicates are allowed to contain SHOULD modifier.
    """
    if predicate.startswith("SHOULD "):
        # SHOULD modifier means that a predicate should already have been TRUE.
        predicate = predicate.lstrip("SHOULD ")
        timestamp = LesserThanRelativeTimestamp(-1)
    else:
        # Without SHOULD modifier, a predicate becomes TRUE at current time step.
        timestamp = RelativeTimestamp(0)

    predicate_graph = PredicateGraph(predicate, MonitoredVariable("VAR"))

    return predicate_graph
