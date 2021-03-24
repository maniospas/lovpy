import unittest

from logipy.importer.gherkin_importer import *
from logipy.logic.monitored_predicate import *


class TestConvertSpecificationToGraph(unittest.TestCase):

    def test_conclusion_with_negated_past_and_positive_present_and_special_function(self):
        spec = "WHEN call acquire "
        spec += "THEN SHOULD NOT locked AND locked AND PRINT locked [VAR.logipy_value()]"

        property_graph = convert_specification_to_graph(spec)

        var = MonitoredVariable("VAR")

        # Assumption.
        call_pred_graph = Call("acquire").convert_to_graph()
        call_pred_graph.set_timestamp(RelativeTimestamp(0))

        # Conclusion.
        should_not_locked_pred = PredicateGraph("locked", var)
        should_not_locked_pred.logical_not()
        should_not_locked_pred.set_timestamp(LesserThanRelativeTimestamp(-1))

        locked_pred = PredicateGraph("locked", var)
        locked_pred.set_timestamp(RelativeTimestamp(0))

        # print_pred = PredicateGraph("PRINT locked [VAR.logipy_value()]", var)
        # print_pred.set_timestamp(RelativeTimestamp(0))

        conclusion_graph = should_not_locked_pred
        conclusion_graph.logical_and(locked_pred)
        # conclusion_graph.logical_and(print_pred)

        final_custom_graph = call_pred_graph
        final_custom_graph.logical_implication(conclusion_graph)

        self.assertTrue(property_graph.contains_property_graph(final_custom_graph))
        self.assertTrue(final_custom_graph.contains_property_graph(property_graph))

    def test_property_with_given_clause(self):
        spec = "GIVEN locked "
        spec += "WHEN call release "
        spec += "THEN PRINT released by[METHOD] AND NOT locked"

        property_graph = convert_specification_to_graph(spec)

        var = MonitoredVariable("VAR")

        locked_pred = PredicateGraph("locked", var)
        locked_pred.set_timestamp(LesserThanRelativeTimestamp(-1))

        call_release_graph = Call("release").convert_to_graph()
        call_release_graph.set_timestamp(RelativeTimestamp(0))

        not_locked_pred = locked_pred.get_copy()
        not_locked_pred.set_timestamp(RelativeTimestamp(0))
        not_locked_pred.logical_not()

        final_custom_graph = call_release_graph
        final_custom_graph.logical_and(locked_pred)
        final_custom_graph.logical_implication(not_locked_pred)

        self.assertTrue(property_graph.contains_property_graph(final_custom_graph))
        self.assertTrue(final_custom_graph.contains_property_graph(property_graph))