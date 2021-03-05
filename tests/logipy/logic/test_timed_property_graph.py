import unittest

from logipy.logic.monitored_predicate import *
from logipy.logic.timed_property_graph import TimedPropertyGraph
from logipy.logic.timed_property_graph import PredicateNode
from logipy.logic.timed_property_graph import PredicateGraph


class TestTimedPropertyGraph(unittest.TestCase):

    def test_contains_property_graph_on_itself(self):
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

        print_pred = PredicateGraph("PRINT locked [VAR.logipy_value()]", var)
        print_pred.set_timestamp(RelativeTimestamp(0))

        conclusion_graph = should_not_locked_pred
        conclusion_graph.logical_and(locked_pred)
        conclusion_graph.logical_and(print_pred)

        final_custom_graph = call_pred_graph
        final_custom_graph.logical_implication(conclusion_graph)

        self.assertTrue(final_custom_graph.contains_property_graph(final_custom_graph))

    # def test_and_logical_operation(self):
    #     P = PredicateGraph("call")
    #     Q = PredicateGraph("returned_by")
    #     R = PredicateGraph("called_by")
    #     P.logical_and(Q, Timestamp(1))
    #     P.logical_and(R, Timestamp(2))