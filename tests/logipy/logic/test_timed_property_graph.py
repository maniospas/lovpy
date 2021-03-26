import unittest

from logipy.logic.monitored_predicate import *
from logipy.logic.timed_property_graph import TimedPropertyGraph
from logipy.logic.timed_property_graph import PredicateNode
from logipy.logic.timed_property_graph import PredicateGraph
from logipy.monitor.time_source import get_zero_locked_timesource, TimeSource


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

    def test_is_uniform_timestamped_with_specific_relative_timestamps(self):
        var = MonitoredVariable("VAR")

        locked_pred = PredicateGraph("locked", var)
        locked_pred.set_timestamp(LesserThanRelativeTimestamp(-1))
        locked_pred.set_time_source(get_zero_locked_timesource())

        not_locked_pred = locked_pred.get_copy()
        not_locked_pred.set_timestamp(RelativeTimestamp(0))
        not_locked_pred.logical_not()
        not_locked_pred.set_time_source(get_zero_locked_timesource())

        timestamp2 = RelativeTimestamp(0)
        timestamp2.set_time_source(get_zero_locked_timesource())
        self.assertTrue(not_locked_pred.is_uniform_timestamped(timestamp2))

        timestamp3 = LesserThanRelativeTimestamp(-1)
        timestamp3.set_time_source(get_zero_locked_timesource())
        self.assertFalse(not_locked_pred.is_uniform_timestamped(timestamp3))

    def test_is_uniform_timestamped_with_relative_timestamps_different_time_sources(self):
        var = MonitoredVariable("VAR")

        locked_pred = PredicateGraph("locked", var)
        locked_pred.set_timestamp(LesserThanRelativeTimestamp(-1))
        locked_pred.set_time_source(get_zero_locked_timesource())

        not_locked_pred = locked_pred.get_copy()
        not_locked_pred.set_timestamp(RelativeTimestamp(0))
        not_locked_pred.logical_not()
        not_locked_pred.set_time_source(get_zero_locked_timesource())

        timestamp2 = RelativeTimestamp(0)
        new_timesource = TimeSource()
        new_timesource.stamp_and_increment()
        new_timesource.stamp_and_increment()
        timestamp2.set_time_source(new_timesource)
        self.assertTrue(not_locked_pred.is_uniform_timestamped(timestamp2))

        timestamp3 = LesserThanRelativeTimestamp(-1)
        timestamp3.set_time_source(get_zero_locked_timesource())
        self.assertFalse(not_locked_pred.is_uniform_timestamped(timestamp3))

    def test_is_uniform_timestamped_with_specific_relative_timestamps_without_time_sources(self):
        var = MonitoredVariable("VAR")

        locked_pred = PredicateGraph("locked", var)
        locked_pred.set_timestamp(LesserThanRelativeTimestamp(-1))

        not_locked_pred = locked_pred.get_copy()
        not_locked_pred.set_timestamp(RelativeTimestamp(0))
        not_locked_pred.logical_not()

        timestamp2 = RelativeTimestamp(0)
        self.assertTrue(not_locked_pred.is_uniform_timestamped(timestamp2))

        timestamp3 = LesserThanRelativeTimestamp(-1)
        self.assertFalse(not_locked_pred.is_uniform_timestamped(timestamp3))

    # def test_and_logical_operation(self):
    #     P = PredicateGraph("call")
    #     Q = PredicateGraph("returned_by")
    #     R = PredicateGraph("called_by")
    #     P.logical_and(Q, Timestamp(1))
    #     P.logical_and(R, Timestamp(2))