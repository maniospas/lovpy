import unittest

from logipy.logic.monitored_predicate import *
from logipy.logic.timed_property_graph import TimedPropertyGraph
from logipy.logic.timed_property_graph import PredicateNode
from logipy.logic.timed_property_graph import PredicateGraph
from logipy.monitor.time_source import get_zero_locked_timesource, TimeSource
from tests.logipy.importer.sample_properties import get_counter_sample_properties
import logipy.logic.prover as prover


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

    def test_apply_modus_ponens_when_assumption_equals_subject_graph(self):
        returned_by_range = ReturnedBy("range").convert_to_graph()
        returned_by_range.set_timestamp(Timestamp(12))

        counter_properties = get_counter_sample_properties()
        counter_theorems, _ = prover.split_into_theorems_and_properties_to_prove(counter_properties)

        modus_ponens_applications = \
            prover.find_possible_theorem_applications(returned_by_range, counter_theorems)

        exception_raised = False

        try:
            returned_by_range.apply_modus_ponens(modus_ponens_applications[0])
        except Exception:
            exception_raised = True
        if exception_raised:
            self.fail("Modus Ponen application failed.")

        # TODO: Improve this test.

    def test_get_all_paths(self):
        graph = self._generate_sample_execution_graph_1()
        all_paths = graph.get_all_paths()

        self.assertEqual(len(all_paths), 10)
        for p in all_paths:
            self.assertIsInstance(p, TimestampedPath)

    def test_find_path_timestamp(self):
        graph = self._generate_sample_execution_graph_1()
        all_paths = graph.get_all_paths()
        # TODO: Implement

    def test_remove_subgraph_with_orphan_and(self):
        graph = self._generate_sample_execution_graph_1()
        before_and_counts = 0
        for n in graph.graph.nodes:
            if isinstance(n, AndOperator):
                before_and_counts += 1
        # graph.visualize()

        subgraph = self._generate_sample_execution_graph_1_subgraph()
        # subgraph.visualize()

        graph.remove_subgraph(subgraph)
        # graph.visualize()
        after_and_counts = 0
        for n in graph.graph.nodes:
            if isinstance(n, AndOperator):
                after_and_counts += 1

        self.assertLess(after_and_counts, before_and_counts)

    def test_remove_subgraph_with_orphan_logical_operators(self):
        pass  # TODO: implement

    def test_get_basic_predicates(self):
        graph = self._generate_sample_execution_graph_1()
        basic_predicates = graph.get_basic_predicates()
        self.assertEqual(len(basic_predicates), 5)
        # for p in basic_predicates:
        #     p.visualize()

        predicates = self._generate_sample_execution_graph_1_predicates()

        for basic_predicate in basic_predicates:
            for predicate in predicates:
                matches, _, _, _ = predicate.find_equivalent_subgraphs(basic_predicate)
                if matches:
                    break
            else:
                self.fail("Original predicate did not matched by any basic predicate.")

    @staticmethod
    def _generate_sample_execution_graph_1():
        predicates = TestTimedPropertyGraph._generate_sample_execution_graph_1_predicates()
        graph = predicates.pop(0)
        for p in predicates:
            graph.logical_and(p)

        return graph

    @staticmethod
    def _generate_sample_execution_graph_1_predicates():
        pred1 = Call("acquire").convert_to_graph()
        pred1.set_timestamp(Timestamp(1))
        pred2 = Call("lock").convert_to_graph()
        pred2.set_timestamp(Timestamp(3))
        pred3 = Call("release").convert_to_graph()
        pred3.set_timestamp(Timestamp(5))
        pred4 = Call("lock").convert_to_graph()
        pred4.set_timestamp(Timestamp(11))
        pred5 = Call("release").convert_to_graph()
        pred5.set_timestamp(Timestamp(29))
        return [pred1, pred2, pred3, pred4, pred5]

    @staticmethod
    def _generate_sample_execution_graph_1_subgraph():
        pred2 = Call("lock").convert_to_graph()
        pred2.set_timestamp(Timestamp(3))
        return pred2

    # def test_and_logical_operation(self):
    #     P = PredicateGraph("call")
    #     Q = PredicateGraph("returned_by")
    #     R = PredicateGraph("called_by")
    #     P.logical_and(Q, Timestamp(1))
    #     P.logical_and(R, Timestamp(2))


def retrieve_path_by_text(paths, nodes_labels):
    paths = paths.copy()

    # Remove paths whose size doesn't match give one.
    paths = [p for p in paths if len(p) == len(nodes_labels)]

    for i in range(nodes_labels):
        new_paths = []
        for p in paths:
            label = nodes_labels[i]
            n = p[i]
            if label == "AND" and isinstance(n, AndOperator):
                new_paths.append(p)
            elif label == "NOT" and isinstance(n, NotOperator):
                new_paths.append(p)
            elif label == "call" and isinstance(n, PredicateNode) \
                    and str(n).startswith("call"):
                new_paths.append(p)
            elif label == "called_by" and isinstance(n, PredicateNode) \
                    and str(n).startswith("called_by"):
                new_paths.append(p)
            elif label == "returned_by" and isinstance(n, PredicateNode) \
                    and str(n).startswith("returned_by"):
                new_paths.append(p)
            elif label == n:
                new_paths.append(p)
        paths = new_paths

    return paths


def convert_edges_path_to_nodes_path(edges_path):
    pass  # TODO: Implement
