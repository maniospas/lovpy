import unittest

from lovpy.graphs.dynamic_temporal_graph import DynamicGraph
from lovpy.graphs.timed_property_graph import PredicateGraph, PredicateNode
from lovpy.graphs.timestamps import Timestamp


class TestDynamicGraph(unittest.TestCase):

    def test_evaluate(self):
        a = PredicateGraph("call", "acquire")
        b = PredicateGraph("locked_$a$")
        b.logical_not()
        a.logical_and(b)
        a.set_timestamp(Timestamp(1))

        a_eval = PredicateGraph("call", "acquire")
        b_eval = PredicateGraph("locked_5")
        b_eval.logical_not()
        a_eval.logical_and(b_eval)
        a_eval.set_timestamp(Timestamp(1))

        dynamic = DynamicGraph.to_dynamic(a)
        evaluated_cases = list(dynamic.evaluate(locs={"a": 5}))

        self.assertEqual(len(evaluated_cases), 1)
        self.assertEqual(evaluated_cases[0], a_eval)

    def test_evaluate_with_multiple_evaluations(self):
        a = PredicateGraph("call", "acquire")
        b = PredicateGraph("locked_$a$")
        b.logical_not()
        a.logical_and(b)
        a.set_timestamp(Timestamp(1))

        a_eval = PredicateGraph("call", "acquire")
        b_eval = PredicateGraph("locked_1")
        b_eval.logical_not()
        a_eval.logical_and(b_eval)
        a_eval.set_timestamp(Timestamp(1))

        dynamic = DynamicGraph.to_dynamic(a)
        evaluated_cases = list(dynamic.evaluate(locs={"a": [1, 6, 9]}))

        self.assertEqual(len(evaluated_cases), 3)
        self.assertEqual(evaluated_cases[0], a_eval)

    def test_evaluate_with_lib_call(self):
        import threading

        a = PredicateGraph("call", "release")
        b = PredicateGraph("locked_$threading.get_ident()$")
        b.logical_not()
        a.logical_implication(b)
        a.set_timestamp(Timestamp(1))

        a_eval = PredicateGraph("call", "release")
        b_eval = PredicateGraph(f"locked_{threading.get_ident()}")
        b_eval.logical_not()
        a_eval.logical_implication(b_eval)
        a_eval.set_timestamp(Timestamp(1))

        dynamic = DynamicGraph.to_dynamic(a)
        evaluated_cases = list(dynamic.evaluate(locs=locals()))

        self.assertEqual(len(evaluated_cases), 1)
        self.assertEqual(evaluated_cases[0], a_eval)

    def test_to_dynamic(self):
        a = PredicateGraph("call", "acquire")
        b = PredicateGraph("locked_$a$")
        b.logical_not()
        a.logical_and(b)

        dynamic = DynamicGraph.to_dynamic(a)

        predicate_node = None
        for n in a.graph.nodes:
            if isinstance(n, PredicateNode) and n.predicate == "locked_$a$":
                predicate_node = n
                break

        self.assertEqual(dynamic.dynamic_mappings, {predicate_node: ["$a$"]})
