import unittest
from functools import reduce
from itertools import product

from lovpy.graphs.dynamic_temporal_graph import DynamicGraph, EvaluatedDynamicGraph
from lovpy.graphs.timed_property_graph import TimedPropertyGraph, PredicateGraph, PredicateNode
from lovpy.graphs.timestamps import Timestamp, RelativeTimestamp


class TestDynamicGraph(unittest.TestCase):

    def test_evaluate(self) -> None:
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
        for g in evaluated_cases:
            self.assertIsInstance(g, EvaluatedDynamicGraph)

    def test_evaluate_with_multiple_evaluations(self) -> None:
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
        for g in evaluated_cases:
            self.assertIsInstance(g, EvaluatedDynamicGraph)

    def test_evaluate_with_multiple_evaluations_of_multiple_vars(self) -> None:
        graph = PredicateGraph("call", "acquire_$a$").logical_and(
            PredicateGraph("locked_$b$").logical_not()
        ).set_timestamp(Timestamp(1))
        vars_evaluation = {"a": [1, 6, 9], "b": [2, 3, 4]}

        dynamic = DynamicGraph.to_dynamic(graph)
        evaluated_cases = list(dynamic.evaluate(locs=vars_evaluation))

        self.assertEqual(len(evaluated_cases),
                         reduce((lambda x, y: x*y), [len(x) for x in vars_evaluation.values()]))
        for g in evaluated_cases:
            self.assertIsInstance(g, EvaluatedDynamicGraph)
        for a, b in product(vars_evaluation["a"], vars_evaluation["b"]):
            eval_graph = PredicateGraph("call", f"acquire_{a}").logical_and(
                PredicateGraph(f"locked_{b}").logical_not()
            ).set_timestamp(Timestamp(1))
            self.assertIn(eval_graph, evaluated_cases)

    def test_evaluate_with_lib_call(self) -> None:
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
        for g in evaluated_cases:
            self.assertIsInstance(g, EvaluatedDynamicGraph)

    def test_evaluate_with_dynamic_arguments(self) -> None:
        graph, dyn_var = TestDynamicGraph.create_dynamic_property_rule()

        dynamic = DynamicGraph.to_dynamic(graph)
        evaluated_cases = list(dynamic.evaluate(locs={dyn_var: ["case_a", "case_b"]}))

        self.assertEqual(len(evaluated_cases), 2)
        for g in evaluated_cases:
            self.assertIsInstance(g, EvaluatedDynamicGraph)
        self.assertTrue(("a_case_a" in evaluated_cases[0].get_leaves()) !=
                        ("a_case_a" in evaluated_cases[1].get_leaves()))
        self.assertTrue(("a_case_b" in evaluated_cases[0].get_leaves()) !=
                        ("a_case_b" in evaluated_cases[1].get_leaves()))

    def test_to_dynamic(self) -> None:
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

        self.assertIsInstance(dynamic, DynamicGraph)
        self.assertEqual(dynamic.dynamic_mappings, {predicate_node: ["$a$"]})

    def test_to_dynamic_with_dynamic_arguments(self) -> None:
        graph, dyn_var = TestDynamicGraph.create_dynamic_property_rule()

        dynamic = DynamicGraph.to_dynamic(graph)

        self.assertEqual(len(dynamic.dynamic_mappings), 1)
        self.assertEqual(dynamic.dynamic_mappings, {"a_$var$": ["$var$"]})

    @staticmethod
    def create_dynamic_property_rule() -> tuple[TimedPropertyGraph, str]:
        rule = PredicateGraph("A", "a_$var$", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a_$var$").set_timestamp(RelativeTimestamp(-1)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(-1))
            ))
        return rule, "var"


class TestEvaluatedDynamicGraph(unittest.TestCase):

    def test_init(self) -> None:
        graph = EvaluatedDynamicGraph(dynamic_graph=None)
        self.assertIsInstance(graph, EvaluatedDynamicGraph)
        self.assertIsInstance(graph, TimedPropertyGraph)
        self.assertIsNone(graph.dynamic_graph)

    def test_convert_temporal_graph(self) -> None:
        temporal_graph = PredicateGraph("A", "a_$var$", "b")
        dynamic_graph = DynamicGraph.to_dynamic(temporal_graph)
        evaluated = EvaluatedDynamicGraph.convert_temporal_graph(temporal_graph, dynamic_graph)

        self.assertIsInstance(evaluated, EvaluatedDynamicGraph)
        self.assertIsInstance(evaluated, TimedPropertyGraph)
        self.assertEqual(evaluated.dynamic_graph, dynamic_graph)
