from unittest import TestCase

from lovpy.logic.properties import RuleSet
from lovpy.graphs.timed_property_graph import TimedPropertyGraph, PredicateGraph
from lovpy.graphs.dynamic_temporal_graph import DynamicGraph, EvaluatedDynamicGraph
from lovpy.graphs.timestamps import RelativeTimestamp


class TestRuleSet(TestCase):

    def setUp(self) -> None:
        self.property_rule = TestRuleSet.create_property_rule()
        self.property_rule.freeze()
        self.theorem_rule = TestRuleSet.create_theorem_rule()
        self.theorem_rule.freeze()

    def test_add_rule_with_property(self) -> None:
        rules = RuleSet()
        rules.add_rule(self.property_rule)

        self.assertEqual(len(rules.rules), 1)
        self.assertIn(self.property_rule, rules.rules)
        self.assertEqual(len(rules.properties), 1)
        self.assertEqual(len(rules.negative_properties), 1)
        self.assertEqual(len(rules.theorems), 0)
        self.assertEqual(len(rules.neg_to_pos_property_mapping), 1)
        self.assertEqual(rules.neg_to_pos_property_mapping[rules.negative_properties.pop()],
                         rules.properties.pop())

    def test_add_rule_with_theorem(self) -> None:
        rules = RuleSet()
        rules.add_rule(self.theorem_rule)

        self.assertEqual(len(rules.rules), 1)
        self.assertIn(self.theorem_rule, rules.rules)
        self.assertEqual(len(rules.properties), 0)
        self.assertEqual(len(rules.negative_properties), 0)
        self.assertEqual(len(rules.theorems), 1)
        self.assertEqual(len(rules.neg_to_pos_property_mapping), 0)

    def test_add_rule_with_complex_rule(self) -> None:
        rule = TestRuleSet.create_rule_with_theorem_and_property()
        rule.freeze()

        rules = RuleSet()
        rules.add_rule(rule)

        self.assertEqual(len(rules.rules), 1)
        self.assertIn(rule, rules.rules)
        self.assertEqual(len(rules.properties), 1)
        self.assertEqual(len(rules.negative_properties), 1)
        self.assertEqual(len(rules.theorems), 1)
        self.assertEqual(len(rules.neg_to_pos_property_mapping), 1)
        self.assertEqual(rules.neg_to_pos_property_mapping[rules.negative_properties.pop()],
                         rules.properties.pop())

    def test_add_rule_with_dynamic_complex_rule(self) -> None:
        rule, _ = TestRuleSet.create_dynamic_complex_rule()
        rule.freeze()

        rules = RuleSet()
        rules.add_rule(rule)

        self.assertEqual(len(rules.rules), 1)
        self.assertIn(rule, rules.rules)
        self.assertEqual(len(rules.properties), 1)
        self.assertEqual(len(rules.negative_properties), 1)
        self.assertEqual(len(rules.theorems), 1)
        self.assertEqual(len(rules.neg_to_pos_property_mapping), 1)
        self.assertEqual(rules.neg_to_pos_property_mapping[rules.negative_properties.pop()],
                         rules.properties.pop())
        for t in rules.theorems:
            self.assertIsInstance(t, DynamicGraph)

    def test_get_evaluated_theorems_with_static_theorems(self) -> None:
        theorem1 = TestRuleSet.create_theorem_rule().freeze()
        theorem2 = TestRuleSet.create_theorem_rule().freeze()

        rules = RuleSet()
        rules.add_rule(theorem1)
        rules.add_rule(theorem2)

        eval_theorems = rules.get_evaluated_theorems()

        self.assertEqual(len(eval_theorems), 2)
        self.assertIn(theorem1, eval_theorems)
        self.assertIn(theorem2, eval_theorems)

    def test_get_evaluated_theorems_with_dynamic_theorems(self) -> None:
        theorem1, var1 = TestRuleSet.create_dynamic_theorem_rule()
        theorem2, var2 = TestRuleSet.create_dynamic_theorem_rule()
        theorem1.freeze()
        theorem2.freeze()

        rules = RuleSet()
        rules.add_rule(theorem1)
        rules.add_rule(theorem2)

        replacement = "new_var"
        locs = {var1: replacement}  # var1 == var2

        eval_theorems = rules.get_evaluated_theorems(locs=locs)

        self.assertEqual(len(eval_theorems), 2)
        self.assertNotIn(theorem1, eval_theorems)
        self.assertNotIn(theorem2, eval_theorems)
        for t in eval_theorems:
            self.assertIsInstance(t, EvaluatedDynamicGraph)
            self.assertIn(replacement, str(t.get_leaves()))

    def test_get_evaluated_theorems_with_mixed_theorems(self) -> None:
        theorem1 = TestRuleSet.create_theorem_rule()
        theorem2, var2 = TestRuleSet.create_dynamic_theorem_rule()
        theorem1.freeze()
        theorem2.freeze()

        rules = RuleSet()
        rules.add_rule(theorem1)
        rules.add_rule(theorem2)

        replacement = "new_var"
        locs = {var2: replacement}  # var1 == var2

        eval_theorems = rules.get_evaluated_theorems(locs=locs)

        self.assertEqual(len(eval_theorems), 2)
        self.assertIn(theorem1, eval_theorems)
        self.assertNotIn(theorem2, eval_theorems)
        self.assertTrue(isinstance(eval_theorems[0], EvaluatedDynamicGraph) or
                        isinstance(eval_theorems[1], EvaluatedDynamicGraph))
        for t in eval_theorems:
            if isinstance(t, EvaluatedDynamicGraph):
                self.assertIn(replacement, str(t.get_leaves()))

    def test_get_evaluated_properties_with_static_properties(self) -> None:
        property1 = TestRuleSet.create_property_rule().freeze()
        property2 = TestRuleSet.create_property_rule().freeze()

        rules = RuleSet()
        rules.add_rule(property1)
        rules.add_rule(property2)

        eval_properties = rules.get_evaluated_properties()

        self.assertEqual(len(eval_properties), 2)
        self.assertIn(property1, eval_properties)
        self.assertIn(property2, eval_properties)

    def test_get_evaluated_properties_with_dynamic_properties(self) -> None:
        property1, var1 = TestRuleSet.create_dynamic_property_rule()
        property2, var2 = TestRuleSet.create_dynamic_property_rule()
        property1.freeze()
        property2.freeze()

        rules = RuleSet()
        rules.add_rule(property1)
        rules.add_rule(property2)

        replacement = "new_var"
        locs = {var1: replacement}  # var1 == var2

        eval_properties = rules.get_evaluated_properties(locs=locs)

        self.assertEqual(len(eval_properties), 2)
        self.assertNotIn(property1, eval_properties)
        self.assertNotIn(property2, eval_properties)
        for p in eval_properties:
            self.assertIsInstance(p, EvaluatedDynamicGraph)
            self.assertIn(replacement, str(p.get_leaves()))

    def test_get_evaluated_properties_with_mixed_properties(self) -> None:
        property1 = TestRuleSet.create_property_rule()
        property2, var2 = TestRuleSet.create_dynamic_property_rule()
        rule3, var3 = TestRuleSet.create_dynamic_complex_rule()
        property1.freeze()
        property2.freeze()
        rule3.freeze()

        rules = RuleSet()
        rules.add_rule(property1)
        rules.add_rule(property2)
        rules.add_rule(rule3)

        replacement = "new_var"
        locs = {var2: replacement}  # var2 == var3

        eval_properties = rules.get_evaluated_properties(locs=locs)

        self.assertEqual(len(eval_properties), 3)
        self.assertIn(property1, eval_properties)
        self.assertNotIn(property2, eval_properties)
        evaluated_dynamic_properties = 0
        for p in eval_properties:
            if isinstance(p, EvaluatedDynamicGraph):
                evaluated_dynamic_properties += 1
        self.assertEqual(evaluated_dynamic_properties, 2)

    @staticmethod
    def create_property_rule() -> TimedPropertyGraph:
        rule = PredicateGraph("A", "a", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a").set_timestamp(RelativeTimestamp(-1)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(-1))
            ))
        return rule

    @staticmethod
    def create_theorem_rule() -> TimedPropertyGraph:
        rule = PredicateGraph("A", "a", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a").set_timestamp(RelativeTimestamp(0)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(0))
            ))
        return rule

    @staticmethod
    def create_rule_with_theorem_and_property() -> TimedPropertyGraph:
        rule = PredicateGraph("A", "a", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a").set_timestamp(RelativeTimestamp(0)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(-1))
            ))
        return rule

    @staticmethod
    def create_dynamic_property_rule() -> tuple[TimedPropertyGraph, str]:
        rule = PredicateGraph("A", "a_$var$", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a_$var$").set_timestamp(RelativeTimestamp(-1)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(-1))
            ))
        return rule, "var"

    @staticmethod
    def create_dynamic_theorem_rule() -> tuple[TimedPropertyGraph, str]:
        rule = PredicateGraph("A", "a_$var$", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a_$var$").set_timestamp(RelativeTimestamp(0)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(0))
            ))
        return rule, "var"

    @staticmethod
    def create_dynamic_complex_rule() -> tuple[TimedPropertyGraph, str]:
        rule = PredicateGraph("A", "a_$var$", "b").set_timestamp(RelativeTimestamp(0))
        rule = rule.logical_implication(
            PredicateGraph("B", "a_$var$").set_timestamp(RelativeTimestamp(0)).logical_and(
                PredicateGraph("C", "c").set_timestamp(RelativeTimestamp(-1))
            ))
        return rule, "var"
