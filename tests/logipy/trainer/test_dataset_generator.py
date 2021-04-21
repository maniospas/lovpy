import unittest

from logipy.trainer.dataset_generator import *
from logipy.logic.prover import split_into_theorems_and_properties_to_prove

from tests.logipy.importer.sample_properties import get_threading_sample_properties


class TestDatasetEntity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.threading_properties = get_threading_sample_properties()
        cls.threading_theorems, cls.threading_properties_to_prove = \
            prover.split_into_theorems_and_properties_to_prove(cls.threading_properties)
        cls.threading_properties_to_prove = \
            prover.negate_conclusion_part_of_properties(cls.threading_properties_to_prove)

    def test_add_property_to_prove_on_empty_graph(self):
        entity = DatasetEntity()
        self.assertFalse(entity.contains_property_to_prove())

        entity.add_property_to_prove(self.threading_properties_to_prove[0])
        self.assertTrue(entity.contains_property_to_prove())

        entity.current_graph.visualize("Current Graph after adding property on empty graph.")
        return entity

    def test_expand_with_theorem(self):
        entity = self.test_add_property_to_prove_on_empty_graph()

        theorem_applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        self.assertGreater(len(theorem_applications), 0)
        entity.current_graph.visualize("Current graph before theorem application.")
        theorem_to_apply = theorem_applications[0]
        theorem_to_apply.implication_graph.visualize()

        entity.expand_with_theorem(theorem_applications[0])
        entity.current_graph.visualize("Current graph after expanding with theorem.")

        return entity

    def test_add_properties_of_theorem(self):
        entity = self.test_add_property_to_prove_on_empty_graph()

        entity.add_properties_of_theorem(self.threading_theorems[0])
        entity.current_graph._keep_most_recent_parallel_paths_out_of_inverted_ones()
        self.threading_theorems[0].visualize("Theorem whose properties will be added.")
        entity.current_graph.visualize("Current graph after adding theorem properties.")

    def test_with_5_deep_theorem(self):
        entity = self.test_expand_with_theorem()

        entity.add_properties_of_theorem(self.threading_theorems[0])
        self.threading_theorems[0].visualize("Theorem whose properties will be added.")
        entity.current_graph.visualize("Current graph after adding theorem properties.")

        entity.add_properties_of_theorem(self.threading_theorems[0])
        self.threading_theorems[0].visualize("Theorem whose properties will add.")
        entity.current_graph.visualize("Current graph after adding theorem properties.")

        theorem_applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        self.assertGreater(len(theorem_applications), 0)
        entity.current_graph.visualize("Current graph before theorem application.")
        theorem_to_apply = theorem_applications[0]
        theorem_to_apply.implication_graph.visualize()

    # def test_not_provable_graph(self):
    #     entity = DatasetEntity()
    #     ass, con = self.threading_properties[0].get_top_level_implication_subgraphs()
    #     negated = ass.get_copy()
    #     negated.logical_and(con)
    #     entity.add_property_to_prove(ass)
    #     entity.current_graph.visualize()

    def test_with_multiple_applications_of_call_acquire(self):
        # Initial property addition.
        entity = self.test_add_property_to_prove_on_empty_graph()

        self.threading_properties_to_prove[0]

        # Add properties of a theorem that negate final conclusion.
        entity.add_properties_of_theorem(self.threading_theorems[0])
        entity.current_graph.visualize("Adding release property.")
        self.assertFalse(bool(entity.next_theorem))

        # Expand by reversely applying theorem for release() predicate to show up.
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

    def test_suppressed_predicates_addition(self):
        # Initial property addition.
        entity = self.test_add_property_to_prove_on_empty_graph()

        # Expand by reversely applying theorem for call(acquire).
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

        # Expand by adding a suppressed predicate.
        suppressed = entity.get_suppressed_predicates()
        entity.add_suppressed_predicate(suppressed[0])
        entity.current_graph.visualize("Added suppressed predicate.")

        # Expand by reversely applying theorem for release() predicate to show up.
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

        # Expand by reversely applying theorem for call(acquire).
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

        # Expand by adding a suppressed predicate.
        suppressed = entity.get_suppressed_predicates()
        entity.add_suppressed_predicate(suppressed[0])
        entity.current_graph.visualize("Added suppressed predicate.")

        # Expand by reversely applying theorem for release() predicate to show up.
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

        # Expand by reversely applying theorem for call(acquire).
        # TODO: Fix theorem application.
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")


class TestDatasetGenerator(unittest.TestCase):

    def test_simple_threading_dataset(self):
        generator = DatasetGenerator(get_threading_sample_properties(), 7, 5, verbose=False)
        samples = list(generator)
        # self.assertEqual(len(samples), 1)
        i = 1
        for s in samples:
            s.current_graph.visualize(f"Sample graph #{i}")
            can_prove = "Can prove property" if s.is_provable else "Impossible to prove property."
            s.goal.visualize(can_prove)
            i += 1

    def test_simple_threading_theorem(self):
        generator = DatasetGenerator(get_threading_sample_properties(), 5, 3)
