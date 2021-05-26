import unittest
from collections import Counter

from logipy import config
from logipy.models.dataset_generator import *
from logipy.logic.prover import split_into_theorems_and_properties_to_prove
from logipy.importer.gherkin_importer import import_gherkin_path, import_gherkin_file
from logipy.logic.properties import get_global_properties

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
        entity = DatasetEntity(self.threading_theorems)
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

    # def test_add_properties_of_theorem(self):
    #     entity = self.test_add_property_to_prove_on_empty_graph()
    #
    #     entity.add_properties_of_theorem(self.threading_theorems[0])
    #     entity.current_graph._keep_most_recent_parallel_paths_out_of_inverted_ones()
    #     self.threading_theorems[0].visualize("Theorem whose properties will be added.")
    #     entity.current_graph.visualize("Current graph after adding theorem properties.")
    #
    # def test_with_5_deep_theorem(self):
    #     entity = self.test_expand_with_theorem()
    #
    #     entity.add_properties_of_theorem(self.threading_theorems[0])
    #     self.threading_theorems[0].visualize("Theorem whose properties will be added.")
    #     entity.current_graph.visualize("Current graph after adding theorem properties.")
    #
    #     entity.add_properties_of_theorem(self.threading_theorems[0])
    #     self.threading_theorems[0].visualize("Theorem whose properties will add.")
    #     entity.current_graph.visualize("Current graph after adding theorem properties.")
    #
    #     theorem_applications = entity.get_reverse_theorem_applications(self.threading_theorems)
    #     self.assertGreater(len(theorem_applications), 0)
    #     entity.current_graph.visualize("Current graph before theorem application.")
    #     theorem_to_apply = theorem_applications[0]
    #     theorem_to_apply.implication_graph.visualize()

    # def test_not_provable_graph(self):
    #     entity = DatasetEntity()
    #     ass, con = self.threading_properties[0].get_top_level_implication_subgraphs()
    #     negated = ass.get_copy()
    #     negated.logical_and(con)
    #     entity.add_property_to_prove(ass)
    #     entity.current_graph.visualize()

    # def test_with_multiple_applications_of_call_acquire(self):
    #     # Initial property addition.
    #     entity = self.test_add_property_to_prove_on_empty_graph()
    #
    #     self.threading_properties_to_prove[0]
    #
    #     # Add properties of a theorem that negate final conclusion.
    #     entity.add_properties_of_theorem(self.threading_theorems[0])
    #     entity.current_graph.visualize("Adding release property.")
    #     self.assertFalse(bool(entity.next_theorem))
    #
    #     # Expand by reversely applying theorem for release() predicate to show up.
    #     applications = entity.get_reverse_theorem_applications(self.threading_theorems)
    #     applications[0].implication_graph.visualize("Next theorem to reversely apply.")
    #     entity.expand_with_theorem(applications[0])
    #     entity.current_graph.visualize("Current graph after reverse theorem expansion.")

    def test_suppressed_predicates_addition(self):
        # Initial property addition.
        entity = self.test_add_property_to_prove_on_empty_graph()

        # Expand by reversely applying theorem for call(acquire).
        applications = entity.get_reverse_theorem_applications(self.threading_theorems)
        applications[0].implication_graph.visualize("Next theorem to reversely apply.")
        entity.expand_with_theorem(applications[0])
        entity.current_graph.visualize("Current graph after reverse theorem expansion.")

        # Expand by adding a suppressed predicate.
        suppressed = entity.get_non_suppressible_suppressed_predicates()
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
        suppressed = entity.get_non_suppressible_suppressed_predicates()
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

    def test_threading_dataset(self):
        threading_properties = get_threading_sample_properties()
        max_depth = 10
        total_samples = 100
        negative_samples_percentage = 0.8

        generator = DatasetGenerator(
            threading_properties,
            max_depth,
            total_samples,
            random_expansion_probability=0.,
            add_new_property_probability=0.2,
            negative_samples_percentage=negative_samples_percentage,
            verbose=False)
        samples = list(generator)

        # Test number of returned samples.
        self.assertEqual(len(samples), total_samples)

        # Test actual negative samples percentage.
        negative_samples_counter = Counter([s.next_theorem_correct for s in samples])
        actual_negative_samples_ratio = float(negative_samples_counter[False]) / len(samples)
        self.assertAlmostEqual(
                negative_samples_percentage, actual_negative_samples_ratio, delta=0.3)

        # Test that every sample contains current, next and goal graphs.
        for s in samples:
            self.assertIsNotNone(s.current_graph)
            self.assertIsNotNone(s.next_theorem)
            self.assertIsNotNone(s.goal)

    def test_non_chronological_dataset(self):
        get_global_properties().clear()
        import_gherkin_file(
            str(config.LOGIPY_ROOT_PATH.parent/"examples/non_chronological_rules.gherkin"))
        properties = get_global_properties()

        max_depth = 10
        total_samples = 5
        negative_samples_percentage = 0.8

        generator = DatasetGenerator(
            properties,
            max_depth,
            total_samples,
            random_expansion_probability=0.,
            add_new_property_probability=0.2,
            negative_samples_percentage=negative_samples_percentage,
            verbose=True)
        samples = list(generator)

        # Test number of returned samples.
        self.assertEqual(len(samples), total_samples)

        # Test actual negative samples percentage.
        negative_samples_counter = Counter([s.next_theorem_correct for s in samples])
        actual_negative_samples_ratio = float(negative_samples_counter[False]) / len(samples)
        self.assertAlmostEqual(
            negative_samples_percentage, actual_negative_samples_ratio, delta=0.3)

        # Test that every sample contains current, next and goal graphs.
        for s in samples:
            self.assertIsNotNone(s.current_graph)
            self.assertIsNotNone(s.next_theorem)
            self.assertIsNotNone(s.goal)

    def test_full_examples_dataset(self):
        # Utilize logipy's import scheme to import all examples.
        print(f"Importing from {config.LOGIPY_ROOT_PATH.parent}")
        import_gherkin_path(config.LOGIPY_ROOT_PATH.parent)
        properties = get_global_properties()

        max_depth = 10
        total_samples = 100
        negative_samples_percentage = 0.8

        generator = DatasetGenerator(
            properties,
            max_depth,
            total_samples,
            random_expansion_probability=0.,
            add_new_property_probability=0.2,
            negative_samples_percentage=negative_samples_percentage,
            verbose=False)
        samples = list(generator)

        # Test number of returned samples.
        self.assertEqual(len(samples), total_samples)

        # Test actual negative samples percentage.
        negative_samples_counter = Counter([s.next_theorem_correct for s in samples])
        actual_negative_samples_ratio = float(negative_samples_counter[False]) / len(samples)
        self.assertAlmostEqual(
            negative_samples_percentage, actual_negative_samples_ratio, delta=0.3)

        # Test that every sample contains current, next and goal graphs.
        for s in samples:
            self.assertIsNotNone(s.current_graph)
            self.assertIsNotNone(s.next_theorem)
            self.assertIsNotNone(s.goal)
