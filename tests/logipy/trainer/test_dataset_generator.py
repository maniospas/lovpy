import unittest

from tests.logipy.importer.sample_properties import get_threading_sample_properties
from logipy.trainer.dataset_generator import *


class TestDatasetGenerator(unittest.TestCase):

    def test_simple_threading_dataset(self):
        generator = DatasetGenerator(get_threading_sample_properties(), 5, 10)
        samples = list(generator)
        self.assertEqual(len(samples), 10)
        for s in samples:
            s.current_graph.visualize()