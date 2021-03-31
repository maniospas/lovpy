import unittest

from tests.logipy.importer.sample_properties import get_threading_sample_properties
from logipy.trainer.dataset_generator import *


class TestGenerateNextTheoremDataset(unittest.TestCase):

    def test_simple_threading_dataset(self):
        sample_properties = get_threading_sample_properties()

        dataset = list(generate_next_theorem_dataset(sample_properties, 3))
        self.assertTrue(True)