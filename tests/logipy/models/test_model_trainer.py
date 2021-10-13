import unittest

from lovpy.models.simple_model import *
from tests.lovpy.importer.sample_properties import get_threading_sample_properties


class TestModelTrainer(unittest.TestCase):

    def test_train_theorem_proving_model(self):
        # model = train_theorem_proving_model(get_threading_sample_properties())
        pass


if __name__ == "__main__":
    TestModelTrainer().test_train_theorem_proving_model()
