from logipy.graphs.timed_property_graph import TimedPropertyGraph
from .train_config import TrainConfiguration


class TheoremProvingModel:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def train(self, dataset, properties, config: TrainConfiguration):
        print("-" * 80)
        print("Active Training Configuration")
        config.print()
        print("-" * 80)
        print(f"Training {self.name}.")
        print("-" * 80)

        self.train_core(dataset, properties, config)

    def train_core(self, dataset, properties, config: TrainConfiguration):
        raise NotImplementedError

    def predict(self,
                current: TimedPropertyGraph,
                theorem_applications: list,
                goal: TimedPropertyGraph):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def plot(self, folder):
        raise NotImplementedError

    @staticmethod
    def load(path):
        raise NotImplementedError
