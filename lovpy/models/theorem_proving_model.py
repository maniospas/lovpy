from lovpy.graphs.timed_property_graph import TimedPropertyGraph
from .train_config import TrainConfiguration


class TheoremProvingModel:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def train(self, dataset, properties, i_train, i_val, config: TrainConfiguration):
        print("-" * 80)
        print("Active Training Configuration")
        config.print()
        print("-" * 80)
        print(f"Training {self.name}.")
        print("-" * 80)

        return self.train_core(dataset, properties, i_train, i_val, config)

    def train_core(self, dataset, properties, i_train, i_val, config: TrainConfiguration):
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


class TrainingResults:
    def __init__(self):
        self.train_loss = None
        self.eval_loss = None
        self.train_accuracy = None
        self.eval_accuracy = None
        self.train_auc = None
        self.eval_auc = None

    def __add__(self, other):
        if isinstance(other, TrainingResults):
            results = TrainingResults()
            results.train_loss = self.train_loss + other.train_loss
            results.eval_loss = self.eval_loss + other.eval_loss
            results.train_accuracy = self.train_accuracy + other.train_accuracy
            results.eval_accuracy = self.eval_accuracy + other.eval_accuracy
            results.train_auc = self.train_auc + other.train_auc
            results.eval_auc = self.eval_auc + other.eval_auc
            return results
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            results = TrainingResults()
            results.train_loss = self.train_loss / other
            results.eval_loss = self.eval_loss / other
            results.train_accuracy = self.train_accuracy / other
            results.eval_accuracy = self.eval_accuracy / other
            results.train_auc = self.train_auc / other
            results.eval_auc = self.eval_auc / other
            return results
        else:
            return NotImplemented
