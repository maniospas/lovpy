import random
import json

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

import logipy.logic.properties
from logipy.graphs.timed_property_graph import TimedPropertyGraph
from logipy.graphs.logical_operators import NotOperator
from . import io
from .train_config import TrainConfiguration
from .theorem_proving_model import TheoremProvingModel


PREDICATES_NUM = 10


class SimpleModel(TheoremProvingModel):
    """A simple model based on an MLP for next theorem evaluation."""

    def __init__(self, name, path, model=None, predicates_map=None):
        super().__init__(name, path)
        self.model = model
        self.predicates_map = predicates_map

    def train_core(self, dataset, properties, config: TrainConfiguration):
        data, outputs, self.predicates_map = create_training_data(properties, dataset, config)
        self.model = create_dense_model(self.predicates_map)
        self.model.fit(x=data, y=outputs, epochs=config.epochs, batch_size=config.batch_size)

    def predict(self,
                current: TimedPropertyGraph,
                theorem_applications: list,
                goal: TimedPropertyGraph):
        scores = []

        for application in theorem_applications:
            data = convert_state_to_matrix(
                current, application.implication_graph, goal, self.predicates_map)

            score = self.model(data)[0]
            scores.append(score)

        return np.array(scores)

    def save(self):
        self.model.save(io.main_model_path)
        json.dump(self.predicates_map.map, io.predicates_map_path.open('w'))

    def plot(self, folder):
        plot_model(
            self.model,
            to_file=folder / f"{self.name}.png",
            show_shapes=True,
            show_layer_names=True
        )

    @staticmethod
    def load(path):
        mlp = load_model(io.main_model_path)
        predicates_map = PredicatesMap(pred_map=json.load(io.predicates_map_path.open('r')))
        if mlp and predicates_map:
            return SimpleModel("Simple Model", path, model=mlp, predicates_map=predicates_map)
        else:
            return None


class PredicatesMap:
    def __init__(self, properties=None, pred_map=None):
        if properties:
            self.map = {}
            self.properties = properties
            self._build_map()
        if pred_map:
            self.map = pred_map

    def __getitem__(self, item):
        if not isinstance(item, TimedPropertyGraph):
            raise RuntimeError("PredicatesMap can only be used for TimedPropertyGraph lookup.")

        text, is_negated = self._get_base_text(item)
        if is_negated:
            text = f"NOT({text})"

        if text in self.map:
            return self.map[text]
        else:
            return 0

    def __len__(self):
        return len(self.map) + 1

    def _build_map(self):
        for prop in self.properties:
            import logipy.logic.prover as prover
            basic_predicates = logipy.logic.properties.convert_implication_to_and(
                prop).get_basic_predicates()

            for pred in basic_predicates:
                base_text, _ = self._get_base_text(pred)
                negative_base_text = f"NOT({base_text})"
                self.map[base_text] = 0  # placeholder value
                self.map[negative_base_text] = 0  # placeholder value

        i = 1  # 0 deserved for None
        for pred_name in self.map.keys():
            self.map[pred_name] = i
            i += 1

    @staticmethod
    def _get_base_text(predicate_graph):
        if isinstance(predicate_graph.get_root_node(), NotOperator):
            pred_name = str(
                list(predicate_graph.graph.successors(predicate_graph.get_root_node()))[0])
            is_negated = True
        else:
            pred_name = str(predicate_graph.get_root_node())
            is_negated = False
        return pred_name, is_negated


def create_dense_model(predicates_map):
    dim = 3 * PREDICATES_NUM * (len(predicates_map) + 1)
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, activation="relu"))
    model.add(Dense(dim, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model


def create_training_data(properties, samples, train_config: TrainConfiguration):
    predicates_map = PredicatesMap(properties)

    data = np.zeros((len(samples), PREDICATES_NUM*3*(len(predicates_map)+1)))
    outputs = np.zeros(len(samples))

    for i in range(len(samples)):
        data[i], outputs[i] = convert_sample_to_data(samples[i], predicates_map)

    return data, outputs, predicates_map


def convert_sample_to_data(sample, predicates_map):
    current_table = convert_property_graph_to_matrix(sample.current_graph, predicates_map)
    next_table = convert_property_graph_to_matrix(sample.next_theorem, predicates_map)
    goal_table = convert_property_graph_to_matrix(sample.goal, predicates_map)

    input_data = np.concatenate((current_table, next_table, goal_table), axis=0)
    output_data = int(sample.is_provable and sample.next_theorem_correct)

    return input_data, output_data


def convert_property_graph_to_matrix(property_graph, predicates_map):
    data = np.zeros((PREDICATES_NUM, len(predicates_map) + 1))
    if property_graph:
        predicates = property_graph.get_basic_predicates()
        predicates_id = []
        predicates_timestamp = []
        max_timestamp = property_graph.get_most_recent_timestamp()
        # Clean predicates from the ones not belonging in any properties.
        for i, p in enumerate(predicates):
            p_id = predicates_map[p]
            if p_id > 0:
                predicates_id.append(p_id)
                predicates_timestamp.append(p.get_most_recent_timestamp())
        # Sample predicates sequence to get at most PREDICATES_NUM predicates.
        if len(predicates_id) > PREDICATES_NUM:
            indexes_to_keep = set(random.sample(list(range(0, len(predicates_id))), PREDICATES_NUM))
            predicates_id = [p_id for p_id in predicates_id if p_id in indexes_to_keep]
            predicates_timestamp = [p_t for p_t in predicates_timestamp
                                    if p_t in predicates_timestamp]

        for i in range(len(predicates_id)):
            try:
                data[i, predicates_id[i]] = 1
            except Exception:
                print("i")
            if max_timestamp._value > 0:
                data[i, -1] = \
                    float(predicates_timestamp[i]._value) / float(abs(max_timestamp._value))
            else:
                data[i, -1] = float(predicates_timestamp[i]._value)
    return data.flatten()


def convert_state_to_matrix(current_graph, next_theorem, goal_property, predicates_map):
    """Converts a triple defining the current state of theorem proving to inference data."""
    current_table = convert_property_graph_to_matrix(current_graph, predicates_map)
    next_table = convert_property_graph_to_matrix(next_theorem, predicates_map)
    goal_table = convert_property_graph_to_matrix(goal_property, predicates_map)
    return np.concatenate((current_table, next_table, goal_table), axis=0).reshape((1, -1))
