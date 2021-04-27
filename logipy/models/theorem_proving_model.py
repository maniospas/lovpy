import random
import json

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import logipy.logic.prover as prover
from logipy.graphs.timed_property_graph import TimedPropertyGraph
from logipy.graphs.logical_operators import NotOperator
from . import MAIN_MODEL_PATH, PREDICATES_MAP_PATH
from .dataset_generator import DatasetGenerator


DATASET_SIZE = 1000
MAX_DEPTH = 12
EPOCHS = 10
BATCH_SIZE = 10
PREDICATES_NUM = 10


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
            basic_predicates = prover.convert_implication_to_and(prop).get_basic_predicates()

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


def train_theorem_proving_model(properties):
    data, outputs, predicates_map = create_dataset(properties)
    model = create_dense_model(predicates_map)

    model.fit(x=data, y=outputs, epochs=EPOCHS, batch_size=1)

    model.save(MAIN_MODEL_PATH)
    json.dump(predicates_map.map, PREDICATES_MAP_PATH.open('w'))

    return model


def create_dense_model(predicates_map):
    dim = 3 * PREDICATES_NUM * (len(predicates_map) + 1)
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, activation="sigmoid"))
    model.add(Dense(dim/4, activation="sigmoid"))
    model.add(Dense(1))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model


def create_dataset(properties):
    generator = DatasetGenerator(properties, MAX_DEPTH, DATASET_SIZE)
    samples = list(generator)

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
    output_data = int(sample.is_provable and sample.is_correct)

    return input_data, output_data


def convert_property_graph_to_matrix(property_graph, predicates_map):
    data = np.zeros((PREDICATES_NUM, len(predicates_map) + 1))
    if property_graph:
        predicates = property_graph.get_basic_predicates()
        predicates_id = []
        predicates_timestamp = []
        max_timestamp = property_graph.get_most_recent_timestamp()
        # Clean predicates from the ones not belonging in any properties.
        for i in range(len(predicates)):
            p = predicates[i]
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
