import json

from tensorflow.keras.models import load_model

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .theorem_proving_model import PredicatesMap, convert_state_to_matrix
from . import io


class NeuralNextTheoremSelector(NextTheoremSelector):
    def __init__(self):
        self.model = load_model(io.main_model_path)
        self.predicates_map = PredicatesMap(pred_map=json.load(io.predicates_map_path.open('r')))

    def select_next(self, graph, theorem_applications, goal, previous_applications):
        scores = []

        for application in theorem_applications:
            data = convert_state_to_matrix(
                graph, application.implication_graph, goal, self.predicates_map)

            score = self.model(data)[0]
            scores.append(score)

        return theorem_applications[scores.index(max(scores))]
