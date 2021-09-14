import numpy as np

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .simple_model import SimpleModel


class NeuralNextTheoremSelector(NextTheoremSelector):
    def __init__(self, model: SimpleModel):
        self.model = model

    def select_next(self, graph, theorem_applications, goal, previous_applications, label=None):
        scores = self.model.predict(graph, theorem_applications, goal)
        return theorem_applications[np.argmax(scores, axis=0)[0]]
