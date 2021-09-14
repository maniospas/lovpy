import numpy as np

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .theorem_proving_model import TheoremProvingModel


class NeuralNextTheoremSelector(NextTheoremSelector):
    def __init__(self, model: TheoremProvingModel):
        self.model = model

    def select_next(self, graph, theorem_applications, goal, previous_applications, label=None):
        scores = self.model.predict(graph, theorem_applications, goal)
        return theorem_applications[np.argmax(scores, axis=0)[0]]
