import numpy as np

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .graph_neural_model import NextTheoremSamplesGenerator, create_three_padded_generators, \
    convert_timedpropertygraph_to_stellargraph


THRESHOLD = 0.5


class GraphNeuralNextTheoremSelector(NextTheoremSelector):
    """A Next Theorem Selector that utilizes Graph Neural Networks based models."""

    def __init__(self, model, nodes_encoder):
        """
        :param model: A model that accepts as input NextTheoremSamplesGenerator generators.
        :param nodes_encoder: An encoder that encodes nodes of the Graph into feature vectors.
        """
        self.model = model
        self.encoder = nodes_encoder

    def select_next(self, graph, theorem_applications, goal, previous_applications):
        current_graph, norm = convert_timedpropertygraph_to_stellargraph(graph, self.encoder)
        goal_graph, _ = convert_timedpropertygraph_to_stellargraph(goal, self.encoder)

        current_generator, goal_generator, next_generator = create_three_padded_generators(
            [current_graph] * len(theorem_applications),
            [goal_graph] * len(theorem_applications),
            [convert_timedpropertygraph_to_stellargraph(
                t_app.actual_implication, self.encoder, norm)[0] for t_app in theorem_applications]
        )

        inference_generator = NextTheoremSamplesGenerator(
            current_generator, goal_generator, next_generator)

        scores = self.model.predict(inference_generator)
        if max(scores) < THRESHOLD:
            return None
        else:
            return theorem_applications[np.argmax(scores, axis=0)[0]]
