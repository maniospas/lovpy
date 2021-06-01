import numpy as np

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .graph_neural_model import ProvingModelSamplesGenerator, create_padded_generators, \
    convert_timedpropertygraph_to_stellargraph
from .io import export_grouped_instance


exported = 0  # Number of next theorem selection processes exported so far.


class GraphNeuralNextTheoremSelector(NextTheoremSelector):
    """A Next Theorem Selector that utilizes Graph Neural Networks based models."""

    def __init__(self, selection_model, termination_model, nodes_encoder, export=False):
        """
        :param selection_model: A model that accepts as input NextTheoremSamplesGenerator
                generators.
        :param termination_model:
        :param nodes_encoder: An encoder that encodes nodes of the Graph into feature vectors.
        :param export:
        """
        self.selection_model = selection_model
        self.termination_model = termination_model
        self.encoder = nodes_encoder
        self.export = export

    def select_next(self, graph, theorem_applications, goal, previous_applications, label=None):
        global exported

        # # Don't use the last applied theorem.
        # used_theorems = \
        #     [previous_applications[-1].implication_graph] if previous_applications else []
        # unused_applications = [t for t in theorem_applications
        #                        if t.implication_graph not in used_theorems]
        # if not unused_applications:
        #     return None

        current_graph, norm = convert_timedpropertygraph_to_stellargraph(graph, self.encoder)
        goal_graph, _ = convert_timedpropertygraph_to_stellargraph(goal, self.encoder)

        should_terminate = False  # self._should_terminate(current_graph, goal_graph)

        if not should_terminate:
            next_application, scores = self._get_next_theorem_application(
                current_graph, goal_graph, norm, theorem_applications)

            if self.export:
                for i, t_app in enumerate(theorem_applications):
                    export_grouped_instance(graph, goal, t_app.actual_implication,
                                            f"Predicted Score: {scores[i]}",
                                            goal.property_textual_representation,
                                            label,
                                            exported+1)
                exported += 1

            return next_application

        return None

    def _get_next_theorem_application(self, current_graph, goal_graph, norm, theorem_applications):
        current_generator, goal_generator, next_generator = create_padded_generators(
            [current_graph] * len(theorem_applications),
            [goal_graph] * len(theorem_applications),
            [convert_timedpropertygraph_to_stellargraph(
                t_app.actual_implication, self.encoder, norm)[0] for t_app in
             theorem_applications]
        )

        inference_generator = ProvingModelSamplesGenerator(
            current_generator, goal_generator, next_generator)

        scores = self.selection_model.predict(inference_generator)

        return theorem_applications[np.argmax(scores, axis=0)[0]], scores

    def _should_terminate(self, current_graph, goal_graph):
        current_generator, goal_generator = create_padded_generators([current_graph], [goal_graph])
        inference_generator = ProvingModelSamplesGenerator(current_generator, goal_generator)
        termination_values = self.termination_model.predict(inference_generator)
        should_terminate = np.argmax(termination_values, axis=1)[0]
        return bool(should_terminate)
