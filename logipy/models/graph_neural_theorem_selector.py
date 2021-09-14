import numpy as np

from logipy.logic.next_theorem_selectors import NextTheoremSelector
from .gnn_model import GNNModel
from .io import export_grouped_instance


exported = 0  # Number of next theorem selection processes exported so far.


class GraphNeuralNextTheoremSelector(NextTheoremSelector):
    """A Next Theorem Selector that utilizes Graph Neural Networks based models."""

    def __init__(self, model: GNNModel, export=False):
        """
        :param GNNModel model: GNN model to be used for next theorem selection.
        :param export:
        """
        self.model = model
        self.export = export

    def select_next(self, graph, theorem_applications, goal, previous_applications, label=None):
        global exported

        scores = self.model.predict(graph, theorem_applications, goal)
        next_application = theorem_applications[np.argmax(scores, axis=0)[0]]

        if self.export:
            for i, t_app in enumerate(theorem_applications):
                export_grouped_instance(graph, goal, t_app.actual_implication,
                                        f"Predicted Score: {scores[i]}",
                                        goal.property_textual_representation,
                                        label,
                                        exported+1)
            exported += 1

        return next_application
