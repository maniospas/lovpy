import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, confusion_matrix

from logipy.logic.prover import prove_property


class ModelEvaluationOnTheoremProvingCallback(Callback):
    """Callback to evaluate a neural theorem proving model on its ability to prove theorems.

    The ability to prove theorems is evaluated in the sense that a model should be able
    to produce a sequence of theorem applications in order to finally prove or consider as
    non-provable the target theorem. This binary output of the prover (proved | not proved)
    is compared against the ground truth value of each sample (provable | not provable).

    It outputs the following metrics:
        -Accuracy
        -Fallout
        -AUC (not yet implemented)
    """

    def __init__(self, train_samples, validation_samples, nodes_encoder):
        super().__init__()
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.nodes_encoder = nodes_encoder

    def on_epoch_end(self, epoch, logs=None):
        acc, fallout = compute_accuracy_fallout_on_samples_proving(
            self.train_samples, self.model, self.nodes_encoder)
        val_acc, val_fallout = compute_accuracy_fallout_on_samples_proving(
            self.validation_samples, self.model, self.nodes_encoder)

        if logs:
            logs["proving_acc"] = acc
            logs["proving_fallout"] = fallout
            logs["val_proving_acc"] = acc
            logs["val_proving_fallout"] = fallout

        print(" - proving_acc: {} - proving_fallout: {}".format(round(acc, 4), round(fallout, 4)))
        print(" - val_proving_acc: {} - val_proving_fallout: {}".format(
            round(val_acc, 4), round(val_fallout, 4)))


def compute_accuracy_fallout_on_samples_proving(samples, model, nodes_encoder):
    from logipy.models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
    theorem_selector = GraphNeuralNextTheoremSelector(model, nodes_encoder)

    for i, s in enumerate(samples):
        proved, _, _ = prove_property(
            s.current_graph,
            s.goal,
            s.all_theorems,
            theorem_selector=theorem_selector
        )

        if i == 0:
            predicted_proved = int(proved)
            actual_proved = int(s.is_provable)
        else:
            predicted_proved = np.vstack((predicted_proved, int(proved)))
            actual_proved = np.vstack((actual_proved, int(s.is_provable)))

    acc = accuracy_score(actual_proved, predicted_proved)
    conf_matrix = confusion_matrix(actual_proved, predicted_proved)
    fallout = conf_matrix[0][1] / np.sum(conf_matrix, axis=1)[1]

    return acc, fallout
