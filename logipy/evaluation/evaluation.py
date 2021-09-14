import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from logipy.logic.prover import prove_property


def evaluate_theorem_selector_on_samples(theorem_selector, samples, verbose=False):
    """Evaluates a theorem selector on its ability to prove theorems.

    The ability to prove theorems is evaluated in the sense that a model should be able
    to produce a sequence of theorem applications in order to finally prove or consider as
    non-provable the target theorem. This binary output of the prover (proved | not proved)
    is compared against the ground truth value of each sample (provable | not provable).

    It outputs the following metrics:
        -Accuracy
        -Fallout
        -AUC (not yet implemented)

    :param samples:
    :param theorem_selector: Theorem selector to evaluate.
    :param verbose:

    :return:
        -Accuracy metric.
        -Fallout metric.
    """
    if not isinstance(theorem_selector, list):
        theorem_selector = [theorem_selector]

    for i, s in enumerate(samples):
        if verbose:
            print("\t{}/{} validating...".format(i, len(samples)), end="\r")

        proved = False
        for ts in theorem_selector:
            proved, _, _ = prove_property(
                s.current_graph,
                s.goal,
                s.all_theorems,
                theorem_selector=ts
            )
            if proved:
                break

        if i == 0:
            predicted_proved = int(proved)
            actual_proved = int(s.is_provable)
        else:
            predicted_proved = np.vstack((predicted_proved, int(proved)))
            actual_proved = np.vstack((actual_proved, int(s.is_provable)))

    acc = accuracy_score(actual_proved, predicted_proved)
    conf_matrix = confusion_matrix(actual_proved, predicted_proved, labels=[0, 1])
    fallout = conf_matrix[0][1] / np.sum(conf_matrix, axis=1)[1]

    return acc, fallout
