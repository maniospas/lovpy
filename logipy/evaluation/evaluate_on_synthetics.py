from logipy.logic.properties import get_global_properties
from logipy.models.dataset_generator import DatasetGenerator
from logipy.models.gnn_model import GNNModel
from logipy.models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
from logipy.evaluation.evaluation import evaluate_theorem_selector_on_samples
from logipy.logic.next_theorem_selectors import BetterNextTheoremSelector


DATASET_SIZE = 1000
MAX_DEPTH = 20
RANDOM_EXPANSION_PROBABILITY = 0.
NEGATIVE_SAMPLES_PERCENTAGE = 0.


def evaluate():
    print("-" * 80)
    print("Evaluating proving systems on synthetic theorems...")
    print("-" * 80)

    properties = get_global_properties()
    generator = DatasetGenerator(properties, MAX_DEPTH, DATASET_SIZE,
                                 random_expansion_probability=RANDOM_EXPANSION_PROBABILITY,
                                 negative_samples_percentage=NEGATIVE_SAMPLES_PERCENTAGE)

    print(f"\tGenerating {DATASET_SIZE} samples...")
    samples = []
    for i, s in enumerate(generator):
        if (i % 10) == 0 or i == DATASET_SIZE - 1:
            print(f"\t\tGenerated {i}/{DATASET_SIZE}...", end="\r")
        samples.append(s)

    print("-" * 80)
    print("Evaluating hybrid proving system...")
    print("-" * 80)
    evaluate_hybrid_selector(samples)

    print("-" * 80)
    print("Evaluating DGCNN proving system...")
    print("-" * 80)
    evaluate_dgcnn_selector(samples)

    # print("-" * 80)
    # print("Evaluating Simple NN proving system...")
    # print("-" * 80)
    # evaluate_simple_nn_selector(samples)

    print("-" * 80)
    print("Evaluating deterministic proving system...")
    print("-" * 80)
    evaluate_deterministic_selector(samples)

    # print("-" * 80)
    # print("Evaluating random selection proving system...")
    # print("-" * 80)
    # evaluate_random_selector(samples)


def evaluate_hybrid_selector(samples):
    gnn_selector = GraphNeuralNextTheoremSelector(GNNModel.load())
    det_selector = BetterNextTheoremSelector()
    acc, fallout = evaluate_theorem_selector_on_samples([det_selector, gnn_selector],
                                                        samples, verbose=True)
    print("\tproving_acc: {} - proving_fallout: {}".format(round(acc, 4), round(fallout, 4)))


def evaluate_dgcnn_selector(samples):
    gnn_selector = GraphNeuralNextTheoremSelector(GNNModel.load())
    acc, fallout = evaluate_theorem_selector_on_samples(gnn_selector, samples, verbose=True)
    print("\tproving_acc: {} - proving_fallout: {}".format(round(acc, 4), round(fallout, 4)))


def evaluate_simple_nn_selector(samples):
    pass  # TODO: Implement


def evaluate_deterministic_selector(samples):
    det_selector = BetterNextTheoremSelector()
    acc, fallout = evaluate_theorem_selector_on_samples(det_selector, samples, verbose=True)
    print("\tproving_acc: {} - proving_fallout: {}".format(round(acc, 4), round(fallout, 4)))


def evaluate_random_selector(samples):
    pass  # TODO: Implement


if __name__ == "__main__":
    evaluate()
