import sys

from sklearn.model_selection import train_test_split

from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_SELECTION_MODEL_NAME, MAIN_MODEL_NAME, \
    SIMPLE_MODEL_TRAIN_OUTPUT_DIR, set_theorem_selector, TheoremSelector
from logipy.evaluation.evaluation import evaluate_theorem_selector_on_samples
from .train_config import TrainConfiguration
from .theorem_proving_model import TheoremProvingModel
from .gnn_model import GNNModel
from .simple_model import SimpleModel
from .dataset_generator import DatasetGenerator
from .io import export_generated_samples, export_theorems_and_properties
from .neural_theorem_selector import NeuralNextTheoremSelector


DATASET_SIZE = 100
MAX_DEPTH = 20
EPOCHS = 2
BATCH_SIZE = 20
TEST_SIZE = 0.25
RANDOM_EXPANSION_PROBABILITY = 0.
NEGATIVE_SAMPLES_PERCENTAGE = 0.7

EXPORT_SAMPLES = False
SAMPLES_TO_EXPORT = 400
EXPORT_PROPERTIES = False
SYSTEM_EVALUATION_AFTER_TRAIN = True
SYSTEM_COMPARISON_TO_DETERMINISTIC_AFTER_TRAIN = True


class MultiLogger(object):
    """Logger that enables simultaneous logging to file and stdout."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = filepath.open("w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def close(self):
        self.log.flush()
        self.log.close()


def load_or_train_model():
    """Attempts to load a neural model and if it fails, trains a new one.

    Loading a model automatically sets active theorem selector to DGCNN one.
    """
    loaded = set_theorem_selector(TheoremSelector.DGCNN)
    if not loaded:
        train_models()
        loaded = set_theorem_selector(TheoremSelector.DGCNN)
        if not loaded:
            raise RuntimeWarning("Failed to load neural model after training.")


def train_models(arch=None):
    # Export train log to file and to stdout simultaneously.
    old_stdout = sys.stdout
    sys.stdout = MultiLogger(get_scratchfile_path("out.txt"))

    properties = get_global_properties()

    models = []   # All models to be trained.
    configs = []  # Configs corresponding to trained models.

    if arch == "simple":
        # Train Simple NN model.
        models.append(SimpleModel("Simple Model", MODELS_DIR + "/" + MAIN_MODEL_NAME))
        configs.append(generate_simple_config())
    elif arch == "gnn":
        # Train GNN model.
        models.append(GNNModel("GNN Model", MODELS_DIR + "/" + GRAPH_SELECTION_MODEL_NAME))
        configs.append(generate_gnn_config())
    else:
        # Train everything.
        models.append(SimpleModel("Simple Model", MODELS_DIR + "/" + MAIN_MODEL_NAME))
        configs.append(generate_simple_config())
        models.append(GNNModel("GNN Model", MODELS_DIR + "/" + GRAPH_SELECTION_MODEL_NAME))
        configs.append(generate_gnn_config())

    dataset = generate_dataset(properties, configs[0])
    # Split train and test data.
    i_train, i_test = train_test_split(list(range(len(dataset))),
                                       test_size=configs[0].test_size,
                                       stratify=[int(s.is_next_theorem_correct()) for s in dataset])

    for m, conf in zip(models, configs):
        m.train(dataset, properties, i_train, i_val, conf)
        m.save()
        m.plot(conf.selection_models_dir)

    for m, conf in zip(models, configs):
        if conf.system_evaluation_after_train:
            _evaluate_model(
                m,
                [dataset[i] for i in i_train],
                [dataset[i] for i in i_val],
                conf
            )

    sys.stdout.close()
    sys.stdout = old_stdout


def generate_simple_config():
    return TrainConfiguration(
        DATASET_SIZE,
        MAX_DEPTH,
        EPOCHS,
        BATCH_SIZE,
        TEST_SIZE,
        EXPORT_SAMPLES,
        SAMPLES_TO_EXPORT,
        EXPORT_PROPERTIES,
        SYSTEM_EVALUATION_AFTER_TRAIN,
        False,
        RANDOM_EXPANSION_PROBABILITY,
        NEGATIVE_SAMPLES_PERCENTAGE,
        (get_scratchfile_path(SIMPLE_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "selection_models",
        (get_scratchfile_path(SIMPLE_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "termination_models"
    )


def generate_gnn_config():
    return TrainConfiguration(
        DATASET_SIZE,
        MAX_DEPTH,
        EPOCHS,
        BATCH_SIZE,
        TEST_SIZE,
        EXPORT_SAMPLES,
        SAMPLES_TO_EXPORT,
        EXPORT_PROPERTIES,
        SYSTEM_EVALUATION_AFTER_TRAIN,
        SYSTEM_COMPARISON_TO_DETERMINISTIC_AFTER_TRAIN,
        RANDOM_EXPANSION_PROBABILITY,
        NEGATIVE_SAMPLES_PERCENTAGE,
        (get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "selection_models",
        (get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "termination_models"
    )


def generate_dataset(properties, config: TrainConfiguration):
    generator = DatasetGenerator(properties, config.max_depth, config.dataset_size,
                                 random_expansion_probability=config.random_expansion_probability,
                                 negative_samples_percentage=config.negative_samples_percentage)

    if config.export_properties:
        print(f"\tExporting theorems and properties...")
        export_theorems_and_properties(generator.theorems, generator.valid_properties_to_prove)

    print(f"\tGenerating {config.dataset_size} samples...")
    graph_samples = []
    for i, s in enumerate(generator):
        if (i % 10) == 0 or i == config.dataset_size - 1:
            print(f"\t\tGenerated {i}/{config.dataset_size}...", end="\r")
        graph_samples.append(s)

    # for i, sample in enumerate(graph_samples[:5]):
    #     sample.visualize(f"Sample #{i+1}")

    if config.export_samples:
        print(f"\tExporting samples...")
        export_generated_samples(graph_samples, min(config.dataset_size, config.samples_to_export))

    return graph_samples


def _evaluate_model(model: TheoremProvingModel,
                    train_samples,
                    validation_samples,
                    config: TrainConfiguration):
    print("-" * 80)
    print(f"Evaluating {model.name.lower()} on proving synthetic samples...")
    print("-" * 80)
    theorem_selector = NeuralNextTheoremSelector(model)
    _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples)

    if config.system_comparison_to_deterministic_after_train:
        print("-" * 80)
        print("Evaluating deterministic selector on proving synthetic samples...")
        print("-" * 80)
        from logipy.logic.next_theorem_selectors import BetterNextTheoremSelector
        theorem_selector = BetterNextTheoremSelector()
        _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples)

    print("-" * 80)
    print(f"Evaluating {model.name.lower()}+deterministic selector on proving synthetic samples...")
    print("-" * 80)
    from logipy.logic.next_theorem_selectors import BetterNextTheoremSelector
    theorem_selector = [BetterNextTheoremSelector(), NeuralNextTheoremSelector(model)]
    _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples)


def _evaluate_theorem_selector(theorem_selector, train_samples, validation_samples):
    acc, fallout = evaluate_theorem_selector_on_samples(
        theorem_selector, train_samples, verbose=True)
    print("\tTesting dataset:  proving_acc: {} - proving_fallout: {}".format(
        round(acc, 4), round(fallout, 4)))

    val_acc, val_fallout = evaluate_theorem_selector_on_samples(
        theorem_selector, validation_samples, verbose=True)
    print("\tValidation dataset: val_proving_acc: {} - val_proving_fallout: {}".format(
        round(val_acc, 4), round(val_fallout, 4)))


if __name__ == "__main__":
    train_models()
