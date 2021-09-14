import sys

from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_SELECTION_MODEL_NAME, GRAPH_TERMINATION_MODEL_NAME, GRAPH_ENCODER_NAME, MAIN_MODEL_NAME, \
    set_theorem_selector, TheoremSelector
from .train_config import TrainConfiguration
from .gnn_model import GNNModel
from .simple_model import SimpleModel
from .dataset_generator import DatasetGenerator
from .io import export_generated_samples, export_theorems_and_properties


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
    train_config = generate_config()

    models = []  # All models to be trained.

    if arch == "simple":
        # Train Simple NN model.
        models.append(SimpleModel("Simple Model", MODELS_DIR + "/" + MAIN_MODEL_NAME))
    elif arch == "gnn":
        # Train GNN model.
        models.append(GNNModel("GNN Model", MODELS_DIR + "/" + GRAPH_SELECTION_MODEL_NAME))
    else:
        # Train everything.
        models.append(SimpleModel("Simple Model", MODELS_DIR + "/" + MAIN_MODEL_NAME))
        models.append(GNNModel("GNN Model", MODELS_DIR + "/" + GRAPH_SELECTION_MODEL_NAME))

    dataset = generate_dataset(properties, train_config)

    for m in models:
        m.train(dataset, properties, train_config)
        m.save()

        training_folder = (get_scratchfile_path(f"train_{m.name.lower().replace(' ', '_')}")
                           / MODELS_DIR)
        if not training_folder.exists():
            training_folder.mkdir(parents=True)
        m.plot(training_folder)

    sys.stdout.close()
    sys.stdout = old_stdout


def generate_config():
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


if __name__ == "__main__":
    train_models()
