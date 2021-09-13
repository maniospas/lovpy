import sys

from tensorflow.keras.utils import plot_model

from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.models.graph_neural_model import train_gnn_theorem_proving_models
from logipy.models.io import save_gnn_models
from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_SELECTION_MODEL_NAME, GRAPH_TERMINATION_MODEL_NAME, GRAPH_ENCODER_NAME
from logipy.models.train_config import TrainConfiguration
from . import config


DATASET_SIZE = 5000
MAX_DEPTH = 20
EPOCHS = 100
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
    loaded = config.set_theorem_selector(config.TheoremSelector.DGCNN)
    if not loaded:
        train_models()
        loaded = config.set_theorem_selector(config.TheoremSelector.DGCNN)
        if not loaded:
            raise RuntimeWarning("Failed to load neural model after training.")


def train_models(arch=None):
    # Export train log to file and to stdout simultaneously.
    old_stdout = sys.stdout
    sys.stdout = MultiLogger(get_scratchfile_path("out.txt"))

    properties = get_global_properties()
    train_config = generate_config()

    if arch == "simple":
        # Train Simple NN model.
        train_theorem_proving_model(properties, train_config)
    elif arch == "gnn":
        # Train GNN model.
        train_gnn_model(properties, train_config)
    else:
        # Train everything.
        train_theorem_proving_model(properties, train_config)
        train_gnn_model(properties, train_config)

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


def train_gnn_model(properties, train_config):
    next_theorem_model, proving_termination_model, encoder = train_gnn_theorem_proving_models(
        properties,
        train_config
    )
    save_gnn_models(next_theorem_model, proving_termination_model, encoder)

    # Save GNN model also in scratchdir, for easy retrieval.
    scratch_model_out_base = get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR
    # save_gnn_models(
    #     next_theorem_model,
    #     proving_termination_model,
    #     encoder,
    #     scratch_model_out_base / GRAPH_SELECTION_MODEL_NAME,
    #     scratch_model_out_base / GRAPH_TERMINATION_MODEL_NAME,
    #     scratch_model_out_base / GRAPH_ENCODER_NAME
    # )
    plot_model(
        next_theorem_model,
        to_file=scratch_model_out_base / "dgcnn_next_theorem_model.png",
        show_shapes=True,
        show_layer_names=True
    )
    # plot_model(
    #     proving_termination_model,
    #     to_file=scratch_model_out_base / "dgcnn_termination_model.png",
    #     show_shapes=True,
    #     show_layer_names=True
    # )


if __name__ == "__main__":
    train_models()
