import sys

from tensorflow.keras.utils import plot_model

# from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.models.graph_neural_model import train_gnn_theorem_proving_models
from logipy.models.io import save_gnn_models
from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_SELECTION_MODEL_NAME, GRAPH_TERMINATION_MODEL_NAME, GRAPH_ENCODER_NAME
from logipy.models.train_config import TrainConfiguration


DATASET_SIZE = 60
MAX_DEPTH = 20
EPOCHS = 5
BATCH_SIZE = 40
TEST_SIZE = 0.25
RANDOM_EXPANSION_PROBABILITY = 0.4
NEGATIVE_SAMPLES_PERCENTAGE = 0.7

EXPORT_SAMPLES = True
EXPORT_PROPERTIES = True
SYSTEM_EVALUATION_AFTER_TRAIN = True


class MultiLogger(object):
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


def train_models():
    # Output train log to file.
    old_stdout = sys.stdout
    sys.stdout = MultiLogger(get_scratchfile_path("out.txt"))

    properties = get_global_properties()
    config = TrainConfiguration(
        DATASET_SIZE,
        MAX_DEPTH,
        EPOCHS,
        BATCH_SIZE,
        TEST_SIZE,
        EXPORT_SAMPLES,
        EXPORT_PROPERTIES,
        SYSTEM_EVALUATION_AFTER_TRAIN,
        RANDOM_EXPANSION_PROBABILITY,
        NEGATIVE_SAMPLES_PERCENTAGE,
        (get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "selection_models",
        (get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR) / "termination_models"
    )

    # Train Simple NN model.
    # train_theorem_proving_model(properties)

    # Train GNN model.
    train_gnn_model(properties, config)

    sys.stdout.close()
    sys.stdout = old_stdout


def train_gnn_model(properties, config):
    next_theorem_model, proving_termination_model, encoder = train_gnn_theorem_proving_models(
        properties,
        config
    )
    save_gnn_models(next_theorem_model, proving_termination_model, encoder)

    # Save GNN model also in scratchdir, for easy retrieval.
    scratch_model_out_base = get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR
    save_gnn_models(
        next_theorem_model,
        proving_termination_model,
        encoder,
        scratch_model_out_base / GRAPH_SELECTION_MODEL_NAME,
        scratch_model_out_base / GRAPH_TERMINATION_MODEL_NAME,
        scratch_model_out_base / GRAPH_ENCODER_NAME
    )
    plot_model(
        next_theorem_model,
        to_file=scratch_model_out_base / "dgcnn_next_theorem_model.png",
        show_shapes=True,
        show_layer_names=True
    )
    plot_model(
        proving_termination_model,
        to_file=scratch_model_out_base / "dgcnn_termination_model.png",
        show_shapes=True,
        show_layer_names=True
    )


if __name__ == "__main__":
    train_models()
