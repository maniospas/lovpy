from tensorflow.keras.utils import plot_model

# from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.models.graph_neural_model import train_gnn_theorem_proving_models
from logipy.models.io import save_gnn_models
from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_SELECTION_MODEL_NAME, GRAPH_TERMINATION_MODEL_NAME, GRAPH_ENCODER_NAME


EXPORT_SAMPLES = True
EXPORT_PROPERTIES = True
SYSTEM_EVALUATION_AFTER_TRAIN = True


def train_models():
    properties = get_global_properties()

    # Train Simple NN model.
    # train_theorem_proving_model(properties)

    # Train GNN model.
    train_gnn_model(properties)


def train_gnn_model(properties):
    next_theorem_model, proving_termination_model, encoder = train_gnn_theorem_proving_models(
        properties,
        export_samples=EXPORT_SAMPLES,
        export_properties=EXPORT_PROPERTIES,
        system_validation_after_train=SYSTEM_EVALUATION_AFTER_TRAIN
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
