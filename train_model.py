from tensorflow.keras.utils import plot_model

# from logipy.models.theorem_proving_model import train_theorem_proving_model
from logipy.models.graph_neural_model import train_gnn_theorem_proving_model
from logipy.models.io import save_gnn_model
from logipy.logic.properties import get_global_properties
from logipy.config import get_scratchfile_path, GRAPH_MODEL_TRAIN_OUTPUT_DIR, MODELS_DIR, \
    GRAPH_MODEL_NAME, GRAPH_ENCODER_NAME


if __name__ == "__main__":
    properties = get_global_properties()
    # train_theorem_proving_model(properties)
    model, encoder = train_gnn_theorem_proving_model(properties)
    save_gnn_model(model, encoder)

    # Save gnn model also in scratchdir, for easy retrieval.
    scratch_model_out_base = get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR) / MODELS_DIR
    save_gnn_model(
        model,
        encoder,
        scratch_model_out_base/GRAPH_MODEL_NAME,
        scratch_model_out_base/GRAPH_ENCODER_NAME
    )
    plot_model(
        model,
        to_file=scratch_model_out_base/"model.png",
        show_shapes=True,
        show_layer_names=True
    )
