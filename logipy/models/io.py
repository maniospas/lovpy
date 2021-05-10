import pickle
from tensorflow.keras.models import load_model

# Paths about simple NN model.
main_model_path = None
predicates_map_path = None

# Paths about DGCNN model.
graph_model_path = None
graph_encoder_path = None

# Paths about sample visualization.
current_graph_path = None
goal_graph_path = None
next_graph_path = None


def save_gnn_model(model, encoder):
    """Saves given gnn model along with nodes encoder to disk."""
    model.save(graph_model_path)
    with graph_encoder_path.open("wb") as f:
        pickle.dump(encoder, f)


def load_gnn_model():
    """Loads gnn model along with nodes encoder from disk."""
    model = None
    encoder = None

    if graph_model_path.exists() and graph_encoder_path.exists():
        model = load_model(graph_model_path)
        with graph_encoder_path.open("rb") as f:
            encoder = pickle.load(f)

    return model, encoder


def model_file_exists():
    return main_model_path.exists()
