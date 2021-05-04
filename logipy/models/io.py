import pickle
from tensorflow.keras.models import load_model

from . import GRAPH_MODEL_PATH, GRAPH_ENCODER_PATH


def save_gnn_model(model, encoder):
    """Saves given gnn model along with nodes encoder to disk."""
    model.save(GRAPH_MODEL_PATH)
    with GRAPH_ENCODER_PATH.open("wb") as f:
        pickle.dump(encoder, f)


def load_gnn_model():
    """Loads gnn model along with nodes encoder from disk."""
    model = None
    encoder = None

    if GRAPH_MODEL_PATH.exists() and GRAPH_ENCODER_PATH.exists():
        model = load_model(GRAPH_MODEL_PATH)
        with GRAPH_ENCODER_PATH.open("rb") as f:
            encoder = pickle.load(f)

    return model, encoder
