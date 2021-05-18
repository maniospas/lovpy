import pickle
import random

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

# Paths about training samples exporting.
graph_model_samples_export_dir_path = None


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


def export_generated_samples(samples, max_num=None):
    if graph_model_samples_export_dir_path is None:
        raise RuntimeError("Models module was not correctly initialized.")

    if not graph_model_samples_export_dir_path.exists():
        graph_model_samples_export_dir_path.mkdir(parents=True)

    if not max_num:
        max_num = len(samples)
    samples = random.sample(samples, max_num)
    for i, s in enumerate(samples):
        print(f"\t\tExported {i+1}/{len(samples)}", end="\r")
        s.visualize(f"Sample #{i+1}", graph_model_samples_export_dir_path / f"sample{i + 1}.png")


def model_file_exists():
    return main_model_path.exists()
