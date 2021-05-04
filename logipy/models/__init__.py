import logipy.config


MAIN_MODEL_NAME = "main_model"
PREDICATES_MAP_NAME = "main_model_predicates.json"
MAIN_MODEL_PATH = logipy.config.get_models_dir_path(MAIN_MODEL_NAME)
PREDICATES_MAP_PATH = logipy.config.get_models_dir_path(PREDICATES_MAP_NAME)

GRAPH_MODEL_NAME = "gnn_model"
GRAPH_ENCODER_NAME = "graph_nodes_encoder"
GRAPH_MODEL_PATH = logipy.config.get_models_dir_path(GRAPH_MODEL_NAME)
GRAPH_ENCODER_PATH = logipy.config.get_models_dir_path(GRAPH_ENCODER_NAME)


def model_file_exists():
    return MAIN_MODEL_PATH.exists()
