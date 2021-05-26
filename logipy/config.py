import logging
from enum import Enum
from pathlib import Path
from datetime import datetime

from logipy.logic.next_theorem_selectors import set_default_theorem_selector, \
    get_default_theorem_selector, BetterNextTheoremSelector
import logipy.graphs
import logipy.models
import logipy.logic.prover
from logipy.models.neural_theorem_selector import NeuralNextTheoremSelector
from logipy.models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
from logipy.models.io import load_gnn_models


LOGIPY_ROOT_PATH = Path(__file__).absolute().parent  # Absolute path of logipy's installation.

LOGGER_NAME = "logipy"

# Attributes controlling graph visualization.
SCRATCHDIR_PATH = LOGIPY_ROOT_PATH.parent / "_temp/"
GRAPHVIZ_OUT_FILE = 'temp_graphviz_out.png'

# Attributes controlling models module.
MODELS_DIR = "models"
USE_NEURAL_SELECTOR = True
# Constants for simple NN model.
MAIN_MODEL_NAME = "main_model"
PREDICATES_MAP_NAME = "main_model_predicates.json"
# Constants for DGCNN based proving system.
GRAPH_SELECTION_MODEL_NAME = "gnn_selection_model"
GRAPH_TERMINATION_MODEL_NAME = "gnn_termination_model"
GRAPH_ENCODER_NAME = "graph_nodes_encoder"
GRAPH_SELECTOR_EXPORT_DIR = "dgcnn_selector"
GRAPH_VISUALIZE_SELECTION_PROCESS = False
# Constants for sample visualization.
CURRENT_GRAPH_FILENAME = "temp_current.jpg"
GOAL_GRAPH_FILENAME = "temp_goal.jpg"
NEXT_GRAPH_FILENAME = "temp_next.jpg"
# Constants for training samples export.
GRAPH_MODEL_TRAIN_OUTPUT_DIR = "train_gnn"

_logipy_session_name = ""  # A name of the session to be appended to the output directories.


class TheoremSelector(Enum):
    """An Enum that defines all available theorem selectors in logipy."""
    DETERMINISTIC = 1
    SIMPLE_NN = 2
    DGCNN = 3


def get_scratchfile_path(filename):
    """Returns absolute path of a file with given filename into logipy's scratchdir.

    If scratchdir doesn't exist, it is created first.
    """
    global _logipy_session_name
    current_instance_scratchdir = SCRATCHDIR_PATH / _logipy_session_name
    if not current_instance_scratchdir.exists():
        current_instance_scratchdir.mkdir()
    return current_instance_scratchdir / filename


def remove_scratchfile(filename):
    """Removes given file from logipy's scratchdir.

    If removing the file empties scratchdir, scratchdir is also removed.
    """
    if filename.is_absolute():
        absolute_scratchfile_path = filename
    else:
        absolute_scratchfile_path = SCRATCHDIR_PATH / filename

    if Path(absolute_scratchfile_path).is_file():
        absolute_scratchfile_path.unlink()
    if SCRATCHDIR_PATH.is_dir() and not any(SCRATCHDIR_PATH.iterdir()):
        SCRATCHDIR_PATH.rmdir()


def get_models_dir_path(filename=None):
    """Returns absolute path of the models directory.

    :param filename: A filename to be appended to models directory path.

    :return: A pathlib's Path object pointing to the absolute path of models' directory when
            filename is not provided. If filename is provided, Path points to the absolute path
            of a file with given filename, inside models' directory.
    """
    absolute_path = Path(__file__).absolute().parent.parent / MODELS_DIR
    if not absolute_path.exists():
        absolute_path.mkdir()
    if filename:
        absolute_path = absolute_path / filename
    return absolute_path


def set_theorem_selector(theorem_selector: TheoremSelector):
    """Sets logipy prover's theorem selector to the given one.

    :return: True if requested theorem selector set successfully. In case of an error,
            e.g. when a trained model does not exist for neural selectors, it returns
            False.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if theorem_selector is TheoremSelector.DETERMINISTIC:
        logger.info("Setting theorem prover to the deterministic one.")
        set_default_theorem_selector(BetterNextTheoremSelector())

    elif theorem_selector is TheoremSelector.SIMPLE_NN:
        logger.info("Setting theorem prover to the simple neural one.")
        set_default_theorem_selector(NeuralNextTheoremSelector())

    elif theorem_selector is TheoremSelector.DGCNN:
        selection_model, termination_model, encoder = load_gnn_models()
        if selection_model:
            logger.info("Setting theorem prover to the graph neural one.")
            set_default_theorem_selector(GraphNeuralNextTheoremSelector(
                    selection_model, termination_model, encoder,
                    export=GRAPH_VISUALIZE_SELECTION_PROCESS))
        else:
            logger.warning("Logipy: No model found under {}".format(
                    str(get_models_dir_path(GRAPH_SELECTION_MODEL_NAME))))
            return False
    return True


def is_neural_selector_enabled():
    return USE_NEURAL_SELECTOR


def enable_failure_visualization():
    """Enables visualization of proving process when a failure occurs."""
    logipy.logic.prover.full_visualization_enabled = True


def enable_proving_process_visualization():
    """Enables visualization of the whole proving process."""
    current_theorem_selector = get_default_theorem_selector()
    if isinstance(current_theorem_selector, GraphNeuralNextTheoremSelector):
        current_theorem_selector.export = True


def tearup_logipy(session_name=""):
    """Initializes logipy's modules."""
    # Generate session name.
    global _logipy_session_name
    _logipy_session_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if session_name:
        _logipy_session_name += f"_{session_name}"

    _tearup_graphs_module()
    _tearup_models_module()


def teardown_logipy():
    """Frees up resources allocated by logipy's modules."""
    _teardown_models_module()


def _tearup_graphs_module():
    logipy.graphs.timed_property_graph.graphviz_out_scratchfile_path = \
        get_scratchfile_path(GRAPHVIZ_OUT_FILE)


def _teardown_graphs_module():
    remove_scratchfile(get_scratchfile_path(GRAPHVIZ_OUT_FILE))


def _tearup_models_module():
    # Set model paths.
    logipy.models.io.main_model_path = get_models_dir_path(MAIN_MODEL_NAME)
    logipy.models.io.predicates_map_path = get_models_dir_path(PREDICATES_MAP_NAME)
    logipy.models.io.graph_selection_model_path = get_models_dir_path(GRAPH_SELECTION_MODEL_NAME)
    logipy.models.io.graph_termination_model_path = \
        get_models_dir_path(GRAPH_TERMINATION_MODEL_NAME)
    logipy.models.io.graph_encoder_path = get_models_dir_path(GRAPH_ENCODER_NAME)
    # Set scratch files paths for visualization.
    logipy.models.io.current_graph_path = get_scratchfile_path(CURRENT_GRAPH_FILENAME)
    logipy.models.io.goal_graph_path = get_scratchfile_path(GOAL_GRAPH_FILENAME)
    logipy.models.io.next_graph_path = get_scratchfile_path(NEXT_GRAPH_FILENAME)
    # Set scratch dir paths for exporting training and graph based next theorem selector data.
    logipy.models.io.graph_model_train_output_dir_path = \
        get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR)
    logipy.models.io.dgcnn_selection_process_export_path = \
        get_scratchfile_path(GRAPH_SELECTOR_EXPORT_DIR)


def _teardown_models_module():
    # Cleanup scratch files.
    remove_scratchfile(get_scratchfile_path(CURRENT_GRAPH_FILENAME))
    remove_scratchfile(get_scratchfile_path(GOAL_GRAPH_FILENAME))
    remove_scratchfile(get_scratchfile_path(NEXT_GRAPH_FILENAME))
