import logging
import tempfile
from enum import Enum
from pathlib import Path
from datetime import datetime

from logipy.logic.next_theorem_selectors import set_default_theorem_selector, \
    get_default_theorem_selector, BetterNextTheoremSelector
import logipy.graphs
import logipy.models
import logipy.logic.prover
from . import importer
from logipy.models.neural_theorem_selector import NeuralNextTheoremSelector
from logipy.models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
from logipy.models.io import load_gnn_models


LOGIPY_ROOT_PATH = Path(__file__).absolute().parent  # Absolute path of logipy's installation.

LOGGER_NAME = "logipy"

# Attributes controlling graph visualization.
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
_logipy_temp_dir = Path(tempfile.gettempdir()) / "__logipy_temp__/"


class TheoremSelector(Enum):
    """An Enum that defines all available theorem selectors in logipy."""
    DETERMINISTIC = 1
    SIMPLE_NN = 2
    DGCNN = 3
    HYBRID = 4


def get_scratchdir_path():
    global _logipy_session_name
    return _logipy_temp_dir / _logipy_session_name


def get_scratchfile_path(filename):
    """Returns absolute path of a file with given filename into logipy's scratchdir.

    If scratchdir doesn't exist, it is created first.
    """
    current_instance_scratchdir = get_scratchdir_path()
    if not current_instance_scratchdir.exists():
        current_instance_scratchdir.mkdir(parents=True)
    return current_instance_scratchdir / filename


def remove_scratchfile(filename):
    """Removes given file from logipy's scratchdir.

    If removing the file empties scratchdir, scratchdir is also removed.
    """
    global _logipy_session_name

    if filename.is_absolute():
        absolute_scratchfile_path = filename
    else:
        absolute_scratchfile_path = get_scratchfile_path(filename)

    if absolute_scratchfile_path.is_file():
        absolute_scratchfile_path.unlink()
    # Remove scratchdir if empty.
    if get_scratchdir_path() and not any(get_scratchdir_path().iterdir()):
        get_scratchdir_path().rmdir()
    if not any(_logipy_temp_dir.iterdir()):
        _logipy_temp_dir.rmdir()


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

    elif theorem_selector is TheoremSelector.DGCNN or theorem_selector is TheoremSelector.HYBRID:
        selection_model, termination_model, encoder = load_gnn_models()
        if selection_model:
            if theorem_selector is TheoremSelector.DGCNN:
                logger.info("Setting theorem prover to the graph neural one.")
                set_default_theorem_selector(GraphNeuralNextTheoremSelector(
                        selection_model, termination_model, encoder,
                        export=GRAPH_VISUALIZE_SELECTION_PROCESS))
            else:
                logger.info("Setting theorem prover to the hybrid one.")
                selectors = [
                    BetterNextTheoremSelector(),
                    GraphNeuralNextTheoremSelector(selection_model, termination_model, encoder,
                                                   export=GRAPH_VISUALIZE_SELECTION_PROCESS),
                ]
                set_default_theorem_selector(selectors)
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


def tearup_logipy(session_name="", temp_dir=None):
    """Initializes logipy's modules."""
    global _logipy_session_name
    global _logipy_temp_dir

    # Generate session name.
    _logipy_session_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if session_name:
        _logipy_session_name += f"_{session_name}"

    if temp_dir:
        _logipy_temp_dir = Path(temp_dir).absolute()
        logging.getLogger(LOGGER_NAME).warning("-"*80)
        logging.getLogger(LOGGER_NAME).warning(f"Set logipy's temp dir to {str(_logipy_temp_dir)}")
        logging.getLogger(LOGGER_NAME).warning("-" * 80)

    _tearup_importer_module()
    _tearup_graphs_module()
    _tearup_models_module()


def teardown_logipy():
    """Frees up resources allocated by logipy's modules."""
    _teardown_models_module()
    _teardown_graphs_module()
    _teardown_importer_module()


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


def _tearup_importer_module():
    importer.file_converter.logipy_root_path = LOGIPY_ROOT_PATH
    logipy.importer.file_converter.convert_path()
    logipy.importer.gherkin_importer.import_gherkin_path()


def _teardown_importer_module():
    importer.file_converter.restore_path()
