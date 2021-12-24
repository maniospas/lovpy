import logging
import tempfile
from enum import Enum
from pathlib import Path
from datetime import datetime

from lovpy.logic.next_theorem_selectors import set_default_theorem_selector, \
    get_default_theorem_selector, BetterNextTheoremSelector
import lovpy.graphs.timed_property_graph
import lovpy.logic.prover
from .importer import file_converter, gherkin_importer


VERSION = "0.0.1"

LOVPY_ROOT_PATH = Path(__file__).absolute().parent  # Absolute path of lovpy's installation.

LOGGER_NAME = "lovpy"

# Attributes controlling graph visualization.
GRAPHVIZ_OUT_FILE = 'temp_graphviz_out.png'

# Attributes controlling models module.
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
CURRENT_GRAPH_FILENAME = "temp_current.png"
GOAL_GRAPH_FILENAME = "temp_goal.png"
NEXT_GRAPH_FILENAME = "temp_next.png"
# Constants for training samples export.
SIMPLE_MODEL_TRAIN_OUTPUT_DIR = "train_simple"
GRAPH_MODEL_TRAIN_OUTPUT_DIR = "train_gnn"

_models_dir = "examples/models"
_lovpy_session_name = ""  # A name of the session to be appended to the output directories.
_lovpy_temp_dir = Path(tempfile.gettempdir()) / "__lovpy_temp__/"


class TheoremSelector(Enum):
    """An Enum that defines all available theorem selectors in lovpy."""
    DETERMINISTIC = 1
    SIMPLE_NN = 2
    DGCNN = 3
    HYBRID = 4


def get_scratchdir_path():
    global _lovpy_session_name
    return _lovpy_temp_dir / _lovpy_session_name


def get_scratchfile_path(filename):
    """Returns absolute path of a file with given filename into lovpy's scratchdir.

    If scratchdir doesn't exist, it is created first.
    """
    current_instance_scratchdir = get_scratchdir_path()
    if not current_instance_scratchdir.exists():
        current_instance_scratchdir.mkdir(parents=True)
    return current_instance_scratchdir / filename


def remove_scratchfile(filename):
    """Removes given file from lovpy's scratchdir.

    If removing the file empties scratchdir, scratchdir is also removed.
    """
    global _lovpy_session_name

    if filename.is_absolute():
        absolute_scratchfile_path = filename
    else:
        absolute_scratchfile_path = get_scratchfile_path(filename)

    if absolute_scratchfile_path.is_file():
        absolute_scratchfile_path.unlink()
    # Remove scratchdir if empty.
    if get_scratchdir_path() and not any(get_scratchdir_path().iterdir()):
        get_scratchdir_path().rmdir()
    if not any(_lovpy_temp_dir.iterdir()):
        _lovpy_temp_dir.rmdir()


def get_models_dir_path(filename=None):
    """Returns absolute path of the models directory.

    :param filename: A filename to be appended to models directory path.

    :return: A pathlib's Path object pointing to the absolute path of models' directory when
            filename is not provided. If filename is provided, Path points to the absolute path
            of a file with given filename, inside models' directory.
    """
    absolute_path = Path(_models_dir).absolute()
    if not absolute_path.exists():
        absolute_path.mkdir()
    if filename:
        absolute_path = absolute_path / filename
    return absolute_path


def set_theorem_selector(theorem_selector: TheoremSelector):
    """Sets lovpy prover's theorem selector to the given one.

    :return: True if requested theorem selector set successfully. In case of an error,
            e.g. when a trained model does not exist for neural selectors, it returns
            False.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if theorem_selector is TheoremSelector.DETERMINISTIC:
        logger.info("Setting theorem prover to the deterministic one.")
        set_default_theorem_selector(BetterNextTheoremSelector())

    elif theorem_selector is TheoremSelector.SIMPLE_NN:
        logger.info("Setting theorem prover to the simple neural one.")
        from .models.neural_theorem_selector import NeuralNextTheoremSelector
        from .models.simple_model import SimpleModel
        set_default_theorem_selector(NeuralNextTheoremSelector(SimpleModel.load()))

    elif theorem_selector is TheoremSelector.DGCNN or theorem_selector is TheoremSelector.HYBRID:
        from .models.gnn_model import GNNModel
        gnn_model = GNNModel.load()
        if gnn_model:
            from .models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector
            if theorem_selector is TheoremSelector.DGCNN:
                logger.info("Setting theorem prover to the graph neural one.")
                set_default_theorem_selector(GraphNeuralNextTheoremSelector(
                    gnn_model, export=GRAPH_VISUALIZE_SELECTION_PROCESS))
            else:
                logger.info("Setting theorem prover to the hybrid one.")
                selectors = [
                    BetterNextTheoremSelector(),
                    GraphNeuralNextTheoremSelector(gnn_model,
                                                   export=GRAPH_VISUALIZE_SELECTION_PROCESS)
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
    lovpy.logic.prover.full_visualization_enabled = True


def enable_proving_process_visualization():
    """Enables visualization of the whole proving process."""
    from .models.graph_neural_theorem_selector import GraphNeuralNextTheoremSelector

    current_theorem_selector = get_default_theorem_selector()
    if isinstance(current_theorem_selector, GraphNeuralNextTheoremSelector):
        current_theorem_selector.export = True


def tearup_lovpy(session_name="", temp_dir=None, models_dir=None):
    """Initializes lovpy's modules."""
    global _lovpy_session_name
    global _lovpy_temp_dir
    global _models_dir

    # Generate session name.
    _lovpy_session_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if session_name:
        _lovpy_session_name += f"_{session_name}"

    if temp_dir:
        _lovpy_temp_dir = Path(temp_dir).absolute()
        logging.getLogger(LOGGER_NAME).warning("-"*80)
        logging.getLogger(LOGGER_NAME).warning(f"Set lovpy's temp dir to {str(_lovpy_temp_dir)}")
        logging.getLogger(LOGGER_NAME).warning("-" * 80)

    if models_dir:
        _models_dir = models_dir

    _tearup_importer_module()

    try:
        import tensorflow
        _tearup_models_module()
    except ModuleNotFoundError:
        pass


def teardown_lovpy():
    """Frees up resources allocated by lovpy's modules."""
    try:
        import tensorflow
        _teardown_models_module()
    except ModuleNotFoundError:
        pass
    _teardown_importer_module()


def _tearup_models_module():
    import lovpy.models.io

    # Set model paths.
    lovpy.models.io.main_model_path = get_models_dir_path(MAIN_MODEL_NAME)
    lovpy.models.io.predicates_map_path = get_models_dir_path(PREDICATES_MAP_NAME)
    lovpy.models.io.graph_selection_model_path = get_models_dir_path(GRAPH_SELECTION_MODEL_NAME)
    lovpy.models.io.graph_termination_model_path = \
        get_models_dir_path(GRAPH_TERMINATION_MODEL_NAME)
    lovpy.models.io.graph_encoder_path = get_models_dir_path(GRAPH_ENCODER_NAME)
    # Set scratch files paths for visualization.
    lovpy.models.io.current_graph_path = get_scratchfile_path(CURRENT_GRAPH_FILENAME)
    lovpy.models.io.goal_graph_path = get_scratchfile_path(GOAL_GRAPH_FILENAME)
    lovpy.models.io.next_graph_path = get_scratchfile_path(NEXT_GRAPH_FILENAME)
    # Set scratch dir paths for exporting training and graph based next theorem selector data.
    lovpy.models.io.graph_model_train_output_dir_path = \
        get_scratchfile_path(GRAPH_MODEL_TRAIN_OUTPUT_DIR)
    lovpy.models.io.dgcnn_selection_process_export_path = \
        get_scratchfile_path(GRAPH_SELECTOR_EXPORT_DIR)


def _teardown_models_module():
    # Cleanup scratch files.
    remove_scratchfile(get_scratchfile_path(CURRENT_GRAPH_FILENAME))
    remove_scratchfile(get_scratchfile_path(GOAL_GRAPH_FILENAME))
    remove_scratchfile(get_scratchfile_path(NEXT_GRAPH_FILENAME))


def _tearup_importer_module():
    file_converter.lovpy_root_path = LOVPY_ROOT_PATH
    file_converter.convert_path()
    gherkin_importer.import_gherkin_path()


def _teardown_importer_module():
    file_converter.restore_path()
