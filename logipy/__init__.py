import sys
import logging
import atexit

from logipy.importer.file_converter import cleanup
from logipy.logic.rules import add_rule
from logipy.wrappers import LogipyPrimitive as logipy_obj
from logipy.wrappers import logipy_call as logipy_call
import logipy.importer.file_converter
import logipy.importer.gherkin_importer
import logipy.importer.exception_handler
from logipy.logic.next_theorem_selectors import set_default_theorem_selector
from logipy.models import MAIN_MODEL_PATH
from logipy.models.neural_theorem_selector import NeuralNextTheoremSelector
from . import config


LOGGER_NAME = "logipy"


logipy.importer.file_converter.convert_path()
logipy.importer.gherkin_importer.import_gherkin_path()
# TODO: Uncomment cleanup.
# atexit.register(cleanup)
sys.excepthook = logipy.importer.exception_handler.logipy_exception_handler

# Choose between neural and deterministic prover.
if config.is_neural_selector_enabled():
    logger = logging.getLogger(LOGGER_NAME)
    if MAIN_MODEL_PATH.exists():
        set_default_theorem_selector(NeuralNextTheoremSelector())
        logger.info("Setting theorem prover to the neural one.")
    else:
        logger.warning(f"Logipy: No model found under {str(MAIN_MODEL_PATH)}")
        logger.warning("\tTrain a model by executing train_model.py script.")
        logger.warning("\tFalling back to deterministic theorem prover.")
