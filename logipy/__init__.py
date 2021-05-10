import sys
import logging
import atexit
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from logipy.importer.file_converter import cleanup
from logipy.logic.rules import add_rule
from logipy.wrappers import LogipyPrimitive as logipy_obj
from logipy.wrappers import logipy_call as logipy_call
import logipy.importer.file_converter
import logipy.importer.gherkin_importer
import logipy.importer.exception_handler

from . import config


LOGGER_NAME = "logipy"


logipy.importer.file_converter.convert_path()
logipy.importer.gherkin_importer.import_gherkin_path()
# TODO: Uncomment cleanup.
# atexit.register(cleanup)
atexit.register(config.teardown_logipy)
sys.excepthook = logipy.importer.exception_handler.logipy_exception_handler

config.tearup_logipy()

# Choose between neural and deterministic prover.
if config.is_neural_selector_enabled():
    logger = logging.getLogger(LOGGER_NAME)
    if not config.set_theorem_selector(config.TheoremSelector.DGCNN):
        config.set_theorem_selector(config.TheoremSelector.DETERMINISTIC)
        logger.warning("\tTrain a model by executing train_model.py script.")
        logger.warning("\tFalling back to deterministic theorem prover.")
