import sys
import atexit
import os
import logging

# Configure environment of tensorflow before importing any other module.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf_installed = True
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf_installed = False
    print("-" * 80)
    print("Tensorflow is not installed - Only basic engine is available.")
    print("-" * 80)

if tf_installed and os.environ.get("LOVPY_DISABLE_GPU", 0) == "1":
    # Disable GPU usage.
    tf.config.set_visible_devices([], 'GPU')

from .monitor.wrappers import LogipyPrimitive, lovpy_call
from .models.train_model import load_or_train_model
from . import exception_handler
from . import config


LOGGER_NAME = "lovpy"

session_name = os.environ.get("LOVPY_SESSION_NAME", "")
temp_dir = os.environ.get("LOVPY_TEMP_DIR", None)
models_dir = os.environ.get("LOVPY_MODELS_DIR", None)
config.tearup_lovpy(session_name=session_name, temp_dir=temp_dir, models_dir=None)

atexit.register(config.teardown_lovpy)
if os.environ.get("LOVPY_DEV_MODE", 0) == "1":
    sys.excepthook = exception_handler.lovpy_dev_exception_handler
else:
    sys.excepthook = exception_handler.lovpy_exception_handler


# Choose between hybrid and deterministic prover.
theorem_selector = os.environ.get("LOVPY_ENGINE", "HYBRID")

if theorem_selector == "BASIC" or not config.is_neural_selector_enabled() or not tf_installed:
    config.set_theorem_selector(config.TheoremSelector.DETERMINISTIC)
elif theorem_selector == "MLP":
    config.set_theorem_selector(config.TheoremSelector.SIMPLE_NN)
elif theorem_selector == "GNN":
    config.set_theorem_selector(config.TheoremSelector.DGCNN)
elif theorem_selector == "HYBRID":
    logger = logging.getLogger(LOGGER_NAME)
    if not config.set_theorem_selector(config.TheoremSelector.HYBRID):
        config.set_theorem_selector(config.TheoremSelector.DETERMINISTIC)
        logger.warning("\tTrain a model by executing train_model.py script.")
        logger.warning("\tFalling back to deterministic theorem prover.")
else:
    config.set_theorem_selector(config.TheoremSelector.DETERMINISTIC)
