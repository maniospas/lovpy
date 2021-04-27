import os
from pathlib import Path


SCRATCHDIR_RELATIVE_PATH = "./_temp/"
full_visualization_enabled = False

MODELS_DIR = "models"
USE_NEURAL_SELECTOR = True


def get_scratchfile_path(filename):
    if not os.path.exists(SCRATCHDIR_RELATIVE_PATH):
        os.mkdir(SCRATCHDIR_RELATIVE_PATH)
    path = SCRATCHDIR_RELATIVE_PATH + filename
    return os.path.abspath(path)


def remove_scratchfile(filename):
    os.remove(filename)
    if len(os.listdir(SCRATCHDIR_RELATIVE_PATH)) == 0:
        os.rmdir(SCRATCHDIR_RELATIVE_PATH)


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


def is_neural_selector_enabled():
    return USE_NEURAL_SELECTOR


def enable_full_visualization():
    global full_visualization_enabled
    full_visualization_enabled = True


def is_full_visualization_enabled():
    return full_visualization_enabled
