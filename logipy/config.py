import os

SCRATCHDIR_RELATIVE_PATH = "./_temp/"
MODELS_DIR = "./models/"


def get_scratchfile_path(filename):
    if not os.path.exists(SCRATCHDIR_RELATIVE_PATH):
        os.mkdir(SCRATCHDIR_RELATIVE_PATH)
    path = SCRATCHDIR_RELATIVE_PATH + filename
    return os.path.abspath(path)


def remove_scratchfile(filename):
    os.remove(filename)
    if len(os.listdir(SCRATCHDIR_RELATIVE_PATH)) == 0:
        os.rmdir(SCRATCHDIR_RELATIVE_PATH)


def get_models_dir():
    abs_path = os.path.abspath(MODELS_DIR)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
    return abs_path
