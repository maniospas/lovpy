from pathlib import Path

from . import config


def get_logipy_system_files():
    """Returns absolute paths of all files of current logipy installation.

    :return: A sequence of Path objects, containing the absolute paths to all files of
            current logipy installation.
    """
    return Path(config.LOGIPY_ROOT_PATH).rglob("*.py")
