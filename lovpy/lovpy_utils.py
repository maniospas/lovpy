from pathlib import Path

from . import config


def get_lovpy_system_files():
    """Returns absolute paths of all files of current lovpy installation.

    :return: A sequence of Path objects, containing the absolute paths to all files of
            current lovpy installation.
    """
    return Path(config.LOVPY_ROOT_PATH).rglob("*.py")
