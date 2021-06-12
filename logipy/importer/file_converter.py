from shutil import copy2, rmtree
from pathlib import Path

from . import text_converter
from . import logipy_ignore


BACKUP_FOLDER = "__logipy_backup__"

logipy_root_path = None  # Set to current logipy installation directory at module initialization.


def convert_path(root_path=""):
    """Converts all .py files under root_path to logipy testable units.

    Original files are backed-up under BACKUP_FOLDER.
    """
    # TODO: Find out how to handle entry point file without including it to .logipyignore.

    python_files = Path(root_path).absolute().rglob("*.py")
    python_files = list(_remove_files_to_ignore(python_files))
    _backup_files(root_path, python_files)

    for path in python_files:
        if path.is_file():
            convert_file(path)


def convert_file(path: Path):
    """Converts a single .py file to a logipy testable unit."""
    if not path.suffix == ".py":
        raise Exception("Can only convert .py files: "+str(path))

    # Load file in memory.
    lines = list()
    with path.open("r") as file:
        for line in file:
            lines.append(line)
        file.close()

    # Replace file with the converted one.
    lines = text_converter.transform_lines(lines)
    with path.open("w") as save_file:
        for line in lines:
            save_file.write(line)
        save_file.close()


def restore_path(root_path=""):
    """Restores original python files from backup directory."""
    backup_base = Path(root_path).absolute() / BACKUP_FOLDER

    for backup_path in backup_base.rglob("*.*"):
        original_path = backup_path.relative_to(backup_base).absolute()
        copy2(backup_path, original_path)

    rmtree(backup_base)


def _backup_files(root_path, paths):
    """Backs-up all python files under root path."""
    backup_base = Path(root_path).absolute() / BACKUP_FOLDER
    for p in paths:
        backup_file = backup_base / p.relative_to(Path(root_path).absolute())
        if not backup_file.parent.exists():
            backup_file.parent.mkdir(parents=True)
        copy2(p, backup_file)


def _remove_files_to_ignore(paths):
    """Removes from given file paths all files that should not be converted by logipy.

    Files that should not be converted:
        -current logipy installation
        -files and directories defined in .logipyignore

    :param paths: An iterable of Path objects.

    :return: A generator yielding paths safe to be converted.
    """
    ignore_paths = set()
    for ignore_file in logipy_ignore.find_logipy_ignore():
        for pattern in logipy_ignore.parse_logipy_ignore(ignore_file):
            for path in Path().glob(pattern):
                ignore_paths.add(path.absolute())
    ignore_paths.add(logipy_root_path)

    for p in paths:
        for ignore in ignore_paths:
            if p.is_relative_to(ignore):
                break
        else:
            yield p
