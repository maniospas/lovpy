import sys
from shutil import copy2, rmtree
from pathlib import Path

from lovpy.config import LOVPY_ROOT_PATH
from . import text_converter
from . import lovpy_ignore


# TODO: Implement verified execution without modifying source files. It can be
#  implemented either by utilizing sys.settrace() to set a custom execution tracer,
#  or by using metaclasses and overriding builtins.__build_class__ method. The
#  later works only for monitoring custom defined classes.


BACKUP_FOLDER = "__lovpy_backup__"


def convert_path(root_path=""):
    """Converts all .py files under root_path to lovpy testable units.

    Original files are backed-up under BACKUP_FOLDER.
    """
    # TODO: Find out how to handle entry point file without including it to .lovpyignore.

    python_files = Path(root_path).absolute().rglob("*.py")
    python_files = list(_remove_files_to_ignore(python_files))
    _backup_files(root_path, python_files)

    for path in python_files:
        if path.is_file():
            convert_file(path)


def convert_file(path: Path):
    """Converts a single .py file to a lovpy testable unit."""
    if not path.suffix == ".py":
        raise Exception("Can only convert .py files: "+str(path))

    # Load file in memory.
    lines = list()
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                lines.append(line)
            file.close()
    except UnicodeDecodeError as e:
        print(f"An error occurred while decoding {str(path)}.", file=sys.stderr)
        raise e

    # Replace file with the converted one.
    lines = text_converter.transform_lines(lines)
    with path.open("w", encoding="utf-8") as save_file:
        for line in lines:
            save_file.write(line)
        save_file.close()


def restore_path(root_path: Path = Path.cwd()) -> None:
    """Restores original python files from backup directory.

    Backup directory is expected to be located right under `root_path`
    and named according to `BACKUP_FOLDER`.
    """
    backup_base: Path = root_path.absolute() / BACKUP_FOLDER

    for backup_path in backup_base.rglob("*.*"):
        original_path: Path = root_path / backup_path.relative_to(backup_base)
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
    """Removes from given file paths all files that should not be converted by lovpy.

    Files that should not be converted:
        -current lovpy installation
        -files and directories defined in .lovpyignore

    :param paths: An iterable of Path objects.

    :return: A generator yielding paths safe to be converted.
    """
    ignore_paths = set()
    for ignore_file in lovpy_ignore.find_lovpy_ignore():
        for pattern in lovpy_ignore.parse_lovpy_ignore(ignore_file):
            for path in Path().glob(pattern):
                ignore_paths.add(path.absolute())
    ignore_paths.add(LOVPY_ROOT_PATH)

    for p in paths:
        for ignore in ignore_paths:
            if p.is_relative_to(ignore):
                break
        else:
            yield p
