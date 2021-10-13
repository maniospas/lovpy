from pathlib import Path


def find_lovpy_ignore():
    """Returns paths of all .lovpyignore files found under current working directory."""
    return list(Path().absolute().rglob(".lovpyignore"))


def parse_lovpy_ignore(path: Path):
    """Returns all patters contained in given .lovpyignore file."""
    if not path.name == ".lovpyignore":
        raise RuntimeError(f"Invalid .lovpyignore file: {str(path)}")

    ignore_paths = []

    with path.open("r") as f:
        for line in f:
            ignore_paths.append(line.rstrip("\n"))

    return ignore_paths
