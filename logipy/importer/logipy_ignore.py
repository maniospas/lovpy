from pathlib import Path


def find_logipy_ignore():
    """Returns paths of all .logipyignore files found under current working directory."""
    return list(Path().absolute().rglob(".logipyignore"))


def parse_logipy_ignore(path: Path):
    """Returns all patters contained in given .logipyignore file."""
    if not path.name == ".logipyignore":
        raise RuntimeError(f"Invalid .logipyignore file: {str(path)}")

    ignore_paths = []

    with path.open("r") as f:
        for line in f:
            ignore_paths.append(line.rstrip("\n"))

    return ignore_paths
