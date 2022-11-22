import logging
import os
import shutil
from pathlib import Path

from icb.helper.utility import exist_directory

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def mkdir_p(path: Path) -> None:
    """Simulate bash command 'mkdir -p'."""
    try:
        path.mkdir(parents=True)
    except OSError as error:
        if path.is_dir():
            pass
        else:
            raise error


def create_directory(path: Path) -> Path:
    """Create directory with the given path."""

    if not path.exists():
        mkdir_p(path)

    return path


def copy_file(source: Path, destination: Path) -> None:
    """Copy file from source (path) to destination (path)."""
    shutil.copy2(source, destination)


def change_directory(path: Path) -> None:
    """Change directory to path."""
    if exist_directory(path):
        os.chdir(path)