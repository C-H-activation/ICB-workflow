import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def exist_file(path: Path) -> bool:
    """Check if file with 'path' exists."""
    return path.exists() and path.is_file()


def exist_directory(path: Path) -> bool:
    """Check if directory with 'path' exists."""
    return path.exists() and path.is_dir()


def is_exe(path: Path) -> bool:
    """Check if program is available and executable."""
    return path.exists() and path.is_file() and os.access(path, os.X_OK)