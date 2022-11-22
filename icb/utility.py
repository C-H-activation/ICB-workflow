import importlib
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Union

__all__ = [
    "change_working_directory",
    "copy_file",
    "create_non_existing_directory",
    "get_resource_path",
    "save_to_file",
]

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def change_working_directory(path: Path) -> None:
    """Change directory to path."""

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"path={path}")

    if path.exists() and path.is_dir():
        os.chdir(path)
    else:
        msg = f"the directory {path.absolute()} do not exist."
        if LOG.isEnabledFor(logging.ERROR):
            LOG.error(msg)
        raise FileNotFoundError(msg)


def copy_file(source: Path, destination: Path) -> None:
    """Copy file from source (path) to destination (path)."""

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"source={source}, destination={destination}")

    shutil.copy2(source, destination)


def create_non_existing_directory(name: str, path: Optional[Path] = None) -> Path:
    """Create directory for mentioned file_path."""

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"file_path={name}, path={path}")

    candidate_path = path / name if path else Path.cwd() / name

    if candidate_path.exists() and not candidate_path.is_dir():
        msg = f"the path {candidate_path.absolute()} exist and is not a directory."
        if LOG.isEnabledFor(logging.ERROR):
            LOG.error(msg)
        raise OSError(msg)

    if not candidate_path.exists():
        candidate_path.mkdir(parents=True)

    return candidate_path


def get_resource_path(package: str, resource: str) -> Optional[Path]:
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"package={package}, resource={resource}")

    spec = importlib.util.find_spec(package)
    if spec is None:
        return None
    mod = sys.modules.get(package)
    if mod is None or not hasattr(mod, "__file__") or not mod.__file__:
        return None

    parts: List[str] = resource.split(os.sep)
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)

    return Path(resource_name)


def save_to_file(
    data: Optional[Union[bytes, str]] = None, file_path: Optional[Path] = None
) -> int:
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(f"data={data!r}, file_path={file_path!r}")

    num_written = 0
    if data is None:
        return num_written
    if file_path is None:
        return num_written

    m = 'wt' if isinstance(data, str) else 'wb'
    with file_path.open(mode=m) as f:
        num_written = f.write(data)

    return num_written