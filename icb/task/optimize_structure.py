import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Text

from icb import constant, utility

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def extract_energy(input_file_path: Path) -> float:
    """Extract energy from optimized xtb structure."""
    if input_file_path.exists():
        with input_file_path.open(mode="rt") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 1:
                    fields = line.split()
                    energy = fields[1]
                    break
        return float(energy)
    else:
        return float(14)


def perform_xtb_calculation(
    input_file_path: Path, args: List[str], dir_path: Path
) -> float:
    """Perform an xTB calculation.

    Inputs:
    input_file_path: Path to the input file (e.g., structures/input.xyz).
    args: Arguments defining the xTB calculation.
    dir_path: Name of temporary directory.

    """
    xtb_exec_path: Optional[str] = shutil.which(constant.XTB_EXEC_NAME)
    if not xtb_exec_path:
        msg = f"the {constant.XTB_EXEC_NAME} executable is not in PATH"
        if LOG.isEnabledFor(logging.CRITICAL):
            LOG.critical(msg=msg)
        raise OSError(msg)

    stderr_file_path = dir_path / constant.XTB_EXEC_STDERR_FILE_NAME
    stdout_file_path = dir_path / constant.XTB_EXEC_STDOUT_FILE_NAME

    args = [xtb_exec_path, input_file_path.name] + args
    exec_result: subprocess.CompletedProcess[Text] = subprocess.run(
        args,
        capture_output=True,
        cwd=f"{dir_path}",
        text=True,
    )

    if 0 != exec_result.returncode:
        msg = (
            f"failed during execution of {xtb_exec_path}"
            f", exit_code={exec_result.returncode}"
            f", stderr={exec_result.stderr}"
            f", stdout={exec_result.stdout}"
        )
        if LOG.isEnabledFor(logging.CRITICAL):
            LOG.critical(msg=msg)
        raise OSError(msg)

    utility.save_to_file(
        data=exec_result.stderr,
        file_path=stderr_file_path,
    )
    utility.save_to_file(
        data=exec_result.stdout,
        file_path=stdout_file_path,
    )

    return extract_energy(input_file_path=dir_path / constant.XTB_OPTIMISED_XYZ_FILE_NAME)