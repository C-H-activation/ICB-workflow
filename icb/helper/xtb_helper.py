import logging
import os
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# define global line seperator
LINE_SEP = os.linesep


def fix_atoms(atoms: List[int], file_path: Optional[Path] = None) -> None:
    """Write a xtb fixing file for passed in atoms."""

    _file_path = file_path if file_path else Path("fix.inp")

    # build string
    string = ", "
    string = string.join(map(str, atoms))

    with _file_path.open(mode="wt") as f:
        f.write(f"$fix{LINE_SEP} atoms: {string}{LINE_SEP}$end{LINE_SEP}")


def write_xtb_constraints(index: int) -> None:
    """Write xtb constraints depending on transition-state template index."""

    if index in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # fix dihedral: B(63)-Ir(19)-H(20)-C(86)
        fix_atoms(atoms=[63, 19, 20, 86])
        return

    # fix dihedral: B(42)-Ir(19)-H(20)-C(86)
    fix_atoms(atoms=[42, 19, 20, 86])
    return


def calculate_barrier(ts: float, cat: float, sub: float, conv: float = 2625.5) -> float:
    """Calculate all barriers for C-H activation using energies for
    ts: transition-state energy in Hartree
    cat: catalyst energy in Hartree
    sub: substrate energy in Hartree.
    Returns by default a barrier in kJ/mol."""

    # default conversion factor for  Hartree -> kJ/mol
    return (ts - cat - sub) * conv


def extract_homo_lumo_gap(inp: Path) -> float:
    """Extract the homo-lumo gap from a xtb output."""
    try:
        with inp.open(mode="r+") as f:
            contents = f.readlines()
        target = "HOMO-LUMO GAP"
        targets = [ss for ss in contents if target in LINE_SEP]
        # return homo-lumo gap
        return float(targets[0].split()[3])
    except Exception:
        return float(0)