import logging
from pathlib import Path
from typing import IO, List, Text

import kallisto.atom
import kallisto.molecule
import kallisto.units

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def construct_molecule(geometry: Path) -> kallisto.molecule.Molecule:
    """Helper function to construct a Molecule."""
    try:
        with geometry.open(mode="r+") as fileObject:
            # read atoms from input file
            atoms = read_xyz(fileObject)
            # create molecule from atoms
            molecule = kallisto.molecule.Molecule(symbols=atoms)
    except FileNotFoundError:
        raise FileNotFoundError("Input file not found.")
    return molecule


def read_xyz(f: IO[Text]) -> List[kallisto.atom.Atom]:
    """Read xmol file and return list of atoms."""
    atoms = []
    lines = f.readlines()
    nat = int(lines[0])
    for line in lines[2 : nat + 2]:
        atom, x, y, z = line.split()[:4]
        symbol = atom.strip()[0].upper() + atom.strip()[1:].lower()
        position = [
            float(x) / kallisto.units.Bohr,
            float(y) / kallisto.units.Bohr,
            float(z) / kallisto.units.Bohr,
        ]
        atoms.append(kallisto.atom.Atom(symbol=symbol, position=position))
    return atoms