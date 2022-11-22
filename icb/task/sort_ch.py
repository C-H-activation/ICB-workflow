import logging
from pathlib import Path
from typing import List

from icb.helper.chem_helper import sort_xyz
from icb.helper.kallisto_helper import construct_molecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def sort_ch_with_breadth_first_search(
    atoms: List[int], index: int, jobid: str
) -> List[List[int]]:
    """Sort xmol structures according to breadth first search algorithm.
    List 'atoms' contains all C-H positions for which structures should be
    sorted. The number tells us which atom should be on zeroth position when
    The structure is sorted.

    Example:
     atoms = [2, 4]

     There exist two C-H positions in the molecule (2 and 4).
     There should exist already a directory 'positions', which incorporates one
     directory for each C-H position.

     Example from above:
      ./positions/2/2.xyz and ./positions/4/4.xyz

     Those structures MUST exist since they come from the first task in the DAGs definition.
     For structure ./position/2/2.xyz this present function will take the 2nd atom of the xmol file and
     will sort all entries such that the order will match a breadth first search order with respect
     to the connectivity of all atoms in the molecule."""

    if LOG.isEnabledFor(logging.INFO):
        LOG.info(f"Value of ch_positions {atoms} and type {type(atoms)}")

    # extract atom by index
    atom = atoms[index]

    # build path to files
    path = Path.cwd() / f"{jobid}" / f"{atom}"
    inp = path / "wo.xyz"
    out = f"{atom}_wo_sorted.xyz"
    print(path, inp)
    sort_xyz(inp=inp, start_atom=atom, out=out, path=path)

    # construct kallisto Molecule from sorted structure
    fdir = path / out
    kallisto_molecule = construct_molecule(geometry=fdir)
    graph = kallisto_molecule.get_bonds()

    return [sorted(x) for x in graph]