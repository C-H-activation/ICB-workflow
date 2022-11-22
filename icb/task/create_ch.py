import logging
from pathlib import Path
from typing import Any, List

# noinspection PyPackageRequirements
from rdkit import Chem

from icb.helper import rdkit_helper
from icb.helper.bash import copy_file, create_directory
from icb.helper.chem_helper import remove_atom_by_index
from icb.types import T_RDKitAtom, T_RDKitMolecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_hydrogen_reduced_structure(jobid: str, molecule: T_RDKitMolecule) -> List[int]:
    """Identify all aromatic C-H positions from SMILES and
    cut away the Hydrogen atom attached to aromatic Carbon.
    Write substructures to directories.
    Return list of aromatic C-H atoms."""

    # path specific
    positions = f"{jobid}"

    # initialize base directory for method
    create_directory(path=Path(positions))

    # create substrate from input
    substrate = rdkit_helper.copy_rdkit_molecule(molecule=molecule)

    rdatoms = substrate.GetAtoms()

    # identify aromatic Carbon atoms from pattern
    # and unpack tuple to get list of aromatic Carbon atoms
    pattern: T_RDKitMolecule = Chem.MolFromSmarts("c")
    aromatic_carbon_positions = [
        atomPosition for (atomPosition,) in substrate.GetSubstructMatches(pattern)
    ]

    # get ring information and extract rings
    ri = substrate.GetRingInfo()
    rings = ri.AtomRings()

    # C-H candidates
    atoms: List[int] = []
    hydrogen_neighbors: List[int] = []

    # setup initial Hydrogen count and shift
    hcount = -1
    shift = 0

    nitrogen_nuclear_charge = 7
    # Check for each atom
    for index, atom in enumerate(substrate.GetAtoms()):
        shift += 1
        hcount += atom.GetNumImplicitHs()
        hcount += atom.GetNumExplicitHs()
        for ring in rings:
            #################################################
            # SPECIAL CASE for Nitrogen
            #################################################
            # sort out ortho to Nitrogen in six-membered ring
            if len(ring) == 6:
                if check_for_neighbor_atom(
                    atom=atom, nuclear_charge=nitrogen_nuclear_charge
                ):
                    continue
            # sort out ortho to Nitrogen in five-membered ring
            # ONLY when more than one Nitrogen is in ring
            elif len(ring) == 5:
                at = [rdatoms[atom].GetAtomicNum() for atom in ring]
                if check_for_atom_duplicates(at, nitrogen_nuclear_charge):
                    if check_for_neighbor_atom(
                        atom=atom, nuclear_charge=nitrogen_nuclear_charge
                    ):
                        continue
            #################################################
            # extract Carbon atoms that
            # 1) belong to a ring system
            # 2) are aromatic
            # 3) occur only once (exclude doubles)
            # 4) have ONE C-H bond.
            if (
                (index in ring)
                and (index in aromatic_carbon_positions)
                and (index not in atoms)
                and (atom.GetNumImplicitHs() == 1)
            ):
                atoms.append(index)
                hydrogen_neighbors.append(hcount)

    # Hydrogen atoms are added to the end of the xmol file
    # therefore shift all indices with respect to the number
    # of atoms that are not Hydrogen
    hydrogen_neighbors = [x + shift for x in hydrogen_neighbors]

    # prepare xmol structures
    xyz_name = "smiles.xyz"
    dirp = Path.cwd() / positions
    rdkit_helper.rdkit_molecule_to_xyz(xyz_name=xyz_name, molecule=molecule, path=dirp)
    # combine path and file name
    source = dirp / xyz_name
    for i, k in enumerate(atoms):
        path = dirp / f"{k}"
        create_directory(path=path)
        destination = path / f"{k}.xyz"
        copy_file(source, destination)
        # remove Hydrogen atom connected to aromatic Carbon
        remove_atom_by_index(index=hydrogen_neighbors[i], inp=destination, path=path)

    return atoms


def check_for_atom_duplicates(atoms: List[int], nuclear_charge: Any) -> bool:
    """Check for duplicates of nuclear_charges in atoms list."""
    # check if nuclear charge exists at all
    if nuclear_charge not in atoms:
        return False
    # set up a hash table to count occurrence
    hash_dict = {}
    for elem in atoms:
        if elem not in hash_dict:
            hash_dict[elem] = 1
        else:
            hash_dict[elem] += 1
    # nuclear charge is more than once in atom list
    if hash_dict[nuclear_charge] > 1:
        return True

    return False


def check_for_neighbor_atom(atom: T_RDKitAtom, nuclear_charge: float) -> bool:
    """Check if atom is next to a specific Atom (True) or not (False)."""

    # get all neighbors
    nbrs = atom.GetNeighbors()
    # iterate over neighbors

    for nbr in nbrs:
        if nbr.GetAtomicNum() == nuclear_charge:
            return True

    return False