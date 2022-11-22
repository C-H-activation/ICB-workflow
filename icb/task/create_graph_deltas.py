import logging
from collections import defaultdict
from typing import Dict, List, cast

import numpy as np
from kallisto.atom import Atom
from kallisto.molecule import Molecule
from kallisto.rmsd import recursiveGetSubstructures
from kallisto.sterics import getClassicalSterimol
from kallisto.units import Bohr

# noinspection PyPackageRequirements
from rdkit import Chem

# noinspection PyPackageRequirements
from rdkit.Chem import AllChem, GetPeriodicTable

from icb.helper import rdkit_helper
from icb.helper.chem_helper import get_atom_neighbors
from icb.types import T_RDKitAtom, T_RDKitMolecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def atom_bound_to_excluded(atom: T_RDKitAtom) -> bool:
    """Returns if atoms is bounded to an excluded nuclear charge."""
    # exclude positions next to one of those nuclear charges
    # Nitrogen, Oxygen, and Sulfur
    excluded = [7, 8, 16]

    nbrs = atom.GetNeighbors()
    for nbr in nbrs:
        if nbr.GetAtomicNum() in excluded:
            return True
    return False


def create_graph_deltas(molecule: T_RDKitMolecule) -> Dict[int, float]:
    """Calculate the penalty depending on neighbors in molecular graph."""

    # copy RDKit molecule
    substrate = rdkit_helper.copy_rdkit_molecule(molecule)

    # identify aromatic Carbon atoms from pattern
    # and unpack tuple to get list of aromatic Carbon atoms
    pattern = Chem.MolFromSmarts("c")
    aromatic_carbon_positions = [
        atomPosition for (atomPosition,) in substrate.GetSubstructMatches(pattern)
    ]

    # get ring information and extract rings
    ri = substrate.GetRingInfo()
    rings = ri.AtomRings()

    # C-H count
    count = 0
    atoms = []

    # setup initial Hydrogen count and shift
    hcount, shift = -1, 0

    # Check for each atom
    for index, atom in enumerate(substrate.GetAtoms()):
        shift += 1
        hcount += atom.GetNumImplicitHs()
        hcount += atom.GetNumExplicitHs()
        for ring in rings:
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
                count += 1

    # add Hydrogen and embed
    substrate = Chem.AddHs(substrate)
    AllChem.EmbedMolecule(substrate)

    # build a kallisto molecule, get number of atoms
    # and molecular graph

    mk = kallisto_molecule_from_rdkit_molecule(substrate)
    # get coordinates of molecule
    molecule_coords = mk.get_positions()
    # get molecular graph
    molecular_graph = mk.get_bonds()
    # get atomic number of molecules
    atomic_numbers = mk.get_atomic_numbers()
    # get coordination numbers
    coordination_numbers = mk.get_cns(cntype="cov")
    # write xmol file (debug only)
    mk.writeMolecule("kallisto.xyz")
    # get number of atoms
    number_of_atoms = mk.get_number_of_atoms()

    # get atoms
    rdk_atoms = substrate.GetAtoms()

    # get nuclear charges
    atomic_numbers = [atom.GetAtomicNum() for atom in rdk_atoms]

    # define atoms that connect rings from dict
    atom_rings: Dict[int, int] = defaultdict(lambda: 0)
    for ring in rings:
        for atom in ring:
            atom_rings[atom] += 1
    # sort out atoms that are only in one ring and don't penalize position for excluded atom types
    connecting_ring_atoms: Dict[int, int] = dict(
        (k, v) for k, v in atom_rings.items() if v > 1
    )

    # get sorted rings with indices
    rings_sorted: Dict[int, List[int]] = defaultdict(lambda: [])
    for index, ring in enumerate(rings):
        rings_sorted[index] = sorted(ring)

    # iterate over all CHs
    sterical_penalties: Dict[int, Dict[int, float]] = {}

    for atom in atoms:
        # initialize
        sterical_penalties[atom] = {}

        # get RDKit atom object
        rdk_atom = rdk_atoms[atom]

        # extract neighbors of RDKit atom object
        neighbors = rdk_atom.GetNeighbors()

        # periodic table
        periodic_table = Chem.GetPeriodicTable()

        # iterate over neighbors
        for neighbor in neighbors:

            # sort out Hydrogen partners
            if neighbor.GetAtomicNum() != 1:

                # extract substructures using kallisto
                substructures = recursiveGetSubstructures(
                    number_of_atoms, molecular_graph, neighbor.GetIdx()
                )

                for substructure in substructures:
                    # sort out the ring substructures (see below)
                    # keep substituent only (has no 'atom' in substructure)
                    if atom not in substructure:
                        neighbor_covalent = [x.GetIdx() for x in neighbor.GetNeighbors()]
                        # tuple unpack single element set to int
                        (neighbor_neighbor,) = set(substructure).intersection(
                            set(neighbor_covalent)
                        )
                        # type cast to int is necessary
                        i_neighbor_neighbor = int(neighbor_neighbor)
                        # get nearest-nearest atom
                        neighbor_neighbor_atom = rdk_atoms[i_neighbor_neighbor]

                        # Only penalties for heavy atoms (not Hydrogen)
                        if rdk_atoms[i_neighbor_neighbor].GetAtomicNum() != 1:
                            # create molecule for modified Sterimol
                            subatoms = [
                                Atom(
                                    symbol=6,
                                    position=molecule_coords[neighbor.GetIdx()][:],
                                )
                            ]
                            for idx in substructure:
                                subatoms.append(
                                    Atom(
                                        symbol=atomic_numbers[idx],
                                        position=molecule_coords[idx][:],
                                    )
                                )
                            submolecule = Molecule(symbols=subatoms)
                            # write substituent (debug only)
                            # submolecule.writeMolecule("sub.xyz")
                            # Sterimol descriptors: L, Bmin, Bmax
                            l, bmin, bmax = getClassicalSterimol(submolecule, 0, 1)
                            if coordination_numbers[i_neighbor_neighbor] > 1.0:
                                # get Sterimol penalty from multivariant regression model
                                sterical_penalties[atom][
                                    neighbor.GetIdx()
                                ] = sterical_penalty(l, bmin, bmax)
                                # print(
                                #     'nbr, neighbor substituent, L, bmin, bmax, penalty',
                                #     int(nbr.GetIdx()), int(nn), l_value, bmin, bmax,
                                #     sterical_penalties[atom][nbr.GetIdx()], cns[nn], cns[nbr.GetIdx()]
                                # )
                            else:
                                sterical_penalties[atom][neighbor.GetIdx()] = 0

                            # Special cases
                            # increase penalty for atoms by sqrt(atom_mass)
                            if coordination_numbers[i_neighbor_neighbor] < 1.0:
                                print(
                                    sterical_penalties[atom][neighbor.GetIdx()],
                                    np.sqrt(neighbor_neighbor_atom.GetMass()),
                                )
                                sterical_penalties[atom][neighbor.GetIdx()] += np.sqrt(
                                    neighbor_neighbor_atom.GetMass()
                                )

    # create panalties
    graph_deltas: Dict[int, float] = {}
    for atom in atoms:
        graph_deltas[atom] = 0
        penalties = sterical_penalties[atom]
        for _, penalty in penalties.items():
            graph_deltas[atom] += penalty

    # now we construct penalties for atoms next to connected ring atoms
    atoms_connecting_rings = []
    for atom in atoms:
        sterical_penalties[atom] = {}
        # get RDKit atom object
        rdk_atom = rdk_atoms[atom]

        # extract neighbors of RDKit atom object
        neighbors = rdk_atom.GetNeighbors()

        # iterate over neighbors
        for neighbor in neighbors:

            # sort out Hydrogen partners
            if neighbor.GetAtomicNum() != 1:

                # neighbor to connecting ring atom
                __idx = [rdk_atom.GetIdx()]
                if neighbor.GetIdx() in connecting_ring_atoms:
                    atoms_connecting_rings.append(rdk_atom.GetIdx())
                    # print("in ring, connecting", nbr.GetIdx(), rdatom.GetIdx())
                    alpha, beta, gamma = get_atom_neighbors(
                        neighbor.GetIdx(), molecular_graph, [rdk_atom.GetIdx()]
                    )
                    subatoms = [
                        Atom(symbol=6, position=molecule_coords[neighbor.GetIdx()][:])
                    ]
                    for idx in alpha:
                        subatoms.extend(
                            [
                                Atom(
                                    symbol=atomic_numbers[idx],
                                    position=molecule_coords[idx][:],
                                )
                            ]
                        )
                    for idx in beta:
                        subatoms.extend(
                            [
                                Atom(
                                    symbol=atomic_numbers[idx],
                                    position=molecule_coords[idx][:],
                                )
                            ]
                        )
                    for idx in gamma:
                        subatoms.extend(
                            [
                                Atom(
                                    symbol=atomic_numbers[idx],
                                    position=molecule_coords[idx][:],
                                )
                            ]
                        )
                    submolecule = Molecule(symbols=subatoms)
                    # write submolecule (debug only)
                    # submolecule.writeMolecule("submolecule-2.xyz")
                    # Sterimol descriptors: L, Bmin, Bmax
                    l, bmin, bmax = getClassicalSterimol(submolecule, 0, 1)
                    # get a value penalty from multivariant regression model
                    sterical_penalties[atom][neighbor.GetIdx()] = sterical_penalty(
                        l, bmin, bmax
                    )

    # extend penalties with ortho to connecting ring atoms
    for atom in atoms_connecting_rings:
        penalties = sterical_penalties[atom]
        for _, penalty in penalties.items():
            graph_deltas[atom] += penalty
    return graph_deltas


def get_distance(origin: List[float], partner: List[float]) -> float:
    """Calculate the distance between origin and partner."""
    return cast(
        float,
        np.sqrt(
            np.power(origin[0] - partner[0], 2)
            + np.power(origin[1] - partner[1], 2)
            + np.power(origin[2] - partner[2], 2)
        ),
    )


def kallisto_molecule_from_rdkit_molecule(molecule: T_RDKitMolecule) -> Molecule:
    """Create a kallisto molecule from RDKit molecule.
    Args:
        molecule: RDKit molecule
    Returns:
        A kallisto molecule (kallisto.molecule.Molecule)
    """
    plain_xyz = rdkit_helper.convert_rdkit_molecule_to_plain_xyz_data(molecule=molecule)

    # setup periodic table
    pt = GetPeriodicTable()
    # create list of atoms
    atoms = []
    # create kallisto molecule
    for coord in plain_xyz:
        elem, x, y, z = coord.split()[:4]

        # convert atomic coordinates from Angstrom to Bohr
        position = [float(x) / Bohr, float(y) / Bohr, float(z) / Bohr]
        atom = Atom(symbol=pt.GetAtomicNumber(elem), position=position)
        atoms.append(atom)
    return Molecule(symbols=atoms)


def sterical_penalty(l_value: float, bmin: float, bmax: float) -> float:
    """Create penalties from Sterimol descriptors via a multivariant
    linear regression model fitted to B3-LYP/def2-TZVP penalties as obtained
    for SiMe3 probe groups in ortho position to the substituent."""

    intercept = 15.24292747
    l_s = -0.23004497
    bmin_s = -7.76699641
    bmax_s = 4.95959058
    penalty = intercept + l_s * l_value + bmin_s * bmin + bmax_s * bmax
    if penalty < 0.0:
        return float(0)
    return penalty