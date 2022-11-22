import logging
import os
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click
import kallisto.reader.strucreader as ksr
from kallisto.data import chemical_symbols
from kallisto.units import Bohr

from icb.helper.utility import exist_file

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class MolecularGraph:
    """Define a molecular graph.

    Methods:

     add_edge - adds edge to graph
     breadth_first_search_sort - breadth first search sorting

    Variables:

     inp (str) - input xmol structure
     out (str) - output xmol structure (breadth_first_search_sort sorted)
    """

    def __init__(self, inp: Path, out: Path):
        self.inp = inp
        self.out = out
        self.graph: Dict[int, List[int]] = defaultdict(list)
        self.molecule = ksr.constructMolecule(geometry=f"{self.inp}", out=click.File())
        self.nat = self.molecule.get_number_of_atoms()
        self.at = self.molecule.get_atomic_numbers()
        self.coordinates = self.molecule.get_positions()

    # method to add an edge to graph
    def add_edge(self, u: int, v: int) -> None:
        self.graph[u].append(v)

    # method to perform breadth first search
    def breadth_first_search_sort(self, s: int) -> None:

        # mark all vertices as not visited
        visited = [False] * (len(self.graph))

        # create a queue
        q = [s]

        # mark source node as visited and enqueue it
        visited[s] = True

        seperator = os.linesep
        with self.out.open(mode="w") as outFile:
            outFile.write(f"{self.nat}" + seperator)
            outFile.write("Created by Airflow" + seperator)
            while q:

                # dequeue a vertex from queue
                s = q.pop(0)
                outFile.write(
                    f"{chemical_symbols[self.at[s]]:3}"
                    f" {self.coordinates[s][0] * Bohr:9.4f}"
                    f" {self.coordinates[s][1] * Bohr:9.4f}"
                    f" {self.coordinates[s][2] * Bohr:9.4f}"
                    f"{seperator}"
                )

                # get adjacent vertices of dequeued vertex s
                # If an adjacent has not been visited, then mark
                # it as visited and enqueue it
                for i in self.graph[s]:
                    if visited[i] is False:
                        q.append(i)
                        visited[i] = True


def sort_xyz(inp: Path, start_atom: int, out: str, path: Optional[Path] = None) -> None:
    """Sort xyz structure according to connectivity matrix from kallisto program.
    Atom declared by 'start' will be the zeroth atom in the sorted structure."""

    _path = path if path else Path.cwd()

    # we need to create a dummy click output file to serve
    # the implemented kallisto API
    output_file = click.File(mode="r", lazy=True)
    kmol = ksr.constructMolecule(geometry=f"{inp}", out=output_file)

    # get number of atoms
    nat = kmol.get_number_of_atoms()
    bonds = kmol.get_bonds()

    # initialize molecular graph
    output_path = _path / out
    g = MolecularGraph(inp, output_path)

    # travel through the graph
    for i in range(nat):
        partners = bonds[i]
        for j in partners:
            # add edge (j, i) to graph
            g.add_edge(j, i)

    # breath first search sorting:
    #  start_atom will be zeroth atom in new structure
    #  output name defined by self.out
    g.breadth_first_search_sort(start_atom)


def remove_atom_by_index(
    index: int, inp: Path, out: Optional[Path] = None, path: Optional[Path] = None
) -> None:
    """Remove atom with index from structure."""

    _out = out if out else Path("wo.xyz")
    _path = path if path else Path.cwd()

    # check if input is available
    if not exist_file(inp):
        return

    location = _path / inp
    f = location.open(mode="r")
    lines = f.readlines()
    f.close()

    # reduce atom count
    nat = int(lines[0]) - 1
    # drop number of atoms line
    lines.pop(0)
    # increase index by 1
    index += 1
    # drop atom
    lines.pop(index)

    location = _path / _out
    f = location.open(mode="w")
    s = os.linesep
    f.write(f"{nat:>5}{s}")
    for line in lines:
        f.write(line)
    f.close()


def grouper(
    n: int, iterable: Iterable[Any], fillvalue: Optional[Any] = None
) -> Iterable[Tuple[Any, ...]]:
    """Collect data into fixed-length chunks or blocks."""
    # grouper(3, "ABCDEFG", "x") --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def extract_substrate(inp: Path, start: int, path: Optional[Path] = None) -> None:
    """Cut away atoms and extract substrate with nat atoms from catalyst structure."""

    _path = path if path else Path.cwd()

    with inp.open(mode="r+") as f:
        lines = f.readlines()

    nat = int(lines[0]) - start
    xyz = lines[2 + start :]
    dirp = _path / "submolecule.xyz"
    s = os.linesep
    with dirp.open(mode="w") as f:
        f.write(str(nat) + s)
        f.write(s)
        for coord in xyz:
            f.write(coord)


def get_atom_neighbors(
    atom_idx: int,
    molecule_covalent_nbrs: List[List[int]],
    alpha: Optional[List[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Get all neighbors for atom_idx.

        Extract all covalent bonding partner (alpha), all nearest neighbours (beta),
    and all nearest-nearest neighbors (gamma) for atom with index 'atom_idx'.

    Args:
    atom_idx: index of atom to extract neighbors for
    molecule_covalent_list: list of covalent partner of each atom

    Returns:
    alpha: list of all covalent bonding atom indices of atom_idx
    beta: list of nearest neighbor atom indices of atom_idx
    gamma: list of nearest-nearest neighbor atom indices of atom_idx

    """
    # extract alpha neighbors
    effective_alpha: List[int] = alpha if alpha else molecule_covalent_nbrs[atom_idx]

    # extract beta neighbors
    beta: List[int] = []
    for _, a_alpha in enumerate(effective_alpha):
        b: List[int] = molecule_covalent_nbrs[a_alpha]
        diff = list({atom_idx} ^ set(b))
        if len(diff) > 0:
            beta.extend(diff)

    # extract gamma neighbors
    gamma: List[int] = []
    for _, a_beta in enumerate(beta):
        c: List[int] = molecule_covalent_nbrs[a_beta]
        inter = list(set(effective_alpha).intersection(set(c)))
        diff = list(set(inter) ^ set(c))
        gamma.extend(diff)
    gamma = list(dict.fromkeys(gamma))
    return effective_alpha, beta, gamma