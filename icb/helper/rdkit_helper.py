import logging
from pathlib import Path
from typing import Any, List, Optional, cast

# noinspection PyPackageRequirements
import rdkit

# noinspection PyPackageRequirements
import rdkit.Chem.AllChem

# noinspection PyPackageRequirements
import rdkit.Chem.Draw

# noinspection PyPackageRequirements
import rdkit.Chem.rdmolfiles
from kallisto.atom import Atom
from kallisto.molecule import Molecule
from kallisto.units import Bohr

from icb import constant
from icb.helper.bash import change_directory
from icb.types import T_RDKitMolecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def convert_rdkit_molecule_to_plain_xyz_data(molecule: T_RDKitMolecule) -> List[str]:
    # get all xyz coordinates and split into list of lines
    xyz: List[str] = rdkit.Chem.rdmolfiles.MolToXYZBlock(molecule).split("\n")
    # remove number of atoms and eventual molecule name
    xyz = xyz[2:]
    # remove empty lines from list
    xyz = [line for line in xyz if len(line.strip()) > 0]
    return xyz


def convert_rdkit_molecule_to_smiles(molecule: T_RDKitMolecule) -> str:
    return cast(str, rdkit.Chem.MolToSmiles(molecule))


def copy_rdkit_molecule(molecule: T_RDKitMolecule) -> T_RDKitMolecule:
    return rdkit.Chem.Mol(molecule)


def draw_rdkit_molecule_as_svg(molecule: T_RDKitMolecule, width: int, height: int) -> Any:
    svg = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    svg.DrawMolecule(molecule)
    svg.FinishDrawing()

    return svg


def load_mdl_file_as_rdkit_molecule(file_path: Path) -> T_RDKitMolecule:
    if file_path.suffix in constant.SUFFIX_MOL + constant.SUFFIX_SDF:
        return rdkit.Chem.MolFromMolFile(f"{file_path}")
    elif file_path.suffix in constant.SUFFIX_SDF:
        return rdkit.Chem.SDMolSupplier(f"{file_path}")
    else:
        msg = f"unsupported file type {file_path.suffix}, supported {constant.SUFFIX_MOL + constant.SUFFIX_SDF}"
        if LOG.isEnabledFor(logging.ERROR):
            LOG.error(msg)
        raise ValueError(msg)


def load_smiles_as_rdkit_molecule(smiles: str, sanitize: bool = True) -> T_RDKitMolecule:
    return rdkit.Chem.MolFromSmiles(smiles, sanitize=sanitize)


def rdkit_molecule_to_xyz(
    xyz_name: str,
    molecule: T_RDKitMolecule,
    path: Optional[Path] = None,
) -> Path:
    """Create xmol file from SMILES. Return path to xyz file."""

    _path = path if path else Path.cwd()

    source = Path.cwd()
    rdkit_molecule = copy_rdkit_molecule(molecule=molecule)

    change_directory(_path)
    molecule_with_hydrogen = rdkit.Chem.AddHs(rdkit_molecule)
    rdkit.Chem.AllChem.EmbedMolecule(molecule_with_hydrogen)
    plain_xyz = convert_rdkit_molecule_to_plain_xyz_data(molecule=molecule_with_hydrogen)

    pt = rdkit.Chem.GetPeriodicTable()

    # create atom list
    atoms = []
    for coord in plain_xyz:
        elem, x, y, z = coord.split()[:4]
        position = [float(x) / Bohr, float(y) / Bohr, float(z) / Bohr]
        atom = Atom(symbol=pt.GetAtomicNumber(elem), position=position)
        atoms.append(atom)
    # construct kallisto molecule
    kallisto_molecule = Molecule(symbols=atoms)
    # write xmol file
    kallisto_molecule.writeMolecule(name=xyz_name, path=_path)

    change_directory(source)

    return _path / xyz_name