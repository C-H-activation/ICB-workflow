import logging
from pathlib import Path
from typing import Dict

from icb.helper import rdkit_helper
from icb.types import T_RDKitMolecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_boltzmann_weight_image(
    molecule: T_RDKitMolecule,
    weights: Dict[int, float],
    out_file_path: Path,
    print_weights: bool = True,
    threshold: int = 5,
) -> None:
    """Create a molecular SVG image with RDKit where the calculated
    Boltzmann weights are visible at the C-H positions."""

    # copy RDKit molecule
    molecule_copy = rdkit_helper.copy_rdkit_molecule(molecule=molecule)

    # set weights
    for atom in molecule_copy.GetAtoms():
        idx = atom.GetIdx()
        try:
            if weights[idx] >= threshold:
                lbl = "%.0f" % (weights[idx])
                if print_weights:
                    atom.SetProp("atomNote", lbl)
                else:
                    atom.SetProp("atomNote", str(idx))
        except Exception:
            atom.SetProp("atomNote", "")

    # high quality SVG
    d2d = rdkit_helper.draw_rdkit_molecule_as_svg(
        molecule=molecule_copy, width=400, height=400
    )
    with out_file_path.open(mode="w") as f:
        f.write(d2d.GetDrawingText())