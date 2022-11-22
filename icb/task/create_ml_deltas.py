import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# noinspection PyPackageRequirements
from openbabel import pybel

# noinspection PyPackageRequirements
from rdkit import Chem

# noinspection PyPackageRequirements
from rdkit.Chem import AllChem
from scipy.spatial.distance import rogerstanimoto

from icb import utility
from icb.__init__ import __package__ as top_package_name
from icb.helper import rdkit_helper
from icb.helper.bash import copy_file, create_directory
from icb.helper.chem_helper import remove_atom_by_index
from icb.helper.ml.data import NUCLEAR_CHARGE
from icb.helper.utility import exist_file

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_machine_learning_deltas(
    atoms: List[int], jobid: str, templates: List[int], penalty: Dict[int, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Create deltas from machine-learning model (random forest) for all C-H positions."""

    # define Compound object
    class Compound(object):
        """The Compound object is used to store data."""

        def __init__(self, xyz_file_path: Optional[Path] = None) -> None:

            empty_array = np.asarray([], dtype=float)

            self.molid = float("nan")
            self.xyz_file_path = xyz_file_path
            self.subsmile: Optional[str] = None

            # information about the compound
            self.nat = float("nan")
            self.at: Dict[str, int] = {}
            self.atomtypes: List[str] = []
            self.atomtype_indices: Dict[str, List[int]] = defaultdict(list)
            self.nuclear_charges = empty_array
            self.coordinates = empty_array
            self.natoms = -1

            # representation
            self.representation = empty_array

            self.read_xyz()

        def read_xyz(self) -> None:
            """
            (Re-)initializes the Compound-object with data from a xyz-file.

            """

            if self.xyz_file_path is None:
                return

            with self.xyz_file_path.open(mode="r") as f:
                lines = f.readlines()

            self.natoms = int(lines[0])
            self.atomtypes = []
            self.nuclear_charges = np.empty(self.natoms, dtype=int)
            self.coordinates = np.empty((self.natoms, 3), dtype=float)

            for ii, line in enumerate(lines[2 : self.natoms + 2]):
                tokens = line.split()

                if len(tokens) < 4:
                    break

                self.atomtypes.append(tokens[0])
                self.atomtype_indices[tokens[0]].append(ii)
                self.nuclear_charges[ii] = NUCLEAR_CHARGE[tokens[0]]

                self.coordinates[ii] = np.asarray(tokens[1:4], dtype=float)

            self.at = dict(
                [(key, len(value)) for key, value in self.atomtype_indices.items()]
            )

        def xyz_to_smiles(self) -> str:
            """Convert xyz file to smiles."""
            mol = next(pybel.readfile("xyz", f"{self.xyz_file_path}"))
            smiles = mol.write(format="smi")
            smile: str = smiles.split()[0].strip()
            return smile

        def get_representation(self, dimension: int = 128, size: int = 4) -> None:
            """Convert xyz to smile and calculate Morgan representation for cutout."""
            # extract substructure in complex via RDKit (turn off sanitization)
            rdk_molecule = rdkit_helper.load_smiles_as_rdkit_molecule(
                self.xyz_to_smiles(), sanitize=False
            )

            # set Carbon and Iridium nuclear charge
            carbon, iridium = 6, 77

            carbon_index = -1
            for atom_n in rdk_molecule.GetAtoms():
                # get Iridium atom index
                if atom_n.GetAtomicNum() == iridium:
                    # get neighbors of Iridium
                    neighbors = [x.GetAtomicNum() for x in atom_n.GetNeighbors()]
                    # sanity check: Carbon needs to be bound to Iridium
                    if carbon not in neighbors:
                        return
                    # Carbon position in neighbor list
                    carbon_position = neighbors.index(carbon)
                    # get indices of neighbors in SMILES
                    indices = [x.GetIdx() for x in atom_n.GetNeighbors()]
                    # get on-point: Carbon atom attached to Iridium
                    carbon_index = indices[carbon_position]

            # create sub_molecule in rdkit
            env = Chem.FindAtomEnvironmentOfRadiusN(rdk_molecule, size, carbon_index)
            sub_molecule = Chem.PathToSubmol(rdk_molecule, env)
            sub_molecule.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                sub_molecule,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                catchErrors=True,
            )
            Chem.GetSymmSSSR(sub_molecule)
            # set ECFP4 fingerprint as representation
            representation = AllChem.GetMorganFingerprintAsBitVect(
                sub_molecule, 2, nBits=dimension
            )
            self.representation = np.array(representation)
            # set subsmile
            self.subsmile = Chem.MolToSmiles(sub_molecule)
            print(self.subsmile)

    # path specific
    positions = f"{jobid}"
    ml_dir_path = Path(positions) / "predict"

    # initialize base directory for ML prediction
    create_directory(path=ml_dir_path)

    # load machine-learning model
    pls_mod = "pls.mod-scikit-learn-1.1.1-python-3.10.0.joblib"
    model_file_path = utility.get_resource_path(top_package_name, f"data/{pls_mod}")
    print(f"Model path: {model_file_path}")
    if not model_file_path:
        raise FileNotFoundError("can not locate machine-learning model 'pls.mod' file")
    model = joblib.load(str(model_file_path))

    # load CSV data with binary training fingerprints
    db_file_path = utility.get_resource_path(top_package_name, "data/db.csv")
    if not db_file_path:
        raise FileNotFoundError(
            "can not locate binary training fingerprints 'db.csv' file"
        )
    df = pd.read_csv(db_file_path)
    df_list = df.values.tolist()

    # dimension of vector
    dim = 256

    # copy all structures
    deltas: Dict[str, float] = {}
    similarity: Dict[str, float] = {}
    for i, i_atom in enumerate(atoms):
        # save available templates in array
        # some templates might not be converged properly
        available: List[str] = []
        s_atom = f"{i_atom}"
        # initialize similarity
        similarity[s_atom] = 0
        # initialize delta
        deltas[s_atom] = 0
        ml_atom_dir_path = ml_dir_path / s_atom
        create_directory(path=ml_atom_dir_path)
        for i_template in templates:
            s_template = f"{i_template}"
            # create path to xmol file
            source = Path(positions) / s_atom / s_template / "xtbopt.xyz"
            destination = ml_atom_dir_path / f"{i_template}.xyz"
            # skip if optimization was not successful
            if exist_file(source):
                copy_file(source, destination)
                available.append(s_template)
            else:
                continue

        # create prediction vector
        prediction_vector = []
        similarity_list = []
        for s_template in available:
            # create path to xmol file
            destination = ml_atom_dir_path / f"{s_template}.xyz"
            # remove Hydrogen atom bound to Iridium
            # necessary since Iridium should not be hypervalent for openbabel
            remove_atom_by_index(index=19, inp=destination, out=destination)
            # create compound and representation
            cpd = Compound(xyz_file_path=destination)
            if LOG.isEnabledFor(logging.INFO):
                LOG.info(f"cpd.at={cpd.at}")

            # sanity check: compound creation
            if len(cpd.at) == 0:
                continue
            cpd.get_representation(dimension=dim)
            rep = cpd.representation
            # sanity check: representation creation
            if len(rep) == 0:
                continue
            prediction_vector.append(rep)

            # Rogers-Tanimoto similarity between vectors rep and row
            for j, row in enumerate(df_list):
                similarity_term = 1 - rogerstanimoto(rep, row)
                similarity_list.append(similarity_term)

        # extract similarity
        if similarity_list:
            similarity[s_atom] = max(similarity_list)

        # predict only when we have data
        if prediction_vector:
            # create numpy array
            np_x = np.array(prediction_vector)
            # predict deltas using randon forest
            yp = model.predict(np_x)
            # determine mean ML delta
            yp_mean = float(sum(yp) / len(yp))
            # sanity check
            # when predicted negative use graph penalty instead
            if yp_mean < -0.5:
                deltas[s_atom] = penalty[i_atom]
            else:
                # save mean delta to atomic position
                deltas[s_atom] = yp_mean
        else:
            deltas[s_atom] = 14

    return deltas, similarity