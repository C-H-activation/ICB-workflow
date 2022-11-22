__all__ = [
    'T_RDKitAtom',
    'T_RDKitMolecule',
]

# noinspection PyPackageRequirements
import rdkit

T_RDKitAtom = rdkit.Chem.Atom
T_RDKitMolecule = rdkit.Chem.Mol