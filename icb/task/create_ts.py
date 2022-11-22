import logging
from pathlib import Path
from typing import List

import click
import kallisto.reader.strucreader as ksr
from kallisto.rmsd import exchangeSubstructure

from icb.helper.bash import change_directory, copy_file, create_directory
from icb.helper.ts.store import write_transition_state
from icb.helper.xtb_helper import write_xtb_constraints

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_transition_states(
    atoms: List[int], index: int, jobid: str, templates: List[int]
) -> None:
    """Create transition-states and exchange substrates with kallisto"""

    if LOG.isEnabledFor(logging.INFO):
        LOG.info(f"Value of ch_positions {atoms} and type {type(atoms)}")

    cwd = Path.cwd()

    # extract atom by index
    atom = atoms[index]

    # build path to file
    path = Path.cwd() / f"{jobid}" / f"{atom}"
    inp = path / f"{atom}_wo_sorted.xyz"

    # define name of newstructure
    name = "newstructure.xyz"
    for template in templates:
        # create directory for template
        destination = path / f"{template}"
        create_directory(path=destination)
        # directory name for rotated substrates
        rotated = f"{template}r"
        rotated_destination = path / rotated
        # create directory for rotated substrate
        create_directory(path=rotated_destination)
        ts_name = f"ts{template}.xyz"
        write_transition_state(n=template, name=ts_name, path=destination)
        write_transition_state(n=template, name=ts_name, path=rotated_destination)

        # copy sorted substrate to transition-state directory
        copy_file(source=inp, destination=destination)
        copy_file(source=inp, destination=rotated_destination)

        # change to transition-state directory
        change_directory(path=destination)
        # create substrate object
        substrate = ksr.constructMolecule(geometry=f"{inp}", out=click.File())
        # number of atoms in substrate
        substrate_nat = substrate.get_number_of_atoms()
        # covalent bonding partners in substrate
        substrate_bonds = substrate.get_bonds()
        if LOG.isEnabledFor(logging.INFO):
            LOG.info(f"substrate={substrate}, inp={inp}, substrate_nat={substrate_nat}")
        # create transition-state template object
        ts = ksr.constructMolecule(geometry=ts_name, out=click.File())
        # number of atoms in template
        ts_nat = ts.get_number_of_atoms()
        # covalent bonding partners in template
        ts_bonds = ts.get_bonds()
        if LOG.isEnabledFor(logging.INFO):
            LOG.info(f"ts={ts}, ts_name={ts_name}, ts_nat={ts_nat}")

        # exchange benzene with substrate
        # benzene has substructure number 2
        benzene = 2
        # Iridium has index 18 in xmol
        iridium = 18
        mol = exchangeSubstructure(
            n=ts_nat,
            center=iridium,
            subnr=benzene,
            bonds=ts_bonds,
            ref=ts,
            newsub=substrate,
            newSubBonds=substrate_bonds,
            name="newstructure",
            rotate=0,
            exclude=False,
        )
        mol.writeMolecule(name=name, path=destination)
        write_xtb_constraints(index=template)
        if LOG.isEnabledFor(logging.INFO):
            LOG.info(f"mol.get_number_of_atoms()={mol.get_number_of_atoms()}")

        # change to transition-state directory with rotation
        change_directory(path=rotated_destination)

        # exchange benzene with substrate
        # and rotate substrate by 180 degrees
        # benzene has substructure number 2
        benzene = 2
        # Iridium has index 18 in xmol
        iridium = 18
        mol = exchangeSubstructure(
            n=ts_nat,
            center=iridium,
            subnr=benzene,
            bonds=ts_bonds,
            ref=ts,
            newsub=substrate,
            newSubBonds=substrate_bonds,
            name="newstructure",
            rotate=180,
            exclude=False,
        )
        mol.writeMolecule(name=name, path=rotated_destination)
        write_xtb_constraints(index=template)

        if LOG.isEnabledFor(logging.INFO):
            LOG.info(f"mol.get_number_of_atoms()={mol.get_number_of_atoms()}")
        # get back to working directory
        change_directory(path=cwd)

        # remove kallisto generated structure in wrong directory
        name_p = Path(name)
        if name_p.exists():
            name_p.unlink()