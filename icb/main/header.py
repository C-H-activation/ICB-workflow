import logging

from icb.__version__ import __version__

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_header(smiles: str, name: str, cpus: str) -> str:
    """Create header of program."""
    return f"""
        Regioselectivity determination for the Iridium catalyzed borylation.
        Version {__version__}.

        Compound            : {smiles}
        Name                : {name}
        Number of CPU cores : {cpus}
    
    """