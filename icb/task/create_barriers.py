import logging
from typing import Dict, List, Tuple, cast

import numpy as np

# noinspection PyPackageRequirements

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_weights(
    atoms: List[int],
    barriers: Dict[int, float],
    percent: bool = False,
    temperature_kelvin: float = 298.15,
    ideal_gas_constant: float = 0.0083145,
) -> Dict[int, float]:
    """Create Boltzmann weights from barriers given in kJ/mol.
    temperature_kelvin: temperature is given in Kelvin (default=298.15 K).
    ideal_gas_constant: R, the ideal gas constant given in kJ/mol (default=0.0083145 J/(K mol)."""

    # initialize weights
    weights: Dict[int, float] = {}

    norm = 0
    for i, atom in enumerate(atoms):
        exp_term = np.exp(-barriers[atom] / (ideal_gas_constant * temperature_kelvin))
        weights[atom] = exp_term
        norm += exp_term

    # divide by norm to get weights
    for atom in atoms:
        weights[atom] /= norm

    if percent:
        for key, value in weights.items():
            weights[key] = value * 100

    return weights


def similarity_switch(similarity: float, steepness: float) -> float:
    """Return float depending on the entered similarity."""
    return cast(float, 1 - np.tanh(steepness * (1 - similarity)))


def create_barriers(
    atoms: List[int],
    minimum: float,
    xtb_deltas: Dict[str, float],
    ml_deltas: Dict[int, float],
    graph_deltas: Dict[int, float],
    similarity: Dict[int, float],
    molecular_similarity: float,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Create barriers for C-H activation for each position."""

    barriers = {}

    for atom in atoms:
        sim = similarity[atom]
        ml = ml_deltas[atom]
        graph = graph_deltas[atom]

        # for unhindered Hydrogen's use ML barrier
        if graph != 0.0:
            theta = similarity_switch(sim, 8.0)
            om_theta = 1 - theta

            ml *= theta
            graph *= om_theta

        mixed = ml + graph
        barriers[atom] = mixed + minimum

    #  T = 298.15 K and ideal_gas_constant = 0.0083145 J/(K mol)
    weights = create_weights(atoms=atoms, barriers=barriers, percent=True)

    return barriers, weights