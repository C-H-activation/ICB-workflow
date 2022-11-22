import logging
import statistics
from typing import Dict, List, Tuple, cast

import numpy as np

from icb.helper.xtb_helper import calculate_barrier

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_sterical_penalty(max_prox: float, prox: float) -> float:
    """Input a proximity shell as calculated from kallisto and return a sterical hindrance value in kJ/mol."""

    # logger = logging.getLogger("create_barriers")
    if max_prox < 12:
        max_prox = 12
    if LOG.isEnabledFor(logging.INFO):
        LOG.info(f"maxProx={max_prox}")
    return cast(float, 0.5 * (12 - 12 * np.tanh((max_prox - prox) / 2)))


def create_xtb_deltas(
    atoms: List[int],
    templates: List[int],
    substrate: float,
    xtb_energy: Dict[str, float],
    xtb_gaps: Dict[str, float],
    xtb_ignore: Dict[int, List[int]],
) -> Tuple[float, Dict[str, float]]:
    """Create deltas for all C-H positions using GFN2-xtb energies."""

    # catalyst = -120.568193063973 # opt-tight GFN2-xtb/ALPB(THF) energy in Hartree: catalyst
    catalyst = (
        -120.566060481458
    )  # opt-lax   GFN2-xtb/ALPB(THF) energy in Hartree: catalyst

    # errors for skipping certain templates
    errors = [
        "grep: opt2.xyz: No such file or directory",
    ]

    # dictionary for all barriers
    barriers: Dict[str, float] = {}
    barrier_list: Dict[str, List[float]] = {}
    for i, i_atom in enumerate(atoms):
        s_atom = f"{i_atom}"
        barrier_list[s_atom] = []

        tasks = []
        # start with normal substrate
        for i_template in templates:
            # if template in ignore cycle
            if not xtb_ignore[i_atom] and i_template in xtb_ignore[i_atom]:
                continue
            task = f"run-{i_atom}-{i_template}-2"
            tasks.append(task)
        for j, task in enumerate(tasks):
            # xtb optimization was not successful
            if xtb_energy[task] == 14:
                barrier = 14.0
            # HOMO-LUMO gap too small -> sort out
            if xtb_gaps[task] <= 0.1:
                continue
            barrier = calculate_barrier(
                ts=float(xtb_energy[task]),
                cat=float(catalyst),
                sub=float(substrate),
            )
            barrier_list[s_atom].append(barrier)

        tasks = []
        # continue with rotated substrate
        for i_template in templates:
            template_r = f"{i_template}r"
            # if template in ignore cycle
            if not xtb_ignore[i_atom] and template_r in xtb_ignore[i_atom]:
                continue
            task = f"run-{i_atom}-{template_r}-4"
            tasks.append(task)
        for j, task in enumerate(tasks):
            # xtb optimization was not successful
            if xtb_energy[task] == 14:
                barrier = 14.0
            # HOMO-LUMO gap too small -> sort out
            if xtb_gaps[task] <= 0.1:
                continue
            barrier = calculate_barrier(
                ts=float(xtb_energy[task]),
                cat=float(catalyst),
                sub=float(substrate),
            )
            barrier_list[s_atom].append(barrier)

    # create median barrier
    median: Dict[str, float] = {}
    for i, i_atom in enumerate(atoms):
        s_atom = f"{i_atom}"
        tmp = [b for b in barrier_list[s_atom] if b > 0]  # exclude negative barriers
        if len(tmp) == 0:
            median[s_atom] = 120
        else:
            median[s_atom] = statistics.median(tmp)

    # 14 kJ/mol represent a 300:1 Boltzmann weight
    sigma = 14

    # sort barriers out that are out of limit
    for i, i_atom in enumerate(atoms):
        s_atom = f"{i_atom}"
        # create limit from median barrier
        low_limit = median[s_atom] - sigma
        high_limit = median[s_atom] + 6 * sigma
        tmp = [
            b for b in barrier_list[s_atom] if low_limit <= b <= high_limit
        ]  # sort out barriers
        # in case there is nothing left :(
        if len(tmp) == 0:
            mean: float = 120
        else:
            mean = sum(tmp) / len(tmp)
        barriers[s_atom] = mean

    # get minimum barrier
    argmin = min(barriers, key=lambda x: barriers.get(x, float("nan")))
    minimum = barriers[argmin]

    # create GFN2-xtb deltas
    deltas = {k: v - minimum for k, v in barriers.items()}
    return minimum, deltas