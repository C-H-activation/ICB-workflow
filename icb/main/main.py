#!/usr/bin/env python3
import logging
import os
import re

# arguments via ArgumentParser
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas
import rich.table
import rich.text

# rich output formatter
from rich.console import Console
from rich.markdown import Markdown

from icb import constant

# helper function for structure validation, kallisto molecule and xtb gaps
from icb.helper import (
    chem_helper,
    kallisto_helper,
    rdkit_helper,
    utility,
    xtb_helper,
)

# helper functions
from icb.main.header import create_header

# results object
from icb.main.results import Results

# workflow specific tasks
from icb.task.create_barriers import create_barriers
from icb.task.create_boltzmann_weight_image import create_boltzmann_weight_image
from icb.task.create_ch import create_hydrogen_reduced_structure
from icb.task.create_graph_deltas import create_graph_deltas
from icb.task.create_ml_deltas import create_machine_learning_deltas
from icb.task.create_ts import create_transition_states
from icb.task.create_xtb_deltas import create_xtb_deltas
from icb.task.optimize_structure import perform_xtb_calculation
from icb.task.sort_ch import sort_ch_with_breadth_first_search
from icb.types import T_RDKitMolecule

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


# define the workflow
def main() -> int:
    #######################################################################
    # general setup
    #######################################################################

    # xtb optimization level
    xtb_opt_level = "lax"

    # input arguments: number of CPU's, name and SMILES
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--cpus",
        dest="num_cpu_cores",
        type=int,
        default=constant.DEFAULT_XTB_NUM_CPU_CORES_CONSTRAINT,
        help=(
            f"Number of CPU cores used during XTB calculations"
            f", default is {constant.DEFAULT_XTB_NUM_CPU_CORES_CONSTRAINT}."
        ),
    )
    parser.add_argument(
        "-i",
        "--input_file",
        dest="input_file",
        type=str,
        help=(
            f"Path to a input file."
            f" Supported formats are Daylight SMILES, and MDL MOL and SDFile."
            f" Override the --smiles option."
        ),
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        type=str,
        help=(
            f"Used as working directory and report file stem."
            f" White spaces and '{os.sep}' characters are replaced by '_' characters."
        ),
    )
    parser.add_argument("-s", "--smiles", dest="smiles", type=str)
    args = parser.parse_args()

    # sanity checks for INPUT_FILE, NAME and SMILES values
    rdkit_molecule, smiles = _validate_args(parser=parser, args=args)

    # number of CPUs for xtb optimizations
    num_cpu_cores = args.num_cpu_cores

    # set start time
    start_time = datetime.now()

    #######################################################################
    # construct a rich console and create header
    #######################################################################
    console = Console()

    name = re.sub(os.sep, "_", re.sub(r"\s", "_", args.name))

    header = create_header(smiles, name, args.num_cpu_cores)
    md = Markdown(header)
    console.print(md)

    #######################################################################
    # workflow begin
    #######################################################################

    # initialize a results object
    res = Results()

    _create_all_ch_positions(
        console=console, results=res, name=name, molecule=rdkit_molecule
    )
    _optimize_substrate_with_gfn2_xtb_alpb_thf(
        console=console,
        results=res,
        name=name,
        num_cpu_cores=num_cpu_cores,
        xtb_opt_level=xtb_opt_level,
    )
    _sort_ch_positions(console=console, results=res, name=name)
    _create_all_transition_states(console=console, results=res, name=name)
    _optimize_transition_states_with_gfn2_xtb_alpb_thf(
        console=console,
        results=res,
        name=name,
        num_cpu_cores=num_cpu_cores,
        xtb_opt_level=xtb_opt_level,
    )
    _calculate_xtb_deltas_from_energies(console=console, results=res)
    _calculate_graph_deltas_from_molecular_graph(
        console=console,
        results=res,
        molecule=rdkit_molecule,
    )

    _calculate_machine_learning_deltas_using_ts_structures(
        console=console, results=res, name=name
    )

    _calculate_barriers_in_kilo_joule_per_mol_and_weights_in_percent(
        console=console, results=res
    )

    #######################################################################
    # workflow end
    #######################################################################

    console.print()
    end_time = datetime.now()

    rich_table = _print_properties_to_console(console, res)
    console.print()
    data = _rich_table_to_pandas_dataframe(
        table=rich_table,
    )
    _create_csv_report(
        console=console,
        data=data,
        name=name,
    )
    console.print()
    _create_molecular_image_report(
        console=console,
        results=res,
        name=name,
        molecule=rdkit_molecule,
    )

    console.print()
    console.print("Overall calculation time: ", end_time - start_time)

    return 0


def _calculate_barriers_in_kilo_joule_per_mol_and_weights_in_percent(
    console: Console, results: Results
) -> None:
    #######################################################################
    # calculate barriers in kJ/mol and weights in percent
    #######################################################################
    md = Markdown(" Final barriers and Boltzmann weights")
    console.print(md)
    step_begin = datetime.now()
    res_barriers, res_weights = create_barriers(
        atoms=results.atoms,
        minimum=results.xtb_minimum,
        xtb_deltas=results.xtb_deltas,
        ml_deltas=results.ml_deltas,
        graph_deltas=results.graph_deltas,
        similarity=results.similarity,
        molecular_similarity=results.molecular_similarity,
    )
    if res_barriers is not None:
        results.barriers = res_barriers
    if res_weights is not None:
        results.weights = res_weights
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _calculate_graph_deltas_from_molecular_graph(
    console: Console, results: Results, molecule: T_RDKitMolecule
) -> None:
    #######################################################################
    # calculate graph deltas from molecular graph
    #######################################################################
    md = Markdown(" Molecular graph penalties")
    console.print(md)
    step_begin = datetime.now()
    res_graph_deltas = create_graph_deltas(molecule=molecule)
    if res_graph_deltas is not None:
        results.graph_deltas = res_graph_deltas
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _calculate_machine_learning_deltas_using_ts_structures(
    console: Console, results: Results, name: str
) -> None:
    #######################################################################
    # calculate machine-learning deltas using TS structures
    #######################################################################
    md = Markdown(" PLS-P2 machine-learning penalties")
    console.print(md)
    step_begin = datetime.now()
    res_ml_deltas, res_similarity = create_machine_learning_deltas(
        atoms=results.atoms,
        jobid=name,
        templates=results.templates,
        penalty=results.graph_deltas,
    )
    if res_ml_deltas is not None:
        results.ml_deltas = {int(k): v for k, v in res_ml_deltas.items()}
    if res_similarity is not None:
        results.similarity = {int(k): v for k, v in res_similarity.items()}
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _calculate_xtb_deltas_from_energies(console: Console, results: Results) -> None:
    #######################################################################
    # calculate xtb deltas from energies
    #######################################################################
    md = Markdown(" GFN2-xtb/ALPB(THF) minimum and penalties")
    console.print(md)
    step_begin = datetime.now()
    res_xtb_minimum, res_xtb_deltas = create_xtb_deltas(
        atoms=results.atoms,
        templates=results.templates,
        substrate=results.substrate_energy,
        xtb_energy=results.xtb_energy,
        xtb_gaps=results.xtb_gaps,
        xtb_ignore=results.xtb_ignore,
    )
    if res_xtb_minimum is not None:
        results.xtb_minimum = res_xtb_minimum
    if res_xtb_deltas is not None:
        results.xtb_deltas = res_xtb_deltas
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _create_all_ch_positions(
    console: Console,
    results: Results,
    name: str,
    molecule: T_RDKitMolecule,
) -> None:
    #######################################################################
    # create all CH positions
    #######################################################################
    md = Markdown(" Generate all C-H positions")
    console.print(md)
    step_begin = datetime.now()
    res_atoms = create_hydrogen_reduced_structure(jobid=name, molecule=molecule)
    if res_atoms is not None:
        results.atoms = res_atoms
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _create_all_transition_states(console: Console, results: Results, name: str) -> None:
    #######################################################################
    # create all transition states
    #######################################################################
    md = Markdown(" Dock compound into transition-state templates")
    console.print(md)

    # define templates of the transition states
    # results.templates = [1, 4, 7, 19, 22, 25]
    step_begin = datetime.now()
    for i, atom in enumerate(results.atoms):
        create_transition_states(
            atoms=results.atoms, index=i, jobid=name, templates=results.templates
        )
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _create_csv_report(
    console: Console,
    data: Dict[str, List[float]],
    name: str,
    decimal_char: str = constant.CSV_DECIMAL_CHAR,
    index: bool = False,
    sep_char: str = constant.CSV_SEP_CHAR,
) -> Path:
    csv_file_path = Path.cwd() / f"{name}.csv"
    dataframe = pandas.DataFrame(data)
    with csv_file_path.open(mode='wt') as f:
        dataframe.to_csv(path_or_buf=f, decimal=decimal_char, index=index, sep=sep_char)

    console.print(f"Created a CSV file, {csv_file_path.name}, with relevant data.")

    return csv_file_path


def _create_molecular_image_report(
    console: Console,
    results: Results,
    name: str,
    molecule: T_RDKitMolecule,
) -> Path:
    #######################################################################
    # Create molecular image report, with calculated Boltzmann weights
    #######################################################################
    img_file_path = Path.cwd() / f"{name}.svg"
    create_boltzmann_weight_image(
        molecule=molecule,
        weights=results.weights,
        out_file_path=img_file_path,
    )

    console.print(
        f"Created molecular image, {img_file_path.name}, with Boltzmann weights."
    )

    return img_file_path


def _optimize_substrate_with_gfn2_xtb_alpb_thf(
    console: Console, results: Results, name: str, num_cpu_cores: int, xtb_opt_level: str
) -> None:
    #######################################################################
    # optimize substrate with GFN2-xtb/ALPB(THF)
    #######################################################################
    md = Markdown(" Optimize compound using GFN2-xtb/ALP(THF)")
    console.print(md)
    step_begin = datetime.now()
    path = Path.cwd() / f"{name}"
    file_path = path / "smiles.xyz"
    xtb_args = ["--opt", xtb_opt_level, "--alpb", "thf", "-P", f"{num_cpu_cores}"]
    res_substrate_energy = perform_xtb_calculation(
        input_file_path=file_path, args=xtb_args, dir_path=path
    )
    if res_substrate_energy is not None:
        results.substrate_energy = res_substrate_energy
    step_end = datetime.now()
    console.print("time: ", step_end - step_begin)


def _optimize_transition_states_with_gfn2_xtb_alpb_thf(
    console: Console, results: Results, name: str, num_cpu_cores: int, xtb_opt_level: str
) -> None:
    #######################################################################
    # GFN2-xtb/ALPB(THF) optimizations of transition states
    #######################################################################
    md = Markdown(" Optimize transition states with GFN2-xtb/ALPB(THF)")
    console.print(md)
    step_begin = datetime.now()
    xtb_args = [
        "--opt",
        xtb_opt_level,
        "--alpb",
        "thf",
        "--input",
        "constrain.inp",
        "-P",
        f"{num_cpu_cores}",
    ]
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for template in results.templates:
            template_dir_path = base_dir_path / f"{template}"
            file_path = template_dir_path / "newstructure.xyz"
            xtb_begin = datetime.now()
            tmp = perform_xtb_calculation(
                input_file_path=Path(file_path), args=xtb_args, dir_path=template_dir_path
            )
            xtb_end = datetime.now()
            if tmp != float(14):
                print(f" - opt 1: {atom} - {template} {xtb_end - xtb_begin}")
            else:
                print(f" - opt 1: {atom} - {template} failed")

    # optimize everything and fix a dihedral
    xtb_args = [
        "--opt",
        xtb_opt_level,
        "--alpb",
        "thf",
        "--input",
        "fix.inp",
        "-P",
        f"{num_cpu_cores}",
    ]
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for template in results.templates:
            template_dir_path = base_dir_path / f"{template}"
            file_path = template_dir_path / "xtbopt.xyz"
            xtb_begin = datetime.now()
            key = f"run-{atom}-{template}-2"
            results.xtb_energy[key] = perform_xtb_calculation(
                input_file_path=Path(file_path), args=xtb_args, dir_path=template_dir_path
            )
            xtb_end = datetime.now()
            if results.xtb_energy[key] != float(14):
                print(f" - opt 2: {atom} - {template} {xtb_end - xtb_begin}")
            else:
                print(f" - opt 2: {atom} - {template} failed")

    # HL gaps 1
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for template in results.templates:
            template_dir_path = base_dir_path / f"{template}"
            file_path = template_dir_path / "xtb.out"
            key = f"run-{atom}-{template}-2"
            results.xtb_gaps[key] = xtb_helper.extract_homo_lumo_gap(file_path)

    # check if GFN2-xtb optimized structure is okay
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        results.xtb_ignore[atom] = []
        for template in results.templates:
            template_dir_path = base_dir_path / f"{template}"
            file_path = template_dir_path / "xtbopt.xyz"
            chem_helper.extract_substrate(file_path, 85, template_dir_path)
            ttpath = template_dir_path / "submolecule.xyz"
            try:
                kallisto_molecule = kallisto_helper.construct_molecule(geometry=ttpath)
                graph = kallisto_molecule.get_bonds()
                for j, node in enumerate(graph):
                    if sorted(node) == results.substrate_graph[atom][j]:
                        continue
                    else:
                        results.xtb_ignore[atom].append(template)
            except Exception:
                results.xtb_ignore[atom].append(template)

    # optimize substrate and fix catalyst (rotated)
    xtb_args = [
        "--opt",
        xtb_opt_level,
        "--alpb",
        "thf",
        "--input",
        "constrain.inp",
        "-P",
        f"{num_cpu_cores}",
    ]
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for i_template in results.templates:
            s_template = f"{i_template}r"
            template_dir_path = base_dir_path / s_template
            file_path = template_dir_path / "newstructure.xyz"
            xtb_begin = datetime.now()
            tmp = perform_xtb_calculation(
                input_file_path=Path(file_path), args=xtb_args, dir_path=template_dir_path
            )
            xtb_end = datetime.now()
            if tmp != float(14):
                print(f" - opt 3: {atom} - {s_template} {xtb_end - xtb_begin}")
            else:
                print(f" - opt 3: {atom} - {s_template} failed")

    # optimize everything and fix a dihedral (rotated)
    xtb_args = [
        "--opt",
        xtb_opt_level,
        "--alpb",
        "thf",
        "--input",
        "fix.inp",
        "-P",
        f"{num_cpu_cores}",
    ]
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for i_template in results.templates:
            s_template = f"{i_template}r"
            template_dir_path = base_dir_path / s_template
            file_path = template_dir_path / "xtbopt.xyz"
            key = f"run-{atom}-{s_template}-4"
            xtb_begin = datetime.now()
            results.xtb_energy[key] = perform_xtb_calculation(
                input_file_path=Path(file_path), args=xtb_args, dir_path=template_dir_path
            )
            xtb_end = datetime.now()
            if results.xtb_energy[key] != float(14):
                print(f" - opt 4: {atom} - {s_template} {xtb_end - xtb_begin}")
            else:
                print(f" - opt 4: {atom} - {s_template} failed")

    # HL gaps 2
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        for i_template in results.templates:
            s_template = f"{i_template}r"
            template_dir_path = base_dir_path / s_template
            key = f"run-{atom}-{s_template}-4"
            results.xtb_gaps[key] = xtb_helper.extract_homo_lumo_gap(
                template_dir_path / "xtb.out"
            )

    # check if GFN2-xtb optimized structure is okay
    kallisto_molecule = None
    for i, atom in enumerate(results.atoms):
        base_dir_path = Path.cwd() / f"{name}" / f"{atom}"
        results.xtb_ignore[atom] = []
        for i_template in results.templates:
            s_template = f"{i_template}r"
            template_dir_path = base_dir_path / s_template
            file_path = template_dir_path / "xtbopt.xyz"
            chem_helper.extract_substrate(file_path, 85, template_dir_path)
            ttpath = template_dir_path / "submolecule.xyz"
            if utility.exist_file(ttpath):
                kallisto_molecule = kallisto_helper.construct_molecule(geometry=ttpath)
            else:
                results.xtb_ignore[atom].append(i_template)
            graph = kallisto_molecule.get_bonds()
            for j, node in enumerate(graph):
                if sorted(node) == results.substrate_graph[atom][j]:
                    continue
                else:
                    results.xtb_ignore[atom].append(i_template)
    # remove duplicates from ignore list
    for i_key, i_val in results.xtb_ignore.items():
        results.xtb_ignore[i_key] = sorted(set(i_val))
    step_end = datetime.now()
    console.print("Ignore those structures: ", results.xtb_ignore)
    console.print("time: ", step_end - step_begin)


def _sort_ch_positions(console: Console, results: Results, name: str) -> None:
    #######################################################################
    # sort CH positions
    #######################################################################
    md = Markdown(" Prepare all structures for docking (sorting)")
    console.print(md)

    # save sorted substrate molecular graph
    # important to check if TS optimization retrains the structure
    step_begin = datetime.now()
    for i, atom in enumerate(results.atoms):
        results.substrate_graph[atom] = sort_ch_with_breadth_first_search(
            atoms=results.atoms, index=i, jobid=name
        )
    step_end = datetime.now()
    console.print("substrate graphs: ", results.substrate_graph)
    console.print("time: ", step_end - step_begin)


def _print_properties_to_console(console: Console, results: Results) -> rich.table.Table:
    #######################################################################
    # Print all properties
    #######################################################################

    # create a results table including all C-H positions and details
    table = rich.table.Table(show_header=True, header_style="bold magenta")
    table.add_column("[bold]C-H[/bold]", style="dim", width=12, justify="center")
    table.add_column("[bold]Similarity[/bold]", style="dim", width=16, justify="center")
    table.add_column(
        "[bold]Neighbor penalty[/bold]", style="dim", width=16, justify="center"
    )
    table.add_column(
        "[bold]PLS-PF penalty[/bold]", style="dim", width=16, justify="center"
    )
    table.add_column("[bold]Barrier[/bold]", style="dim", width=14, justify="center")
    table.add_column("[bold]Weight [%][/bold]", style="dim", width=14, justify="center")

    # fill table with C-H information (rows)
    for atom in results.atoms:
        table.add_row(
            f"{atom}",
            f"{results.similarity[atom]:.2f}",
            f"{results.graph_deltas[atom]:.2f}",
            f"{results.ml_deltas[atom]:.2f}",
            f"{results.barriers[atom]:.2f}",
            f"{results.weights[atom]:.2f}",
        )
    console.print(table)
    console.print(f"GFN2-xtb minimum energy: {results.xtb_minimum:.2f} kJ/mol")
    console.print("All energies in kJ/mol. Boltzmann weights in percent.")

    return table


def _rich_table_to_pandas_dataframe(table: rich.table.Table) -> Dict[str, List[float]]:
    """
    Convert a rich table into a Python dictionary with any rich formatting removed.

    :param table:
    """

    table_data = {
        rich.text.Text.from_markup(f'{x.header}').plain: [
            float(rich.text.Text.from_markup(f'{y}').plain) for y in x.cells
        ]
        for x in table.columns
    }
    return table_data


def _validate_args(
    parser: ArgumentParser, args: Namespace
) -> Tuple[T_RDKitMolecule, str]:
    # sanity checks for INPUT_FILE, NAME and SMILES values
    if args.name == "":
        parser.error("NAME may not be empty")

    if args.input_file is None:
        if args.smiles:
            return (
                rdkit_helper.load_smiles_as_rdkit_molecule(smiles=args.smiles),
                args.smiles,
            )
        else:
            parser.error("SMILES must be a valid SMILES string.")

    if args.input_file:
        input_file_path = Path(args.input_file)
        if not input_file_path.exists():
            parser.error("INPUT_FILE must exist.")

        valid_suffixes = [".mol", ".sdf", ".smi", ".smiles"]
        if input_file_path.suffix.lower() not in valid_suffixes:
            parser.error(f"INPUT_FILE supported formats {valid_suffixes}.")

        rdkit_molecule = rdkit_helper.load_mdl_file_as_rdkit_molecule(
            file_path=input_file_path
        )
        smiles = rdkit_helper.convert_rdkit_molecule_to_smiles(molecule=rdkit_molecule)
        return rdkit_molecule, smiles

    raise ValueError(f"CLI arguments are not valid, {args}")


if __name__ == "__main__":
    import sys

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    LOG = logging.getLogger(__name__)

    sys.exit(main())