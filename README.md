# Introduction

This repository includes an open-source implementation of the Site of Borylation (SoBo) model that predicts the regioselectivity for the iridium-catalyzed borylation.
The manuscript is available on [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6362ce5aaca1981770efe240) and [JACS](https://doi.org/10.1021/jacs.3c04986).

# Environment setup

Start with setting up a new `conda` virtual environment (venv) (see [here](https://docs.conda.io/en/latest/miniconda.html) how to setup Miniconda)

```bash
> conda create --name sobo python=3.9
```

and activate the created venv

```bash
> conda activate sobo
```

# Install dependencies

Get `openbabel` (3.1.0) dependency

```bash
> conda install -c conda-forge openbabel
```

Get `xtb` dependency (6.4.0) by downloading a precompiled binary from [GitHub](https://github.com/grimme-lab/xtb/releases), by compiling the [source code](https://xtb-docs.readthedocs.io/en/latest/development.html) yourself, or by a `conda` installation (only fixed versions available)

```bash
> conda install -c conda-forge xtb
```

Then install the SoBo (0.2.0) method via pip

```bash
> pip install sobo
```

The typical installation time on a standard desktop computer is in the range of minutes.
Several examples are available with expected output inside the `examples` directory.

# Run predictions

Once the model is setup (check reference outputs in [examples](/examples)) you can perform predictions by entering a [SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) string and a name for the system of interest.

```bash
> sobo --smiles <SMILES> --name <NAME>
```

You can get additional information via

```bash
> sobo --help
```


# Acknowledgements

Johan Westin restructured the package structure of the SoBo method, which has been adapted for the purpose of creating a [PyPi package](https://pypi.org/project/sobo/).

Reference
---------

If you want to apply or reference this work please always cite:

Caldeweyher, Eike and Elkin, Masha and Gheibi, Golsa and Johansson, Magnus and Sköld, Christian and Norrby, Per-Ola and Hartwig, John F., J. Am. Chem. Soc, *2023*. DOI: [10.1021/jacs.3c04986](https://doi.org/10.1021/jacs.3c04986)

```
@article{doi:10.1021/jacs.3c04986,
author = {Caldeweyher, E. and Elkin, M. and Gheibi, G. and Johansson, M. J. and Sköld, C. and Norrby, P-O and Hartwig, J. F.},
title = {Hybrid Machine Learning Approach to Predict the Site Selectivity of Iridium-Catalyzed Arene Borylation},
journal = {J. Am. Chem. Soc.},
year = {2023},
doi = {10.1021/jacs.3c04986},
}
```


