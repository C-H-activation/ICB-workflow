# Introduction

This repository includes an open-source implementation of the Site of Borylation (SoBo) model that predicts the regioselectivity for the iridium-catalyzed borylation.
The manuscript is currently only available on [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6362ce5aaca1981770efe240).

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

# Acknowledgements

Johan Westin restructured the package structure of the SoBo method, which has been adapted for the purpose of creating a PyPi package.
