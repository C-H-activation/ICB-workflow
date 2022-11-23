# Introduction

This repository includes an open-source implementation of the Site of Borylation (SoBo) model that predicts the regioselectivity for the iridium-catalyzed borylation.
The manuscript is currently only available on [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6362ce5aaca1981770efe240).

# Environment setup

Start with setting up a new `conda` virtual environment (venv)

```bash
> conda create --name sobo python=3.9
```

and activate the created venv

```bash
> conda activate sobo
```

# Install dependencies

Get `openbabel` and `xtb` dependencies

```bash
> conda install -c conda-forge openbabel
...
> conda install -c conda-forge xtb
...
```

Then install the SoBo method

```bash
> pip install sobo
```
