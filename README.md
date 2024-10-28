 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)


# scFoundGPert: Package for using Single Cell Foundation Models for Zero-Shot Gene Knockout Predictions

## 0. Introduction & Scope

Introducing **scFoundGPert**

This package synthesizes Gene Perturbations on Single-Cell data using Foundation Models. (Currently Geneformer and scGPT)


## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/ArianAmani/scFoundation-GeneKnockOut.git
```

The relevant use-cases and source codes are located in `scFoundGPert`.
Currently, we support **python >= 3.10**.
It is recommended to install the required dependencies in a separate environment, e.g.
via `conda`.
A simpler alternative is a virtual environment, which is created and activated with:

```shell
conda create --name scFoundGPert python=3.10
conda activate scFoundGPert
```

Dependencies are then installed via `pip`. Currently, this repository is based on the [Helical](https://github.com/helicalAI/helical/) package. No futher dependencies are required.

```shell
pip install helical
```

The `scFoundGPert` project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of `scFoundGPert` is also installed over `pip`:

```shell
pip install -e .
```

## 2. Contributing

New ideas and improvements are always welcome. Feel free to open an issue or contribute
over a pull request.
Our repository has a few automatic checks in place that ensure a compliance with PEP8 and static
typing.
It is recommended to use `pre-commit` as a utility to adhere to the GitHub actions hooks
beforehand.
First, install the package over pip and then set a hook:
```shell
pip install pre-commit
pre-commit install
```
