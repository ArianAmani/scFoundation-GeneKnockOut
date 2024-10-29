[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![python](https://img.shields.io/badge/Python-3.10-brightgreen)


# scFoundGPert: Package for using Single Cell Foundation Models for Zero-Shot Gene Knockout Predictions

## 0. Introduction & Scope

Introducing **scFoundGPert**

This package synthesizes Gene Perturbations on Single-Cell data using Foundation Models. (Currently Geneformer and scGPT)


## 1. Installation

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/ArianAmani/scFoundation-GeneKnockOut.git
cd scFoundation-GeneKnockOut
```

The relevant use-cases and source codes are located in `scFoundGPert`.
Currently, we support **python >= 3.10**.
It is recommended to install the required dependencies in a separate environment, e.g.
via `conda`.

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

## 2. Usage

Two full example notebooks are provided in the `notebooks` directory, one using `Geneformer` and the other using `scGPT` for the very same task.
You can also run them yourself in Colab using these links:
- scFound_scGPT_KO.ipynb: [![scGPT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArianAmani/scFoundation-GeneKnockOut/blob/main/notebooks/scFound_scGPT_KO.ipynb)
- scFound_Geneformer_KO.ipynb: [![scGPT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArianAmani/scFoundation-GeneKnockOut/blob/main/notebooks/scFound_Geneformer_KO.ipynb)

---
**To give an overview of how the package works:**

Import the library and the modules within:
```python
from scFoundGPert import (
    scFoundGPert,
    plot_metrics,
    CellEmbeddingClassifier,
)
from scFoundGPert.data_handler import add_perturbed_cells
```

By creating a list of genes you want to perturb, you can pass your adata to `add_perturbed_cells` and it will create synthetic perturbed cells for you:
```python
perturbs = [
    "STAT1",
    "IRF9",
    "STAT1+IRF9+IFITM3", # You can have combinations of genes to perturb in each cell
]
adata_with_perturbations = add_perturbed_cells(
    adata,
    genes=perturbs,
    gene_name_key=None,
    perturb_value=0,          # Set the perturbed genes to this value in the synthesized cells
    perturbation_key='label', # If there is already an obs column containing some kind of perturbation where you want to choose a control from and synthesizing cells from those control cells
    control='ctrl',
)
```
The `add_perturbed_cells` function will only choose cells, where with the given perturb, all the genes in the perturb combination will be changed. E.g. in the example above, for the perturbation `"STAT1+IRF9+IFITM3"`, only cells will be selected to duplicate and alter, where all three genes in `"STAT1+IRF9+IFITM3"` are non-zero in them.

Then, you can load your model using the `scFoundGPert` class. This is a wrapper for the Helical package, and currently supports `Geneformer` and `scGPT` models:
```python
# Geneformer
config_kwargs = {
    "model_name": "gf-6L-30M-i2048", # Or any other geneformer model, check helical's github for more reference
    "emb_mode": "cell",
}
model = scFoundGPert('Geneformer', device='cuda', batch_size=32, config_kwargs=config_kwargs)
```
```python
# scGPT
config_kwargs = {}
model = scFoundGPert('scGPT', device='cuda', batch_size=128, config_kwargs=config_kwargs)
```

Then, pass the adata you created to the model's tokenizer and embedder to get the embeddings:
```python
data = model.process_data(adata_with_perturbations)
adata_with_perturbations.obsm['X_embedding'] = model.get_embeddings(data)
```

You can then extract some distance/similarity metrics for the embeddings created between each condition in confusion matrix format (example output below):
```python
metrics_for_each_ct = plot_metrics(
    adata_with_perturbations,
    obsm_key='X_embedding',
    perturbation_key='label',
    cell_type_key='cell_type',
)
```
![image](https://github.com/user-attachments/assets/7d56909e-fb73-4900-91e1-6d0a90734a63)

Finally, you are able to train classifiers to classify these perturbations given the foundation model's embeddings.
4 different classifiers are available all with the very same API for your ease of use:
1. Neural Network MLP Classifier:
- Implemented using Torch to support GPU training, wrapped the Module to use the same API as other classifiers.
```python
cell_clf = CellEmbeddingClassifier(
    adata_to_classify,
    obsm_key='X_embedding',
    perturbation_key='label',
    classifier="mlp",
    classif_params={
        'device': 'cuda',
        'epochs': 100,
        'batch_size': 256,
        'dropout': 0.3,
        'n_layers': 3,
        'hidden_dim': 256,
    },
)
```
2. Decision Tree
```python
cell_clf = CellEmbeddingClassifier(
    adata_to_classify,
    obsm_key='X_embedding',
    perturbation_key='label',
    classifier="dt",
)
```
3. Random Forest
```python
cell_clf = CellEmbeddingClassifier(
    adata_to_classify,
    obsm_key='X_embedding',
    perturbation_key='label',
    classifier="rf",
)
```
4. Support Vector Machine
```python
cell_clf = CellEmbeddingClassifier(
    adata_to_classify,
    obsm_key='X_embedding',
    perturbation_key='label',
    classifier="svm",
)
```

And then you can easily train and evaluate your model on the dataset (evaluation is on a test split):
```python
cell_clf.train()

report = cell_clf.evaluate()
```

## 3. Contributing

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
