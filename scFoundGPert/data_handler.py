import gc
from typing import (
    List,
    Optional,
    Union,
)

import anndata as an
import numpy as np
import scanpy as sc


def add_perturbed_cells(
    adata: an.AnnData,
    genes: List[str],
    gene_name_key: Optional[str] = None,
    perturb_value: Union[int, float] = 0,
    perturbation_key: str = 'perturbation',
    control: Optional[str] = None,
    subsample: Union[float, int] = 1.0,
    seed: int = 42,
):
    """
    Add perturbed cells to an AnnData object by knocking out specified genes.

    Parameters:
    -----------
    adata : an.AnnData
        Annotated data matrix.
    genes : list of str
        List of genes to knockout. Each element can be a gene name
        or a combination of genes like "gene1+gene2".
    gene_name_key : Optional[str], default None
        Key in adata.var that contains gene names. If None,gene names
        are assumed to be in adata.var.index.
    perturb_value : Union[int, float], default 0
        Value to set the gene expression to for the knockout.
    perturbation_key : str, default "perturbation"
        Key in adata.obs to store perturbation information.
    control : Optional[str], default None
        Control group for the perturbation to be made from. If None, all cells
        will be assumed as control and a default control group "ctrl" is created.
    subsample : Union[float, int], default 1.0
        Fraction of cells to knockout. If less than 1.0, it is treated as a fraction.
        If greater than 1, it is treated as the number of cells.
    seed : int, default 42
        Random seed for reproducibility. Used in subsampling.

    Returns:
    --------
    an.AnnData
        Annotated data matrix with added perturbed cells.
    """
    # Check if the perturbation key exists in adata.obs
    if perturbation_key not in adata.obs.keys():
        control = 'ctrl'
        adata.obs[perturbation_key] = control
        adata.obs[perturbation_key] = adata.obs[perturbation_key].astype(str)
        # Create a copy of all cells as unperturbed control cells
        adata_unperturbed = adata[adata.obs[perturbation_key] == control].copy()
    else:
        if control is None:
            # Create a copy of all cells as unperturbed control cells
            adata_unperturbed = adata.copy()
        else:
            # Create a copy of control cells
            adata_unperturbed = adata[adata.obs[perturbation_key] == control].copy()

    # Subsample the unperturbed data if needed
    if subsample < 1.0:
        sc.pp.subsample(adata_unperturbed, fraction=subsample, random_state=seed)
    elif subsample > 1:
        sc.pp.subsample(adata_unperturbed, n_obs=subsample, random_state=seed)

    # Get the list of gene names from adata.var
    adata_genes_list = (
        adata.var.index.values
        if gene_name_key is None
        else adata.var[gene_name_key].values
    )

    # Make a list of unique genes that will be perturbed
    perturb_genes = []
    for pert in genes:
        perturb_genes += pert.split('+')
    perturb_genes = list(set(perturb_genes))

    # Check if all genes to perturb are in the gene list
    for gene in perturb_genes:
        if gene not in adata_genes_list:
            raise ValueError(f'Gene {gene} not found in adata.var')

    adatas_to_add = []
    for pert in genes:
        adata_ = adata_unperturbed.copy()

        # Get positions of genes to knockout
        genes_to_knockout = pert.split('+')
        gene_positions = np.where(np.isin(adata_genes_list, genes_to_knockout))[0]
        # Set the expression of the genes to the perturbation value
        adata_.X[:, gene_positions] = perturb_value
        adata_.obs[perturbation_key] = pert

        # Make sure the perturbation is made correctly (code test)
        assert adata_.X[:, gene_positions].sum() == perturb_value * len(
            adata_.X
        ), 'Error in perturbation'

        adatas_to_add.append(adata_.copy())
        del adata_
        gc.collect()

    # Concatenate the original data with the perturbed data
    adata_with_perturbations = an.concat(
        [adata, *adatas_to_add],
        index_unique='_',
        keys=['original', *[pert for pert in genes]],
    )

    return adata_with_perturbations
