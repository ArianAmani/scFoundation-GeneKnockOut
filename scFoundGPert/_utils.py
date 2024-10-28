import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


# Function to compute metrics between two sets of embeddings
def compute_metrics(embs1, embs2):
    """
    Compute various similarity and distance metrics between two sets of embeddings.

    Parameters:
    embs1 (numpy.ndarray): First set of embeddings with shape (n_samples, n_features).
    embs2 (numpy.ndarray): Second set of embeddings with shape (n_samples, n_features).

    Returns:
    tuple: A tuple containing the following metrics:
        - cos_sim (float): Cosine similarity between
        themean embeddings of embs1 and embs2.
        - eucl_dist (float): Euclidean distance between
        the mean embeddings of embs1 and embs2.
        - wass_dist (float): Average Wasserstein distance
        computed for each feature between embs1 and embs2.
    """
    # Mean embeddings for overall similarity comparisons
    mean_embs1 = np.mean(embs1, axis=0)
    mean_embs2 = np.mean(embs2, axis=0)

    # Cosine Similarity
    cos_sim = 1 - distance.cosine(mean_embs1, mean_embs2)

    # Euclidean Distance
    eucl_dist = np.linalg.norm(mean_embs1 - mean_embs2)

    # Wasserstein Distance (computed for each feature, then averaged)
    wass_dist = np.mean(
        [wasserstein_distance(embs1[:, i], embs2[:, i]) for i in range(embs1.shape[1])]
    )

    return cos_sim, eucl_dist, wass_dist


def plot_metrics(
    adata,
    obsm_key: str,
    perturbation_key: str,
    cell_type_key: str,
    return_metrics: bool = True,
):
    """
    Plot confusion matrices for cosine similarity, Euclidean distance, and Wasserstein distance between perturbation groups for each cell type. # noqa .

    Parameters:
    adata (anndata.AnnData): Annotated data matrix.
    obsm_key (str): Key in adata.obsm containing embeddings.
    perturbation_key (str): Key in adata.obs containing perturbation information.
    cell_type_key (str): Key in adata.obs containing cell type information.
    return_metrics (bool): Whether to return metrics for each cell type.
    Default is True.

    Returns:
    dict: A dictionary containing the following metrics for each cell type:
        - cosine_sim_confusion (numpy.ndarray):
            Cosine similarity confusion matrix.
        - euclidean_dist_confusion (numpy.ndarray):
            Euclidean distance confusion matrix.
        - wasserstein_dist_confusion (numpy.ndarray):
            Wasserstein distance confusion matrix.
    """
    perturbations = list(adata.obs[perturbation_key].unique())

    metrics_for_each_ct = get_metric_confusion_matrices(
        adata,
        obsm_key,
        perturbation_key,
        cell_type_key,
    )
    # Mean metrics over cell types
    mean_cosine_sim = np.mean(
        [metrics[0] for metrics in metrics_for_each_ct.values()], axis=0
    )
    mean_euclidean_dist = np.mean(
        [metrics[1] for metrics in metrics_for_each_ct.values()], axis=0
    )
    mean_wasserstein_dist = np.mean(
        [metrics[2] for metrics in metrics_for_each_ct.values()], axis=0
    )

    # Plot the confusion matrices
    fig, axes = plt.subplots(3, 1, figsize=(8, 20))
    sns.heatmap(
        mean_cosine_sim,
        annot=False,
        ax=axes[0],
        xticklabels=perturbations,
        yticklabels=perturbations,
    )
    axes[0].set_title('Cosine Similarity')
    sns.heatmap(
        mean_euclidean_dist,
        annot=False,
        ax=axes[1],
        xticklabels=perturbations,
        yticklabels=perturbations,
    )
    axes[1].set_title('Euclidean Distance')
    sns.heatmap(
        mean_wasserstein_dist,
        annot=False,
        ax=axes[2],
        xticklabels=perturbations,
        yticklabels=perturbations,
    )
    axes[2].set_title('Wasserstein Distance')
    plt.tight_layout()
    plt.show()

    if return_metrics:
        return metrics_for_each_ct
    else:
        return None


def get_metric_confusion_matrices(
    adata,
    obsm_key: str,
    perturbation_key: str,
    cell_type_key: str,
):
    """
    Compute similarity and distance metrics for each cell type and perturbation pair.

    Parameters:
    adata (anndata.AnnData): Annotated data matrix.
    obsm_key (str): Key in adata.obsm containing embeddings.
    perturbation_key (str): Key in adata.obs containing perturbation information.
    cell_type_key (str): Key in adata.obs containing cell type information.

    Returns:
    dict: A dictionary containing the following metrics for each cell type:
        - cosine_sim_confusion (numpy.ndarray):
        Cosine similarity confusion matrix.
        - euclidean_dist_confusion (numpy.ndarray):
        Euclidean distance confusion matrix.
        - wasserstein_dist_confusion (numpy.ndarray):
        Wasserstein distance confusion matrix.
    """
    metrics_for_each_ct = {}
    for cell_type in adata.obs[cell_type_key].unique():
        adata_cell_type = adata[adata.obs[cell_type_key] == cell_type].copy()

        # Dictionary of embeddings for each perturbation group
        embeddings = {}
        for perturbation_group in adata_cell_type.obs[perturbation_key].unique():
            embeddings[perturbation_group] = adata_cell_type[
                adata_cell_type.obs[perturbation_key] == perturbation_group
            ].obsm[obsm_key]

        cosine_sim_confusion = np.zeros((len(embeddings), len(embeddings)))
        euclidean_dist_confusion = np.zeros((len(embeddings), len(embeddings)))
        wasserstein_dist_confusion = np.zeros((len(embeddings), len(embeddings)))
        # Compute metrics for each perturbation pair (create a confusion matrix)
        for i, (perturbation_group1, emb1) in enumerate(embeddings.items()):
            for j, (perturbation_group2, emb2) in enumerate(embeddings.items()):
                cos_sim, eucl_dist, wass_dist = compute_metrics(emb1, emb2)
                cosine_sim_confusion[i, j] = cos_sim
                euclidean_dist_confusion[i, j] = eucl_dist
                wasserstein_dist_confusion[i, j] = wass_dist

        metrics_for_each_ct[cell_type] = [
            cosine_sim_confusion,
            euclidean_dist_confusion,
            wasserstein_dist_confusion,
        ]

    return metrics_for_each_ct
