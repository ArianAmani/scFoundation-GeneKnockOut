from typing import Literal, Optional

import anndata as an
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
    perturbations = sorted(list(adata.obs[perturbation_key].unique()))

    metrics_for_each_ct = get_metric_confusion_matrices(
        adata,
        obsm_key,
        perturbation_key,
        cell_type_key,
    )
    # Mean metrics over cell types
    mean_cosine_sim = np.nanmean(
        [metrics[0] for metrics in metrics_for_each_ct.values()], axis=0
    )
    mean_euclidean_dist = np.nanmean(
        [metrics[1] for metrics in metrics_for_each_ct.values()], axis=0
    )
    mean_wasserstein_dist = np.nanmean(
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
    perturbations = sorted(list(adata.obs[perturbation_key].unique()))
    perturbation_dict = {pert: i for i, pert in enumerate(perturbations)}

    metrics_for_each_ct = {}
    for cell_type in adata.obs[cell_type_key].unique():
        adata_cell_type = adata[adata.obs[cell_type_key] == cell_type].copy()

        # Dictionary of embeddings for each perturbation group
        embeddings = {}
        for perturbation_group in adata_cell_type.obs[perturbation_key].unique():
            embeddings[perturbation_group] = adata_cell_type[
                adata_cell_type.obs[perturbation_key] == perturbation_group
            ].obsm[obsm_key]

        cosine_sim_confusion = np.zeros(
            (len(perturbation_dict), len(perturbation_dict))
        )
        euclidean_dist_confusion = np.zeros(
            (len(perturbation_dict), len(perturbation_dict))
        )
        wasserstein_dist_confusion = np.zeros(
            (len(perturbation_dict), len(perturbation_dict))
        )

        # Compute metrics for each perturbation pair (create a confusion matrix)
        for perturbation_group1, emb1 in embeddings.items():
            for perturbation_group2, emb2 in embeddings.items():
                i, j = (
                    perturbation_dict[perturbation_group1],
                    perturbation_dict[perturbation_group2],
                )

                cos_sim, eucl_dist, wass_dist = compute_metrics(emb1, emb2)
                cosine_sim_confusion[i, j] = cos_sim
                euclidean_dist_confusion[i, j] = eucl_dist
                wasserstein_dist_confusion[i, j] = wass_dist

        # Set non-existant ones to NaN
        for pert in perturbation_dict.keys():
            if pert not in embeddings.keys():
                i = perturbation_dict[pert]
                cosine_sim_confusion[i, :] = np.nan
                cosine_sim_confusion[:, i] = np.nan
                euclidean_dist_confusion[i, :] = np.nan
                euclidean_dist_confusion[:, i] = np.nan
                wasserstein_dist_confusion[i, :] = np.nan
                wasserstein_dist_confusion[:, i] = np.nan

        metrics_for_each_ct[cell_type] = [
            cosine_sim_confusion,
            euclidean_dist_confusion,
            wasserstein_dist_confusion,
        ]

    return metrics_for_each_ct


class CellEmbeddingClassifier:
    def __init__(
        self,
        classifier: Literal['mlp', 'dt', 'rf', 'svm'] = 'mlp',
        classif_params: dict = {},
    ):
        """
        Initialize the cell embedding classifier.

        Parameters:
        embedding_dim (int): Dimension of the cell embeddings.
        n_classes (int): Number of classes to classify.
        classifier (str): Type of classifier to use. Options are 'mlp', 'dt', 'rf', and 'svm'. # noqa
        Each corresponds to a different classifier:
        - 'mlp': Multi-layer Perceptron.
        - 'dt': Decision Tree.
        - 'rf': Random Forest.
        - 'svm': Support Vector Machine.
        Default is 'mlp'.
        classif_params (dict): Parameters for the classifier.
        Default is an empty dictionary.
        """
        if classifier == 'mlp':
            from sklearn.neural_network import MLPClassifier

            params = {
                'hidden_layer_sizes': (
                    128,
                    64,
                ),
                'alpha': 1e-3,
            }
            params.update(classif_params)
            self.classifier = MLPClassifier(**params)
        elif classifier == 'dt':  # Decision Tree
            from sklearn.tree import DecisionTreeClassifier

            params = {}
            params.update(classif_params)
            self.classifier = DecisionTreeClassifier(**params)
        elif classifier == 'rf':  # Random Forest
            from sklearn.ensemble import RandomForestClassifier

            params = {
                'n_estimators': 100,
            }
            params.update(classif_params)
            self.classifier = RandomForestClassifier(**params)
        elif classifier == 'svm':  # Support Vector Machine
            from sklearn.svm import SVC

            params = {}
            params.update(classif_params)
            self.classifier = SVC(**params)

        else:
            raise ValueError(
                "Invalid classifier. Choose from 'mlp', 'dt', 'rf', or 'svm'."
            )

        self.anndata: Optional[an.AnnData] = None
        self.obsm_key: Optional[str] = None
        self.perturbation_key: Optional[str] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def setup(
        self,
        anndata: an.AnnData,
        obsm_key: str,
        perturbation_key: str,
        test_size: float = 0.2,
    ):
        """
        Setup the dataset for training and testing the classifier.

        Parameters:
        anndata (anndata.AnnData): Annotated data matrix.
        obsm_key (str): Key in adata.obsm containing embeddings.
        perturbation_key (str): Key in adata.obs containing perturbation information.
        test_size (float): Fraction of the data to use for testing. Default is 0.2.
        """
        from sklearn.model_selection import train_test_split

        self.anndata = anndata
        self.obsm_key = obsm_key
        self.perturbation_key = perturbation_key

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.anndata.obsm[self.obsm_key],
            self.anndata.obs[self.perturbation_key].values,
            test_size=test_size,
        )

    def train(self):
        """
        Train the classifier on the training data.
        """
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the classifier on the testing data.
        Computes a full classification report.

        Returns:
        --------
        str: Classification report.
        """
        from sklearn.metrics import classification_report

        y_pred = self.classifier.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)

        return report
