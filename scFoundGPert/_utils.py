from typing import Literal

import anndata as an
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import LabelEncoder


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


class MLPClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        device: str = 'cpu',
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 10,
    ):
        super(MLPClassifier, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # Define layers
        self.layers = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList() if use_batchnorm else None
        self.dropouts = torch.nn.ModuleList() if dropout > 0 else None

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            self.batchnorms.append(torch.nn.BatchNorm1d(hidden_dim))
        if dropout > 0:
            self.dropouts.append(torch.nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                self.batchnorms.append(torch.nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                self.dropouts.append(torch.nn.Dropout(dropout))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, n_classes))
        self.to(device)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorms:
                x = self.batchnorms[i](x)
            if self.dropouts:
                x = self.dropouts[i](x)
        return self.layers[-1](x)  # Final layer without activation (for logits)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.train()

        # Convert data to tensors and load to device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                f'Epoch [{epoch+1}/{self.epochs}], \
                    Loss: {epoch_loss / len(train_loader)}'
            )


class CellEmbeddingClassifier:
    def __init__(
        self,
        anndata: an.AnnData,
        obsm_key: str,
        perturbation_key: str,
        classifier: Literal['mlp', 'dt', 'rf', 'svm'] = 'mlp',
        classif_params: dict = {},
        test_size: float = 0.2,
    ):
        from sklearn.model_selection import train_test_split

        self.anndata = anndata
        self.obsm_key = obsm_key
        self.perturbation_key = perturbation_key

        # Encode string labels to integers
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(
            self.anndata.obs[self.perturbation_key].values
        )

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.anndata.obsm[self.obsm_key],
            encoded_labels,
            test_size=test_size,
            stratify=encoded_labels,
            random_state=42,
        )

        if classifier == 'mlp':
            self.classifier = MLPClassifier(
                input_dim=self.X_train.shape[1],
                n_classes=len(set(self.y_train)),
                **classif_params,
            )
        elif classifier == 'dt':  # Decision Tree
            from sklearn.tree import DecisionTreeClassifier

            self.classifier = DecisionTreeClassifier(**classif_params)
        elif classifier == 'rf':  # Random Forest
            from sklearn.ensemble import RandomForestClassifier

            self.classifier = RandomForestClassifier(**classif_params)
        elif classifier == 'svm':  # Support Vector Machine
            from sklearn.svm import SVC

            self.classifier = SVC(**classif_params)
        else:
            raise ValueError(
                "Invalid classifier. Choose from 'mlp', 'dt', 'rf', or 'svm'."
            )

    def train(self):
        if isinstance(self.classifier, MLPClassifier):
            self.classifier.fit(self.X_train, self.y_train)
        else:
            self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        from sklearn.metrics import classification_report

        if isinstance(self.classifier, MLPClassifier):
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(
                self.classifier.device
            )
            y_pred = self.classifier.predict(X_test_tensor).cpu().numpy()
        else:
            y_pred = self.classifier.predict(self.X_test)

        # Decode predictions and true labels to original string labels
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(self.y_test)

        report = classification_report(y_test_labels, y_pred_labels)
        print(report)
        return report
