"""
This module initializes the package by importing key functions and classes from submodules and defining the public API. # noqa: E501 .

Imports:
    compute_metrics: Function to compute various metrics.
    get_metric_confusion_matrices: Function to get confusion matrices for metrics.
    plot_metrics: Function to plot the computed metrics.
    CellEmbeddingClassifier: Class for the CellEmbeddingClassifier model.
    scFoundGPert: Class for the scFoundGPert model.

__all__:
    List of public objects of this module, as interpreted by `import *`.
    - 'scFoundGPert'
    - 'compute_metrics'
    - 'get_metric_confusion_matrices'
    - 'plot_metrics'
    - 'CellEmbeddingClassifier'
"""

from ._utils import (
    CellEmbeddingClassifier,
    compute_metrics,
    get_metric_confusion_matrices,
    plot_metrics,
)
from .model import scFoundGPert

__all__ = [
    'scFoundGPert',
    'compute_metrics',
    'get_metric_confusion_matrices',
    'plot_metrics',
    'CellEmbeddingClassifier',
]
