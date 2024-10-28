import logging
from typing import Union

import anndata as an
import numpy as np
from helical import Geneformer, GeneformerConfig
from helical.models.scgpt.model import scGPT, scGPTConfig

logger = logging.getLogger(__name__)


class scFoundGPert:
    """
    A class to handle single-cell gene perturbation models.

    Attributes:
    ----------
    device : str
        The device to run the model on (e.g., 'cuda' or 'cpu').
    batch_size : int
        The batch size for model processing.
    model_name : str
        The name of the model being used ('scgpt' or 'geneformer').
    model : object
        The instantiated model object (either scGPT or Geneformer).

    Methods:
    -------
    __init__(model: str = "scGPT", device: str = "cuda", batch_size: int = 32)
        Initializes the scFoundGPert class with the specified model,
        device, and batch size.

    process_data(adata: Union[an.AnnData, str], gene_names: str = "index", **kwargs)
        Processes the input data using the specified model.

    get_embeddings(dataset) -> np.array
        Retrieves embeddings from the model for the given dataset.
    """

    def __init__(
        self,
        model: str = 'scGPT',  # scGPT or Geneformer
        device: str = 'cuda',
        batch_size: int = 32,
    ):
        """
        Initializes the scFoundGPert class.

        Parameters:
        ----------
        model : str, optional
            The name of the model to use ('scGPT' or 'Geneformer'). Default is 'scGPT'.
        device : str, optional
            The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        batch_size : int, optional
            The batch size for model processing. Default is 32.
        """
        assert model.lower() in ['scgpt', 'geneformer'], 'Model not supported'

        self.device = device
        self.batch_size = batch_size
        self.model_name = model.lower()

        # Initialize the appropriate model based on the model name
        if self.model_name == 'scgpt':
            config = scGPTConfig(batch_size=batch_size, device=device)
            self.model = scGPT(configurer=config)

        elif self.model_name == 'geneformer':
            config = GeneformerConfig(batch_size=batch_size, device=device)
            self.model = Geneformer(configurer=config)
        else:
            raise ValueError('Model not supported')

        logger.info(f'Model: {self.model_name} initialized')
        logger.info(f'Device: {device}')
        logger.info(f'Batch size: {batch_size}')

        logger.info('Finished initializing model!')

    def process_data(
        self,
        adata: Union[an.AnnData, str],
        gene_names: str = 'index',
        **kwargs,  # Additional model specific arguments
    ):
        """
        Processes the input data using the specified model.

        Parameters:
        ----------
        adata : Union[an.AnnData, str]
            The input data to process.
            Can be an AnnData object or a path to an h5ad file.
        gene_names : str, optional
            The column name in adata.var containing gene names. Default is 'index'.
        **kwargs : dict
            Additional model-specific arguments.

        Returns:
        -------
        processed_data : object
            The processed/tokenized data.
        """
        if isinstance(adata, str):
            adata = an.read_h5ad(
                adata
            )  # Read the data from the file if a path is provided

        # Process the data using the model's process_data method
        processed_data = self.model.process_data(
            adata=adata,
            gene_names=gene_names,
            **kwargs,
        )

        return processed_data

    def get_embeddings(self, dataset) -> np.array:
        """
        Retrieves embeddings from the model for the given dataset.

        Parameters:
        ----------
        dataset : object
            The dataset to retrieve embeddings for.

        Returns:
        -------
        embeddings : np.array
            The embeddings for the dataset.
        """
        return self.model.get_embeddings(dataset)
