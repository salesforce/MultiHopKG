# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:04 2024

@author: Eduin Hernandez

Summary:
This script defines the `ANN_IndexMan` class, which facilitates both exact and approximate nearest neighbor (ANN) search 
on embeddings from Freebase and Wikidata data. The class uses the FAISS library to manage embeddings and perform similarity 
searches, either through an exact L2 index or an approximate IVF index. Users can specify whether to perform exact or approximate 
computations, search for nearest neighbors, map search results to properties in a DataFrame, and calculate hit@N scores to evaluate 
search accuracy.

Core functionalities:
- **Initialization (`__init__`)**: Loads data, embeddings, and sets up the FAISS index. Allows for exact or approximate 
  index creation with clustering (IVF) for scalability.
  
- **search**: Takes a set of target embeddings and retrieves the top-K nearest neighbors from the index, returning distances and indices.

- **index2data**: Maps a 2D array of search result indices to property values in a specified DataFrame column, with options to limit the 
  number of mapped results per query.

- **calculate_hits_at_n**: Calculates the hit@N score, a metric that measures the fraction of queries where the correct index is found 
  within the top-N nearest neighbors.
"""

import numpy as np

import torch
import faiss
import pdb
from typing import Tuple

class ANN_IndexMan():
    """
    A class for managing approximate nearest neighbor (ANN) search and exact nearest neighbor search for
    embeddings from Freebase and Wikidata data. The class can initialize an index with either exact or 
    approximate search capabilities, conduct similarity searches, map search results to data properties, 
    and calculate hit@N scores.
    
    Attributes:
        data_df (pd.DataFrame): DataFrame loaded from the specified data path, containing the properties for each embedding.
        embedding_vectors (np.ndarray): Array of embedding vectors loaded from the specified embedding path.
        nlist (int): Number of clusters to use in the IVF index for approximate search.
        index (faiss.Index): The FAISS index for performing similarity searches.
    """

    def __init__(self, embeddings_weigths:torch.Tensor , exact_computation: bool = True, nlist = 100):
        """
        Initializes the ANN_IndexMan class, loading data, creating embeddings, and setting up the FAISS index.
        
        Args:
            embeddings_path (str): Path to the embedding CSV file, containing embedding vectors.
            exact_computation (bool): If True, initializes an exact L2 search index; if False, initializes an approximate IVF index.
            nlist (int): Number of clusters for the IVF index if exact_computation is False.
        """
        # Ensure that vectors are in float32 for the sake of faise
        self.embedding_vectors = embeddings_weigths.detach().numpy().astype(np.float32)
        nlist = nlist
        
        if exact_computation:
            self.index = faiss.IndexFlatL2(self.embedding_vectors.shape[1])  # L2 distance (Euclidean distance)
            self.index.add(self.embedding_vectors)  # type: ignore
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.embedding_vectors.shape[1]),
                                           self.embedding_vectors.shape[1],
                                           nlist)

            # Train the index (necessary for IVF indices)
            self.index.train(self.embedding_vectors) # type: ignore

            # Add vectors to the index
            self.index.add(self.embedding_vectors) # type: ignore

    
    def search(self, target_embeddings: torch.Tensor, topk) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches for the top-K nearest neighbors for a given set of target embeddings.
        
        Args:
            target_embeddings (np.ndarray): Array of embeddings to search against the index.
            topk (int): Number of nearest neighbors to retrieve.
        
        Returns:
            resulting_embeddings (np.ndarray): entity embeddings retrieved using ANN
        """
        assert len(target_embeddings.shape) == 2, "Target embeddings must be a 2D array"
        assert isinstance(target_embeddings, torch.Tensor), "Target embeddings must be a torch.Tensor"

        # TODO: Check that we are acutally passing the right shape of input
        distances, indices = self.index.search(target_embeddings, topk) # type: ignore

        # Get the Actual Embeddings her
        resulting_embeddings = self.embedding_vectors[indices.squeeze(), :]

        return resulting_embeddings
    
    def calculate_hits_at_n(self, ground_truth: np.ndarray, indices: np.ndarray, topk: int) -> float:
        assert topk <= indices.shape[1], 'Topk must be smaller or equal than the size of index length'
        """
        Calculates the hit@N score, which is the fraction of queries where the correct index is within the top N nearest neighbors.

        Args:
            ground_truth (np.ndarray): Array of ground truth indices for each query.
            indices (np.ndarray): 2D array of indices returned from a nearest-neighbor search (shape: [num_queries, topk]).
            topk (int): Number of top results to consider for a hit.

        Returns:
            float: The hit@N score.
        """
        hits_at_n = sum([1 for i, gt in enumerate(ground_truth) if gt in indices[i, :topk]])
        hit_at_n_score = hits_at_n / len(ground_truth)
        return hit_at_n_score

