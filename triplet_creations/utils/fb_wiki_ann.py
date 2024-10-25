# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:04 2024

@author: Eduin Hernandez

Summary:
This script defines the `FbWikiANN` class, which facilitates both exact and approximate nearest neighbor (ANN) search 
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
from utils.basic import load_pandas, load_embeddings

import numpy as np

import faiss

class FbWikiANN():
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
    
    Methods:
        __init__(data_path, embedding_path, exact_computation=True, nlist=100):
            Initializes the ANN search index and loads embeddings.
    
        search(target_embeddings, topk):
            Searches for the top-K nearest neighbors of the target embeddings in the index.
    
        index2data(indices, column_name='Title', max_indices=1) -> list:
            Maps search result indices to values in a specified column of `data_df`, limiting each result to `max_indices`.
    
        calculate_hits_at_n(ground_truth, indices, topk) -> float:
            Calculates the hit@N score, which measures how often the correct index appears within the top-N results.
    """
    def __init__(self, data_path:str, embedding_path:str, exact_computation: bool = True, nlist = 100):
        """
        Initializes the FbWikiANN class, loading data, creating embeddings, and setting up the FAISS index.
        
        Args:
            data_path (str): Path to the data CSV file, containing properties and metadata for embeddings.
            embedding_path (str): Path to the embedding CSV file, containing embedding vectors.
            exact_computation (bool): If True, initializes an exact L2 search index; if False, initializes an approximate IVF index.
            nlist (int): Number of clusters for the IVF index if exact_computation is False.
        """
        self.data_df = load_pandas(data_path)
        
        embeddings_full = load_embeddings(embedding_path)
        self.embedding_vectors = np.array(embeddings_full['Embedding'].tolist())
        self.nlist = nlist
        
        if exact_computation:
            self.index = faiss.IndexFlatL2(self.embedding_vectors.shape[1])  # L2 distance (Euclidean distance)
            
            self.index.add(self.embedding_vectors)  # Add all node embeddings to the index
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.embedding_vectors.shape[1]),
                                           self.embedding_vectors.shape[1],
                                           self.nlist)

            # Train the index (necessary for IVF indices)
            self.index.train(self.embedding_vectors)

            # Add vectors to the index
            self.index.add(self.embedding_vectors)
    
    def search(self, target_embeddings: np.ndarray, topk) -> [np.ndarray, np.ndarray]:
        """
        Searches for the top-K nearest neighbors for a given set of target embeddings.
        
        Args:
            target_embeddings (np.ndarray): Array of embeddings to search against the index.
            topk (int): Number of nearest neighbors to retrieve.
        
        Returns:
            tuple: A tuple containing:
                - distances (np.ndarray): The distances to the nearest neighbors.
                - indices (np.ndarray): The indices of the nearest neighbors in the index.
        """
        if target_embeddings.ndim == 1: target_embeddings = target_embeddings[None, :]
        return self.index.search(target_embeddings, topk)
    
    def index2data(self, indices, column_name = 'Title', max_indices=1) -> list:
        """
        Maps a 2D array of indices to values in a specified DataFrame column, limiting the size of each inner list.

        Args:
            indices (np.ndarray): 2D array of indices.
            column_name (str): The column name in `data_df` to map indices to values.
            max_indices (int): The maximum number of indices to map per inner list.

        Returns:
            list: List of lists with mapped values, each inner list containing up to `max_indices` elements.
        """
        assert max_indices <= indices.shape[1], 'Max indices must be smaller or equal than the TopK'
        return [[self.data_df.iloc[i][column_name] for i in row[:max_indices]] for row in indices]
    
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