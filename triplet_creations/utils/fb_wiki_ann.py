# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:04 2024

@author: Eduin Hernandez

Summary:
This script defines the `FbWikiANN` class, which facilitates both exact and approximate nearest neighbor (ANN) search 
on embeddings from Freebase and Wikidata data. The class uses the FAISS library to manage embeddings and perform similarity 
searches, either through an exact L2 index or an approximate IVF index. Users can specify whether to perform exact or approximate 
computations, search for nearest neighbors, map search results to properties in a DataFrame, calculate hit@N scores to evaluate 
search accuracy, and retrieve embedding vectors based on specific properties.
"""
import numpy as np
import faiss

from typing import List, Tuple

from utils.basic import load_pandas, load_embeddings

class FbWikiANN():
    """
    A class for managing approximate nearest neighbor (ANN) search and exact nearest neighbor search for
    embeddings from Freebase and Wikidata data. The class can initialize an index with either exact or 
    approximate search capabilities, conduct similarity searches, map search results to data properties, 
    and calculate hit@N scores.
    """
    def __init__(self, data_path:str, embedding_path:str, exact_computation: bool = True, nlist = 100) -> None:
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
        self.embedding_map = dict(zip(embeddings_full['Property'].tolist(), embeddings_full.index))
        self.embedding_vectors = np.array(embeddings_full['Embedding'].tolist())
        self.nlist = nlist
    
        if exact_computation:
            self.index = faiss.IndexFlatL2(self.embedding_vectors.shape[1])  # L2 distance (Euclidean distance)
            
            self.index.add(self.embedding_vectors)  # Add all node embeddings to the index
        else:
            quantizer = faiss.IndexFlatL2(self.embedding_vectors.shape[1])
            self.index = faiss.IndexIVFFlat(quantizer,
                                           self.embedding_vectors.shape[1],
                                           self.nlist,
                                           faiss.METRIC_L2)

            # Train the index (necessary for IVF indices)
            self.index.train(self.embedding_vectors)

            # Add vectors to the index
            self.index.add(self.embedding_vectors)
    
    def search(self, target_embeddings: np.ndarray, topk) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def index2data(self, indices, column_name = 'Title', max_indices=1) -> List[any]:
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
        """
        Calculates the hit@N score, which is the fraction of queries where the correct index is within the top N nearest neighbors.

        Args:
            ground_truth (np.ndarray): Array of ground truth indices for each query.
            indices (np.ndarray): 2D array of indices returned from a nearest-neighbor search (shape: [num_queries, topk]).
            topk (int): Number of top results to consider for a hit.

        Returns:
            float: The hit@N score.
        """
        assert topk <= indices.shape[1], 'Topk must be smaller or equal than the size of index length'
        hits_at_n = sum([1 for i, gt in enumerate(ground_truth) if gt in indices[i, :topk]])
        hit_at_n_score = hits_at_n / len(ground_truth)
        return hit_at_n_score
    
    def get_embedding_vector(self, property_val: str) -> np.ndarray:
        """
        Retrieves the embedding vector corresponding to a specific property value.
        
        Args:
            property_val (str): The property value for which to retrieve the embedding vector.
        
        Returns:
            np.ndarray: The embedding vector for the specified property value, or an empty array if not found.
        """
        idx = self.embedding_map.get(property_val)
        return self.embedding_vectors[idx][None,:] if idx is not None else np.array([], dtype=float)
    
    def get_embedding_vectors(self, properties_list: list) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves embedding vectors for a list of properties, filtering out properties that do not exist in the embedding map.
        
        Args:
            properties_list (list): List of property values for which to retrieve embedding vectors.
        
        Returns:
            tuple: A tuple containing:
                - List of valid properties found in the embedding map.
                - np.ndarray: Array of embedding vectors corresponding to the valid properties.
        """
        properties_list = [property_val for property_val in properties_list if property_val in self.embedding_map.keys()]
        indices = [self.embedding_map.get(property_val) for property_val in properties_list]
        return properties_list, np.array([self.embedding_vectors[idx] for idx in indices])