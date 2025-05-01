"""
Module for searching semantic similarity between code embeddings using Sentence Transformers.

This code is provided strictly for research evaluation purposes only.
Redistribution, modification, or sharing of this code is prohibited.
All rights reserved by the author.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
from typing import Optional, Dict, List, Tuple, Any


class SearchUtils:
    """
    Utility class for searching and ranking similar code snippets based on vector embeddings.

    Supports cosine, dot product, and Euclidean distance as similarity metrics.
    """

    def __init__(
        self,
        removed_embeddings: Optional[torch.Tensor] = None,
        snapshot_embeddings: Optional[Dict] = None,
        val_df: Optional[pd.DataFrame] = None,
        mode: str = "cosine",
        search_metadata: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the search utility with precomputed embeddings and metadata.

        Args:
            removed_embeddings (torch.Tensor): Embeddings of removed code.
            snapshot_embeddings (dict): Embeddings of code snapshots.
            val_df (pd.DataFrame): DataFrame for ground truth validation.
            mode (str): Similarity metric: 'cosine', 'dot', or 'euclidean'.
            search_metadata (dict): Metadata including model name, column, and library info.
        """
        self.removed_embeddings = removed_embeddings
        self.snapshot_embeddings = snapshot_embeddings
        self.val_df = val_df
        self.query_embedding = None
        self.target_embeddings = None
        self.mode = mode
        self.scores = None
        self.model_name = None
        self.column = None
        self.lib = None
        self.__set_similarity_info(search_metadata)

    def __set_similarity_info(self, data: Dict[str, str]) -> None:
        """
        Set model and search context metadata.

        Args:
            data (dict): Dictionary with 'model_name', 'column', and 'lib'.
        """
        self.model_name = data["model_name"]
        self.column = data["column"]
        self.lib = data["lib"]

    def __search_embeddings(self) -> np.ndarray:
        """
        Compute similarity scores between the query and target embeddings.

        Returns:
            np.ndarray: Similarity or distance scores.
        """
        if self.mode == "cosine":
            similarities = util.cos_sim(self.query_embedding, self.target_embeddings)
        elif self.mode == "dot":
            similarities = torch.matmul(self.query_embedding, self.target_embeddings.T)
        elif self.mode == "euclidean":
            similarities = torch.cdist(
                self.query_embedding, self.target_embeddings, p=2
            )
        else:
            raise ValueError("Invalid mode. Choose from 'cosine', 'dot', 'euclidean'")

        return similarities.cpu().numpy()

    def __get_topk(self, k: int) -> Tuple[List[int], List[float]]:
        """
        Get the top-k most similar methods from the similarity scores.

        Args:
            k (int): Number of top results to return.

        Returns:
            Tuple[List[int], List[float]]: Indices and scores of the top-k items.
        """
        if isinstance(self.scores, np.ndarray):
            self.scores = torch.tensor(self.scores)
        top_results = torch.topk(self.scores, k=k, dim=1)
        return (
            top_results.indices.flatten().tolist(),
            top_results.values.flatten().tolist(),
        )

    def store_similarity(
        self, data: List[Dict[str, Any]], col: str, k: int = 100
    ) -> None:
        """
        Store similarity search results to a compressed pickle file.

        Args:
            data (list): List of similarity result dictionaries.
            col (str): Column name associated with the search.
            k (int): Top-k value used in the search.
        """
        head_path = "data/similarity/"
        os.makedirs(head_path, exist_ok=True)

        model_name = self.model_name.replace("/", "_")
        path = os.path.join(
            head_path, f"{model_name}_{self.lib}_{col}_{k}_{self.mode}_sim.pkl"
        )

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def search(self, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a similarity search between removed and snapshot embeddings.

        If previously computed results are available, they are loaded from disk.

        Args:
            k (int): Number of top similar results to return per query.

        Returns:
            List[dict]: List of search results with verification info.
        """
        removed_idx_count = len(self.removed_embeddings[1])
        col = self.column
        model_name = self.model_name.replace("/", "_")
        path = (
            f"../data/similarity/{model_name}_{self.lib}_{col}_{k}_{self.mode}_sim.pkl"
        )

        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        scores = []
        for i in range(removed_idx_count):
            lib = self.removed_embeddings[2]
            removed_id = self.removed_embeddings[1][i]
            self.query_embedding = self.removed_embeddings[0][i].unsqueeze(0)
            snapshot_ids = self.snapshot_embeddings[1]
            self.target_embeddings = self.snapshot_embeddings[0]
            self.scores = self.__search_embeddings()

            val_snapshot_ids = self.val_df[
                self.val_df["removed_method_id"] == removed_id
            ]["snapshot_id"].values

            top_indices, top_scores = self.__get_topk(k=k)
            top_ids = [snapshot_ids[i] for i in top_indices]

            for val_snapshot_id in val_snapshot_ids:
                scores.append(
                    {
                        "top_indices": top_indices,
                        "top_scores": top_scores,
                        "removed_id": removed_id,
                        "library_name": lib,
                        "target_id": val_snapshot_id,
                        "top_ids": top_ids,
                        "verified": val_snapshot_id in top_ids,
                        "column": col,
                        "model_name": model_name,
                        "unique_id": f"{lib}_{removed_id}_{val_snapshot_id}",
                    }
                )

        self.store_similarity(scores, col, k=k)
        return scores
