"""
Module for searching semantic similarity between code embeddings using Sentence Transformers.

License:
This software is licensed strictly for non-commercial academic and research purposes.
Use is permitted only by individual researchers, students, and educators
affiliated with academic institutions, and only for scholarly work.

Prohibited Uses:
- Commercial use in any form, including but not limited to products, services, or for-profit research.
- Redistribution, sublicensing, or modification without prior written permission.
- Use by or integration into any large language models (LLMs), AI agents or systems, bots, or autonomous software
  whether for training, inference, benchmarking, or any other purpose.

By using this software, you agree to abide by these terms.

(c) 2025 Anush Krishna V (anushkrishnav). All rights reserved.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import util
from typing import Optional, Dict, List, Tuple, Any
import torch.nn.functional as F

from src.db_handler import DBHandler


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
        db_handler: Optional[DBHandler] = None  # <-- Added
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
        self.db = db_handler

    def set_similarity_info(self, data: Dict[str, str]) -> None:
        """
        Set model and search context metadata.

        Args:
            data (dict): Dictionary with 'model_name', 'column', and 'lib'.
        """
        self.model_name = data["model_name"]
        self.column = data["column"]
        self.lib = data["lib"]
    
    def __apply_softmax_scaling(self, scores: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        """
        Apply temperature-scaled softmax to similarity scores to amplify differences.
        Higher temperature = smoother distribution; lower = sharper.
        """
        return F.softmax(scores / temperature, dim=-1)

    def __search_embeddings(self) -> np.ndarray:
        """
        Compute similarity scores between the query and target embeddings.

        Returns:
            np.ndarray: Similarity or distance scores.
        """
        if self.mode == "cosine":
            similarities = util.cos_sim(self.query_embedding, self.target_embeddings)
        elif self.mode == "cosine_soft":
            similarities = util.cos_sim(self.query_embedding, self.target_embeddings)
            similarities = self.__apply_softmax_scaling(similarities, temperature=0.05)
        elif self.mode == "dot":
            similarities = torch.matmul(self.query_embedding, self.target_embeddings.T)
        elif self.mode == "angular":
            similarities = util.pytorch_cos_sim(self.query_embedding, self.target_embeddings)
            similarities = torch.clamp(similarities, -1.0 + 1e-5, 1.0 - 1e-5)
            similarities = 1 - torch.acos(similarities) / np.pi
        elif self.mode == "euclidean":
            similarities = torch.cdist(
                self.query_embedding, self.target_embeddings, p=2
            )
        else:
            raise ValueError("Invalid mode. Choose from 'cosine', 'dot', 'euclidean'")

        return similarities.cpu()

    


    def __get_topk(self, k: int) -> Tuple[List[int], List[float]]:
        """
        Get the top-k most similar methods from the similarity scores.

        Args:
            k (int): Number of top results to return.

        Returns:
            Tuple[List[int], List[float]]: Indices and scores of the top-k items.
        """
        # scores are of Shape (Q, T) so for each query we need to get top k and return 
        # the indices and scores as a (Q,K) array
        if self.mode == "euclidean":
            topk = torch.topk(self.scores, k=k, dim=1, largest=False)
        else:
            topk = torch.topk(self.scores, k=k, dim=1)
        return topk.indices, topk.values
    



 

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


    def process_search(self, topk: list =[], removed_ids: Optional[List[int]] = None,snapshot_ids: Optional[List[int]] =None, lib_name: Optional[str] = None) -> List[Dict[str, Any]]:
        k = max(topk) if topk else 1
        top_indices, top_scores = self.__get_topk(k=k)
        results = []
        for i, removed_id in enumerate(removed_ids):
            val_snapshot_ids = self.val_df[self.val_df["removed_method_id"] == removed_id]["snapshot_id"].values

            topk_indices_i = top_indices[i]  # list of k indices
            topk_scores_i = top_scores[i]

            for val_snapshot_id in val_snapshot_ids:
                base_data = {
                    "removed_id": removed_id,
                    "library_name": lib_name,
                    "target_id": val_snapshot_id,
                    "column": self.column,
                    "model_name": self.model_name,
                    "mode": self.mode,
                    
                }
                for k_val in topk:
                    topk_indices_i_kval = topk_indices_i[:k_val]
                    topk_scores_i_kval = topk_scores_i[:k_val]
                    snap_j = [snapshot_ids[j] for j in topk_indices_i_kval]
                    results.append(
                        {
                            **base_data,
                            "k": k_val,
                            "top_indices": topk_indices_i_kval,
                            "top_scores": topk_scores_i_kval,
                            "top_ids": snap_j,
                            "verified": val_snapshot_id in snap_j,
                            "unique_id": f"{lib_name}_{removed_id}_{val_snapshot_id}_{k_val}",
                        }
                    )
        return results



    def search(self, topk: list =[], column: str = '', all_modes=False) -> List[Dict[str, Any]]:

        self.query_embedding, removed_ids, lib_name = self.removed_embeddings[column]
        self.target_embeddings, snapshot_ids, _ = self.snapshot_embeddings[column]
        results = []
        if all_modes:
            modes = ["cosine", "cosine_soft", "dot", "angular", "euclidean"]
            for mode in modes:
                self.mode = mode
                self.scores = self.__search_embeddings() # Shape (Qquery embeddings, N target embeddings)
                results.extend(self.process_search(topk, removed_ids, snapshot_ids, lib_name))
        else:
            self.scores = self.__search_embeddings()
            results.extend(self.process_search(topk, removed_ids, snapshot_ids, lib_name))
        return results
                # print(results[-1])
        
    def weighted_search(self, topk: list = [], weighted_schema: dict = {}, all_modes=False) -> List[Dict[str, Any]]:
        """
        Perform weighted search across multiple modalities.
        Example weighted_schema: {"name": 0.5, "code": 0.5}
        """
        results = []

        if all_modes:
            modes = ["cosine", "cosine_soft", "dot", "angular", "euclidean"]
        else:
            modes = [self.mode]

        for mode in modes:
            self.mode = mode
            scores_sum = []

            for col, weight in weighted_schema.items():
                self.query_embedding, removed_ids, lib_name = self.removed_embeddings[col]
                self.target_embeddings, snapshot_ids, _ = self.snapshot_embeddings[col]
                scores = self.__search_embeddings()
                scores_sum.append(scores * weight)

            # Aggregate weighted scores
            self.scores = torch.sum(torch.stack(scores_sum), dim=0)

            # Process combined results
            results.extend(self.process_search(topk, removed_ids, snapshot_ids, lib_name))

        return results
